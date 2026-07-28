"""Kenyan hate-speech triage: the fine-tuned afro-xlmr 3-class classifier.

`score_new` labels each post neither / offensive / hate with the model promoted
by the 2026-07 fine-tune investigation (Opus x3 `r3-mix3-s1337`, on HF
`tom-h-f/kenya-hatespeech-afroxlmr`) and persists the softmax probabilities plus
a `hate_flag` to the R2 `hatespeech/` prefix, incrementally like embeddings and
sentiment labels.

Own prefix, NOT `labels/` or `incitement/`: latest_* dedups on
platform_post_id alone, so a second writer under those prefixes would shadow
their rows (see docs/analysis/data-model.md).

Deploy rule (investigation STATE.md): flag hate on `p_hate >= 0.28`, an explicit
threshold, not argmax; the taxonomy pushes most coded menace to `offensive`, so
the argmax hate class is deliberately conservative.

    from kma.db import connect
    from kma.hatespeech import score_new
    con = connect()
    score_new(con, limit=500)                 # incremental, persisted

    python -m kma.hatespeech --once --limit 20
    python -m kma.hatespeech --backfill --batch-size 256   # drain (GPU)
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import duckdb
import pandas as pd
import pyarrow as pa

from kma.db import BUCKET, connect, hatespeech_source, incitement_source, posts_source

MODEL = os.getenv("HATESPEECH_MODEL", "tom-h-f/kenya-hatespeech-afroxlmr")
HATE_THRESHOLD = float(os.getenv("HATESPEECH_THRESHOLD", "0.28"))
LABELS = ["neither", "offensive", "hate"]
MAX_LENGTH = 128  # matches the fine-tune / 04_infer inference length

_MODEL: dict[str, object] = {}


def _device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _empty_cache() -> None:
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load():
    if "model" not in _MODEL:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        dev = _device()
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        model.to(dev).eval()
        _MODEL.update(tokenizer=tok, model=model, device=dev)
    return _MODEL["tokenizer"], _MODEL["model"], _MODEL["device"]


def _predict(texts: list[str], batch_size: int = 64) -> pd.DataFrame:
    """Softmax over the 3 classes for each text. Falls back to CPU for any batch
    that OOMs on the accelerator (mirrors classify._run)."""
    import torch

    tok, model, dev = _load()
    probs: list = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tok(
                chunk, truncation=True, max_length=MAX_LENGTH,
                padding=True, return_tensors="pt",
            )
            try:
                out = model(**enc.to(dev)).logits
            except RuntimeError as err:
                if "out of memory" not in str(err).lower():
                    raise
                _empty_cache()
                model.to("cpu")
                out = model(**enc.to("cpu")).logits
                model.to(dev)
            probs.append(torch.softmax(out.float(), dim=-1).cpu())
            _empty_cache()
    p = torch.cat(probs).numpy()
    label_id = p.argmax(axis=1)
    return pd.DataFrame(
        {
            "label": [LABELS[i] for i in label_id],
            "p_neither": p[:, 0],
            "p_offensive": p[:, 1],
            "p_hate": p[:, 2],
        }
    )


def _scored_ids(con: duckdb.DuckDBPyConnection, platform: str) -> set[str]:
    try:
        rel = con.sql(
            f"SELECT DISTINCT platform_post_id FROM {hatespeech_source(platform)}"
        )
    except duckdb.Error:
        return set()
    return set(rel.df()["platform_post_id"].tolist())


def _pending(
    con: duckdb.DuckDBPyConnection, platform: str, limit: int | None
) -> pd.DataFrame:
    df = con.sql(
        f"""
        SELECT platform_post_id, text FROM (
            SELECT * FROM {posts_source(platform)}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
            ) = 1
        )
        WHERE text IS NOT NULL AND length(trim(text)) > 0
        """
    ).df()
    df = df[~df["platform_post_id"].isin(_scored_ids(con, platform))]
    return df.head(limit) if limit else df


MEASURE_COLUMNS = (
    "domain",
    "in_kenya_scope",
    "lexicon_hits",
    "coded_suspect",
    "explicit_toxic",
    "flagged",
)

_NLI_COLUMNS = (
    "dehumanisation_score",
    "violence_call_score",
    "othering_score",
    "political_criticism_score",
)


def _nli_for(con: duckdb.DuckDBPyConnection, platform: str) -> pd.DataFrame:
    """Latest incitement NLI scores per post. Returns an empty frame when the
    prefix does not exist yet, which leaves `coded_suspect` false everywhere -
    the correct conservative default, since a lexicon hit alone is not
    evidence."""
    cols = ["platform_post_id", *_NLI_COLUMNS]
    try:
        return con.sql(
            f"""
            SELECT {", ".join(cols)} FROM (
                SELECT * FROM {incitement_source(platform)}
                QUALIFY row_number() OVER (
                    PARTITION BY platform_post_id ORDER BY scored_at DESC
                ) = 1
            )
            """
        ).df()
    except duckdb.Error:
        return pd.DataFrame(columns=cols)


def _measure_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Run `kma.measure` and rename to the persisted storage contract.

    Delegates every rule to `measure` so the persisted columns cannot drift
    from the read-time ones."""
    from kma.measure import attach_measurement_columns

    out = attach_measurement_columns(df)
    out["lexicon_hits"] = out["lexicon_hits_live"]
    # Operational OR - what an automated consumer should act on. `coded_suspect`
    # is already false when NLI is missing, so this never over-flags on
    # incomplete enrichment.
    out["flagged"] = out["hate_flag"].fillna(False).astype(bool) | out["coded_suspect"]
    return out


def attach_persisted_measure(
    scored: pd.DataFrame, nli: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Measurement columns for a scored batch. `scored` needs
    `platform_post_id`, `text`, `label`, `hate_flag`."""
    df = scored.copy()
    if nli is not None and len(nli):
        df = df.merge(nli, on="platform_post_id", how="left")
    return _measure_frame(df)


def _hate_table(measured: pd.DataFrame, now: datetime) -> pa.Table:
    return pa.table(
        {
            "platform_post_id": measured["platform_post_id"].tolist(),
            "label": measured["label"].tolist(),
            "p_neither": [float(v) for v in measured["p_neither"]],
            "p_offensive": [float(v) for v in measured["p_offensive"]],
            "p_hate": [float(v) for v in measured["p_hate"]],
            "hate_flag": measured["hate_flag"].fillna(False).astype(bool).tolist(),
            "domain": measured["domain"].tolist(),
            "in_kenya_scope": measured["in_kenya_scope"].astype(bool).tolist(),
            "lexicon_hits": [list(h) for h in measured["lexicon_hits"]],
            "coded_suspect": measured["coded_suspect"].astype(bool).tolist(),
            "explicit_toxic": measured["explicit_toxic"].astype(bool).tolist(),
            "flagged": measured["flagged"].astype(bool).tolist(),
            "model": [MODEL] * len(measured),
            "scored_at": [now] * len(measured),
        }
    )


def _write_run(
    con: duckdb.DuckDBPyConnection, table: pa.Table, platform: str, now: datetime
) -> str:
    key = (
        f"hatespeech/platform={platform}"
        f"/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.parquet"
    )
    con.register("_hate_buf", table)
    try:
        con.execute(
            f"COPY _hate_buf TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)"
        )
    finally:
        con.unregister("_hate_buf")
    return key


def score_new(
    con: duckdb.DuckDBPyConnection,
    platform: str = "x",
    limit: int | None = None,
    batch_size: int = 64,
) -> int:
    """Classify unscored posts; persist one Parquet run to R2 `hatespeech/`.

    Measurement columns (`domain`, `coded_suspect`, `flagged`, ...) are written
    alongside the model scores so automated consumers - the collector's target
    selection above all - can read them in pure SQL, without re-running the
    lexicon scan or re-joining the NLI. Returns the number scored."""
    df = _pending(con, platform, limit)
    if df.empty:
        return 0
    nli = _nli_for(con, platform)
    if len(nli):
        df = df.merge(nli, on="platform_post_id", how="left")
    scored = _predict(df["text"].astype(str).tolist(), batch_size=batch_size)
    now = datetime.now(timezone.utc)
    measured = _measure_frame(
        df.assign(
            label=scored["label"].tolist(),
            p_neither=scored["p_neither"].tolist(),
            p_offensive=scored["p_offensive"].tolist(),
            p_hate=scored["p_hate"].tolist(),
            hate_flag=[bool(v >= HATE_THRESHOLD) for v in scored["p_hate"]],
        )
    )
    _write_run(con, _hate_table(measured, now), platform, now)
    return len(df)


def refresh_measure(
    con: duckdb.DuckDBPyConnection,
    platform: str = "x",
    limit: int | None = None,
    chunk: int = 20_000,
) -> int:
    """Recompute measurement columns for already-scored rows, on CPU.

    Model scores are reused verbatim - afro-xlmr never runs - so this is the
    cheap way to bring rows written before the measure columns existed (or
    under older `_CODED_MENACE_MIN` gates) up to the current contract. Rows are
    rewritten as new runs; `latest_hatespeech` then picks them up by
    `scored_at`. Returns the number rewritten."""
    try:
        present = set(con.sql(f"SELECT * FROM {hatespeech_source(platform)} LIMIT 0").columns)
    except duckdb.Error:
        return 0
    # Before the first refresh no run carries the columns at all, so an
    # `IS NULL` predicate cannot even bind - every row needs the rewrite.
    stale = (
        " OR ".join(f"lh.{c} IS NULL" for c in MEASURE_COLUMNS)
        if set(MEASURE_COLUMNS) <= present
        else "TRUE"
    )
    pending = con.sql(
        f"""
        WITH lh AS (
            SELECT * FROM {hatespeech_source(platform)}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY scored_at DESC
            ) = 1
        ), lp AS (
            SELECT platform_post_id, text FROM (
                SELECT * FROM {posts_source(platform)}
                QUALIFY row_number() OVER (
                    PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
                ) = 1
            )
        )
        SELECT lh.platform_post_id, lp.text, lh.label,
               lh.p_neither, lh.p_offensive, lh.p_hate, lh.hate_flag
        FROM lh JOIN lp USING (platform_post_id)
        WHERE {stale}
        {f"LIMIT {int(limit)}" if limit else ""}
        """
    ).df()
    if pending.empty:
        return 0
    nli = _nli_for(con, platform)
    total = 0
    for start in range(0, len(pending), chunk):
        batch = pending.iloc[start : start + chunk].copy()
        if len(nli):
            batch = batch.merge(nli, on="platform_post_id", how="left")
        now = datetime.now(timezone.utc)
        _write_run(con, _hate_table(_measure_frame(batch), now), platform, now)
        total += len(batch)
        print(f"refresh_measure: rewrote {len(batch)} (running total {total})")
    return total


def backfill(
    chunk: int = 20_000, batch_size: int = 256, platform: str = "x"
) -> int:
    """Drain the whole corpus: score_new in bounded runs until nothing pends.
    Resumable - each pass skips already-scored ids.

    `chunk` is the pending cap per run (one R2 Parquet file, and one full-corpus
    anti-join scan); `batch_size` is the GPU inference batch. They are decoupled
    on purpose: a small chunk would re-scan the whole ~130k-post corpus once per
    tiny batch (hours of pure scanning), so keep chunk large. Returns total
    scored."""
    con = connect()
    total = 0
    while True:
        n = score_new(con, platform=platform, limit=chunk, batch_size=batch_size)
        if n == 0:
            break
        total += n
        print(f"backfill: scored {n} (running total {total})")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Score posts for hate speech.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="single bounded pass")
    mode.add_argument("--backfill", action="store_true", help="drain the corpus")
    mode.add_argument(
        "--refresh-measure",
        action="store_true",
        help="recompute measure columns on already-scored rows (CPU, no model)",
    )
    ap.add_argument("--limit", type=int, default=None, help="max posts (once mode)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--platform", default="x")
    args = ap.parse_args()
    if args.backfill:
        total = backfill(batch_size=args.batch_size, platform=args.platform)
        print(f"hate: backfilled {total}")
    elif args.refresh_measure:
        total = refresh_measure(connect(), platform=args.platform, limit=args.limit)
        print(f"hate: refreshed measure columns on {total}")
    else:
        con = connect()
        n = score_new(con, args.platform, args.limit, args.batch_size)
        print(f"hate: scored {n}")


if __name__ == "__main__":
    main()

"""Sentiment / emotion / stance classification (local encoder models).

Sentiment + emotion are persisted to the R2 `labels/` prefix (incremental, like
embeddings) since transformer inference is expensive. Stance is target-based and
combinatorial, so it is computed on demand.

    from kma.db import connect
    from kma.classify import classify_new, stance
    con = connect()
    classify_new(con, limit=500)          # sentiment + emotion, persisted
    stance(con, "Ruto", limit=100)        # zero-shot for/against, live
"""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pyarrow as pa

from kma.db import BUCKET, labels_source, posts_source

SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
EMOTION_MODEL = "MilaNLProc/xlm-emo-t"
STANCE_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

MAX_LENGTH = 256  # tweets are short; hard cap keeps MPS batches small
_PIPES: dict[tuple[str, str], object] = {}


def _device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return -1


def _empty_cache() -> None:
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _pipe(task: str, model: str):
    key = (task, model)
    if key not in _PIPES:
        from transformers import pipeline

        _PIPES[key] = pipeline(
            task, model=model, truncation=True, max_length=MAX_LENGTH, device=_device()
        )
    return _PIPES[key]


def _run(pipe, texts: list[str], batch_size: int, **call_kwargs) -> list:
    """Run a pipeline over `texts` in chunks, clearing the MPS cache between them.
    Falls back to CPU for any chunk that still OOMs on the accelerator."""
    out: list = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        try:
            out.extend(pipe(chunk, batch_size=batch_size, **call_kwargs))
        except RuntimeError as err:
            if "out of memory" not in str(err).lower():
                raise
            _empty_cache()
            prev = pipe.model.device
            pipe.model.to("cpu")
            pipe.device = pipe.model.device
            out.extend(pipe(chunk, batch_size=max(1, batch_size // 4), **call_kwargs))
            pipe.model.to(prev)
            pipe.device = pipe.model.device
        _empty_cache()
    return out


def _labeled_ids(con: duckdb.DuckDBPyConnection, platform: str) -> set[str]:
    try:
        rel = con.sql(f"SELECT DISTINCT platform_post_id FROM {labels_source(platform)}")
    except duckdb.Error:
        return set()
    return set(rel.df()["platform_post_id"].tolist())


def _pending(con: duckdb.DuckDBPyConnection, platform: str, limit: int | None):
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
    df = df[~df["platform_post_id"].isin(_labeled_ids(con, platform))]
    return df.head(limit) if limit else df


def classify_new(
    con: duckdb.DuckDBPyConnection,
    platform: str = "x",
    limit: int | None = None,
    batch_size: int = 64,
) -> int:
    """Sentiment + emotion for posts not yet labelled; persist one Parquet run to
    R2 `labels/`. Returns the number labelled."""
    df = _pending(con, platform, limit)
    if df.empty:
        return 0
    texts = df["text"].tolist()
    sent = _run(_pipe("sentiment-analysis", SENTIMENT_MODEL), texts, batch_size)
    emo = _run(_pipe("text-classification", EMOTION_MODEL), texts, batch_size)
    now = datetime.now(timezone.utc)
    table = pa.table(
        {
            "platform_post_id": df["platform_post_id"].tolist(),
            "sentiment": [r["label"] for r in sent],
            "sentiment_score": [float(r["score"]) for r in sent],
            "emotion": [r["label"] for r in emo],
            "emotion_score": [float(r["score"]) for r in emo],
            "labeled_at": [now] * len(df),
        }
    )
    key = f"labels/platform={platform}/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.parquet"
    con.register("_lab_buf", table)
    try:
        con.execute(
            f"COPY _lab_buf TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)"
        )
    finally:
        con.unregister("_lab_buf")
    return len(df)


def stance(
    con: duckdb.DuckDBPyConnection,
    target: str,
    platform: str = "x",
    limit: int = 200,
):
    """Zero-shot stance (supports / opposes / neutral) toward `target` for posts
    mentioning it. Computed live. Returns a DataFrame with `stance`, `stance_score`."""
    df = con.execute(
        f"""
        SELECT platform_post_id, author_handle, text FROM (
            SELECT * FROM {posts_source(platform)}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
            ) = 1
        )
        WHERE text ILIKE '%' || ? || '%'
        LIMIT {limit}
        """,
        [target],
    ).df()
    if df.empty:
        return df
    labels = [f"supports {target}", f"opposes {target}", f"neutral about {target}"]
    res = _run(
        _pipe("zero-shot-classification", STANCE_MODEL),
        df["text"].tolist(),
        batch_size=16,
        candidate_labels=labels,
    )
    if isinstance(res, dict):
        res = [res]
    df["stance"] = [r["labels"][0].split()[0] for r in res]
    df["stance_score"] = [float(r["scores"][0]) for r in res]
    return df

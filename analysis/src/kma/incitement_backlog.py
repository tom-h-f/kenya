"""Backlog and incremental incitement scoring on a GPU box that holds no R2 keys.

`kma.incitement.score_new` scores a few hundred posts per pass on a CPU host. This
module scores everything still pending, in bounded resumable chunks, and writes the
same `incitement/platform=x/dt=<day>/run=<id>.parquet` layout and schema, so
`incitement._pending` and every reader see the rows unchanged.

    python -m kma.incitement_backlog run --manifest m.json --cache DIR --run-id ID

Storage is a `topics_assign.ManifestStore`: short-lived signed urls made by pi0, so
the shared Windows box never holds a long-lived secret.

Order: posts the relevance classifier calls Kenya (`p_kenya >= 0.5`, or the keyword
gate when a post has no score) first, then the rest; newest first inside each.

Idempotent and crash safe. The plan (ordered ids and texts) is written once per run
id and never rebuilt. Chunk i is always plan rows [i*N, (i+1)*N) and has the fixed key
`run=<run_id>-cNNNN.parquet`, so a resumed run overwrites a chunk instead of adding a
second copy. A chunk is a local file renamed into place after the write, then one
PUT, which R2 applies whole. The completion marker sits under `incitement/_backlog/`,
outside every reader's glob.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from kma.incitement import HYPOTHESES, MODEL, nli_scores, scan_text, score_table
from kma.topics_assign import ManifestStore, S3Store, _readback, _write_json, mirror

PLATFORM = "x"
DEFAULT_CHUNK_ROWS = 10_000
DEFAULT_BATCH = 32
VRAM_CEILING_GIB = 7.0
KENYA_THRESHOLD = 0.5
# Pinned to the PREVIOUS relevance model on purpose, and a literal rather than an
# import so this module keeps running on mike-wsl without boto3. The pi0 cron
# (`deploy/pi0/kenya-incitement-cron.sh`) signs URLs for exactly this slug, so the
# two must change together; until pi0 is switched, a post the old model has not
# scored falls back to the keyword gate, which only affects scoring ORDER.
RELEVANCE_MODEL = "kenya-relevance-afroxlmr-2026-10-11"
MAX_LENGTH = 256

EXIT_MORE_WORK = 3
EXIT_OOM = 4


def posts_prefix(platform: str = PLATFORM) -> str:
    return f"posts/platform={platform}/"


def incitement_prefix(platform: str = PLATFORM) -> str:
    return f"incitement/platform={platform}/"


def relevance_prefix(platform: str = PLATFORM, model: str = RELEVANCE_MODEL) -> str:
    return f"relevance/platform={platform}/model={model}/"


def run_day(run_id: str) -> str:
    return f"{run_id[0:4]}-{run_id[4:6]}-{run_id[6:8]}"


def chunk_key(run_id: str, index: int, platform: str = PLATFORM) -> str:
    return f"{incitement_prefix(platform)}dt={run_day(run_id)}/run={run_id}-c{index:04d}.parquet"


def marker_key(run_id: str, platform: str = PLATFORM) -> str:
    return f"incitement/_backlog/platform={platform}/run={run_id}.json"


def _glob(cache: Path, prefix: str, tail: str) -> str:
    return f"{Path(cache) / 'mirror'}/{prefix}{tail}"


def _parquet(glob: str, hive: bool = True) -> str:
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning={str(hive).lower()})"


def build_plan(cache: Path, platform: str = PLATFORM, model: str = RELEVANCE_MODEL) -> pd.DataFrame:
    """Unscored posts with text, Kenya-relevant first, newest first inside each.

    Mirrors `db.pending_posts` (latest row per post, text non-empty, anti-join on the
    scored ids) over the local mirror, and adds the relevance ordering."""
    from kma import measure

    posts = _parquet(_glob(cache, posts_prefix(platform), "type=*/dt=*/run=*.parquet"))
    scored = _parquet(_glob(cache, incitement_prefix(platform), "dt=*/run=*.parquet"))
    rel_glob = _glob(cache, relevance_prefix(platform, model), "dt=*/run=*.parquet")
    mirror_root = Path(cache) / "mirror"
    has_relevance = any((mirror_root / relevance_prefix(platform, model)).glob("dt=*/run=*.parquet"))
    rel_cte = (
        f"""SELECT platform_post_id, p_kenya FROM {_parquet(rel_glob, hive=False)}
            QUALIFY row_number() OVER (PARTITION BY platform_post_id ORDER BY scored_at DESC) = 1"""
        if has_relevance
        else "SELECT NULL::VARCHAR AS platform_post_id, NULL::DOUBLE AS p_kenya WHERE FALSE"
    )
    con = duckdb.connect()
    try:
        has_scored = any((mirror_root / incitement_prefix(platform)).glob("dt=*/run=*.parquet"))
        anti = (
            f"AND NOT EXISTS (SELECT 1 FROM {scored} s WHERE s.platform_post_id = p.platform_post_id)"
            if has_scored
            else ""
        )
        df = con.sql(
            f"""
            WITH rel AS ({rel_cte})
            SELECT p.platform_post_id, p.text, p.collected_at, rel.p_kenya
            FROM (
                SELECT platform_post_id, text, collected_at FROM {posts}
                QUALIFY row_number() OVER (
                    PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
                ) = 1
            ) p
            LEFT JOIN rel ON rel.platform_post_id = p.platform_post_id
            WHERE p.text IS NOT NULL AND length(trim(p.text)) > 0
            {anti}
            """
        ).df()
    finally:
        con.close()
    if df.empty:
        return df.assign(kenya=pd.Series(dtype=bool))[["platform_post_id", "text", "kenya"]]
    gate = df["text"].map(measure.domain_bucket) == "kenya"
    df["kenya"] = np.where(df["p_kenya"].notna(), df["p_kenya"] >= KENYA_THRESHOLD, gate).astype(bool)
    df = df.sort_values(["kenya", "collected_at"], ascending=[False, False], kind="stable")
    return df[["platform_post_id", "text", "kenya"]].reset_index(drop=True)


def load_or_build_plan(cache: Path, run_dir: Path, platform: str, limit: int | None) -> pa.Table:
    """The plan file is written once. A resumed run must see the same plan."""
    path = run_dir / "plan.parquet"
    if path.is_file():
        return pq.read_table(path)
    df = build_plan(cache, platform)
    if limit is not None:
        df = df.head(limit)
    table = pa.Table.from_pandas(df, preserve_index=False)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.rename(path)
    return table


class Scorer:
    """The production NLI pass on a CUDA device, half precision, length-sorted.

    Sorting by text length only changes how much padding each batch carries."""

    def __init__(self, batch: int = DEFAULT_BATCH, dtype: str = "float16", device: int = 0):
        import torch
        from transformers import pipeline

        self.torch = torch
        self.batch = batch
        self.pipe = pipeline(
            "zero-shot-classification",
            model=MODEL,
            truncation=True,
            max_length=MAX_LENGTH,
            device=device,
            dtype=getattr(torch, dtype),
        )

    def peak_gib(self) -> float:
        return self.torch.cuda.max_memory_allocated() / 2**30

    def score(self, texts: list[str]) -> list[dict[str, float]]:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        while True:
            try:
                got = nli_scores(self.pipe, [texts[i] for i in order], self.batch)
                break
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if self.batch <= 8:
                    raise
                self.batch //= 2
        out: list[dict[str, float]] = [{}] * len(texts)
        for pos, i in enumerate(order):
            out[i] = got[pos]
        if self.peak_gib() > VRAM_CEILING_GIB and self.batch > 8:
            self.batch //= 2
        return out


def chunk_frame(plan: pa.Table, index: int, rows: int) -> pd.DataFrame:
    df = plan.slice(index * rows, rows).select(["platform_post_id", "text"]).to_pandas()
    scans = df["text"].map(scan_text)
    df["lexicon_hits"] = [h for h, _ in scans]
    df["lexicon_categories"] = [c for _, c in scans]
    return df


def run(
    store,
    cache: Path,
    *,
    run_id: str,
    platform: str = PLATFORM,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    batch: int = DEFAULT_BATCH,
    limit: int | None = None,
    dry_run: bool = False,
    status_file: Path | None = None,
    scorer=None,
) -> dict:
    """Score the plan chunk by chunk. Returns the marker dict that was written."""
    t0 = time.time()
    cache = Path(cache)
    run_dir = cache / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    def status(stage: str, **extra) -> None:
        print(f"[{time.time() - t0:8.1f}s] {stage} {extra or ''}", flush=True)
        if status_file:
            _write_json(status_file, {"stage": stage, "run_id": run_id, "elapsed_s": time.time() - t0,
                                      "updated_at": datetime.now(timezone.utc).isoformat(), **extra})

    status("mirror")
    if not (run_dir / "plan.parquet").is_file():
        for prefix in (posts_prefix(platform), incitement_prefix(platform), relevance_prefix(platform)):
            mirror(store, prefix, cache / "mirror")
    plan = load_or_build_plan(cache, run_dir, platform, limit)
    n_chunks = -(-plan.num_rows // chunk_rows)
    kenya = int(pc.sum(plan.column("kenya")).as_py() or 0) if plan.num_rows else 0
    status("plan", pending=plan.num_rows, kenya_first=kenya, chunks=n_chunks)

    puts = set(store.put_keys()) if hasattr(store, "put_keys") else set()
    if not dry_run and puts:
        need = {chunk_key(run_id, i, platform) for i in range(n_chunks)}
        short = need - puts
        if short:
            status("manifest_short", missing_put_urls=len(short))

    if scorer is None and n_chunks:
        scorer = Scorer(batch=batch)
    chunks: list[dict] = []
    uploaded_all = True
    for i in range(n_chunks):
        key = chunk_key(run_id, i, platform)
        local = run_dir / f"c{i:04d}.parquet"
        flag = run_dir / f"c{i:04d}.uploaded"
        if not local.is_file():
            df = chunk_frame(plan, i, chunk_rows)
            by_label = scorer.score(df["text"].tolist())
            table = score_table(df, by_label, datetime.now(timezone.utc))
            tmp = local.with_name(local.name + ".tmp")
            pq.write_table(table, tmp, compression="zstd")
            tmp.rename(local)
        rows = pq.ParquetFile(local).metadata.num_rows
        if not dry_run and not flag.is_file():
            if puts and key not in puts:
                uploaded_all = False
                status("no_put_url", chunk=i)
                break
            store.put(key, local)
            flag.touch()
        chunks.append({"key": key, "rows": rows, "bytes": local.stat().st_size})
        status("chunk", chunk=i, of=n_chunks, rows=rows,
               scored=sum(c["rows"] for c in chunks),
               peak_vram_gib=round(scorer.peak_gib(), 2), batch=scorer.batch,
               rate_posts_s=round(sum(c["rows"] for c in chunks) / max(time.time() - t0, 1e-9), 1))

    scored_rows = sum(c["rows"] for c in chunks)
    readback = _readback(store, chunks, cache, dry_run)
    complete = uploaded_all and scored_rows == plan.num_rows
    marker = {
        "status": "complete" if complete else "partial",
        "run_id": run_id,
        "platform": platform,
        "model": MODEL,
        "hypotheses": list(HYPOTHESES),
        "pending_at_start": plan.num_rows,
        "kenya_first": kenya,
        "scored": scored_rows,
        "chunks": chunks,
        "readback_rows": readback["rows"],
        "readback_distinct": readback["distinct"],
        "dry_run": dry_run,
        "limit": limit,
        "batch": getattr(scorer, "batch", None),
        "peak_vram_gib": round(scorer.peak_gib(), 3) if scorer is not None else None,
        "elapsed_s": round(time.time() - t0, 1),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    mpath = run_dir / "marker.json"
    _write_json(mpath, marker)
    mk = marker_key(run_id, platform)
    if not dry_run and (not puts or mk in puts):
        store.put(mk, mpath)
    status("done", **{k: marker[k] for k in ("status", "scored", "pending_at_start", "peak_vram_gib")})
    return marker


def cmd_run(args) -> int:
    import torch

    store = ManifestStore.load(Path(args.manifest)) if args.manifest else S3Store()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        marker = run(
            store, Path(args.cache), run_id=run_id, platform=args.platform,
            chunk_rows=args.chunk_rows, batch=args.batch, limit=args.limit, dry_run=args.dry_run,
            status_file=Path(args.status_file) if args.status_file else None,
        )
    except torch.cuda.OutOfMemoryError:
        print("GPU out of memory at the smallest batch", file=sys.stderr)
        return EXIT_OOM
    return 0 if marker["status"] == "complete" else EXIT_MORE_WORK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--manifest", help="signed-url manifest; omit to use R2_* env keys")
    r.add_argument("--cache", required=True, help="local mirror, plan and chunk files")
    r.add_argument("--run-id")
    r.add_argument("--platform", default=PLATFORM)
    r.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    r.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="NLI pairs per forward pass (4 per post)")
    r.add_argument("--limit", type=int, help="plan only the first N pending posts")
    r.add_argument("--dry-run", action="store_true", help="write local files only, no PUT")
    r.add_argument("--status-file")
    args = ap.parse_args(argv)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())

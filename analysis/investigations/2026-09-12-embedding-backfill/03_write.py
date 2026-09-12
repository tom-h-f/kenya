"""Write the backfilled vectors into the production embeddings prefix.

    cd analysis && uv run python investigations/2026-09-12-embedding-backfill/03_write.py [--dry-run]
    ... --check-remote        # also ask R2 which ids it already has (~9 min)

Same rows as `kma.semantic.embed_new` writes - platform_post_id, model, dim,
embedding, embedded_at - under the same key pattern, in runs of `--chunk` posts,
so every reader picks them up through `db.embeddings_source` with nothing to
change.

Resumable from `out/written.json`, which records the chunks already uploaded.
Two passes died mid-upload (a dropped PUT, then the Mac sleeping on battery
mid-request, which R2 rejects as RequestTimeTooSkewed), and re-reading every
embedded id to work out where to restart costs about nine minutes each time.
`--check-remote` does that read anyway, for a pass that has to be sure; posts
the enrich worker embedded meanwhile are harmless either way, because readers
keep the latest row per post.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from kma import semantic
from kma.db import BUCKET, connect, embeddings_source

HERE = Path(__file__).parent
ATTEMPTS = 3


def load_manifest(path: Path, chunk: int) -> set[int]:
    if not path.exists():
        return set()
    state = json.loads(path.read_text())
    if state.get("chunk") != chunk:
        raise SystemExit(f"{path} was written with --chunk {state.get('chunk')}, not {chunk}")
    return set(state["done"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-remote", action="store_true", help="ask R2 which ids are already embedded")
    args = ap.parse_args()

    out = HERE / "out"
    ids = pd.read_parquet(out / "vector_ids.parquet")["platform_post_id"].astype(str).to_numpy()
    vectors = np.load(out / "vectors.npy", mmap_mode="r")
    if len(ids) != len(vectors) or vectors.shape[1] != semantic.DIM:
        raise SystemExit(f"ids {len(ids):,} vs vectors {vectors.shape}: not the same backfill")

    manifest = out / "written.json"
    done_chunks = load_manifest(manifest, args.chunk)
    starts = [s for s in range(0, len(ids), args.chunk) if s not in done_chunks]

    con = connect()
    slug = semantic._slug(semantic.MODEL)
    skip = set()
    if args.check_remote:
        skip = set(con.sql(f"SELECT DISTINCT platform_post_id FROM {embeddings_source('x', slug)}").df()
                   ["platform_post_id"].astype(str))
    print(f"{len(ids):,} backfilled, {len(done_chunks)} chunks already written, "
          f"{len(starts)} of {(len(ids) + args.chunk - 1) // args.chunk} to go"
          + (f", {len(skip):,} ids already in R2" if args.check_remote else ""))
    if args.dry_run:
        return

    for start in starts:
        part = np.arange(start, min(start + args.chunk, len(ids)))
        if skip:
            part = part[~np.isin(ids[part], list(skip))]
        if len(part) == 0:
            done_chunks.add(start)
            manifest.write_text(json.dumps({"chunk": args.chunk, "done": sorted(done_chunks)}))
            continue
        now = datetime.now(timezone.utc)
        # Built from the numpy buffer rather than a list per row: 50k rows of
        # 768 Python floats is gigabytes of objects. The type stays list<double>,
        # which is what `semantic.embed_new` writes.
        flat = np.asarray(vectors[part], dtype=np.float64).reshape(-1)
        offsets = pa.array(np.arange(len(part) + 1, dtype=np.int32) * semantic.DIM)
        table = pa.table({
            "platform_post_id": ids[part].tolist(),
            "model": [slug] * len(part),
            "dim": [semantic.DIM] * len(part),
            "embedding": pa.ListArray.from_arrays(offsets, pa.array(flat)),
            "embedded_at": [now] * len(part),
        })
        key = f"embeddings/platform=x/model={slug}/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.parquet"
        con.register("_emb_buf", table)
        try:
            for attempt in range(1, ATTEMPTS + 1):
                try:
                    con.execute(f"COPY _emb_buf TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)")
                    break
                except duckdb.Error as exc:
                    if attempt == ATTEMPTS:
                        raise
                    print(f"  retry {attempt}: {str(exc)[:80]}", flush=True)
                    time.sleep(5 * attempt)
        finally:
            con.unregister("_emb_buf")
        done_chunks.add(start)
        manifest.write_text(json.dumps({"chunk": args.chunk, "done": sorted(done_chunks)}))
        print(f"  wrote {len(part):,} -> {key}", flush=True)
        time.sleep(1.1)  # run ids are per second; never two runs in one key


if __name__ == "__main__":
    main()

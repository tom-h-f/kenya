"""Write the backfilled vectors into the production embeddings prefix.

    cd analysis && uv run python investigations/2026-09-12-embedding-backfill/03_write.py [--dry-run]

Same rows as `kma.semantic.embed_new` writes - platform_post_id, model, dim,
embedding, embedded_at - under the same key pattern, in runs of `--chunk` posts,
so every reader picks them up through `db.embeddings_source` with nothing to
change. Posts the enrich worker embedded while the backfill was running are
skipped; if one slips through, readers already keep the latest row per post.
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from kma import semantic
from kma.db import BUCKET, connect, embeddings_source

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = HERE / "out"
    ids = pd.read_parquet(out / "vector_ids.parquet")["platform_post_id"].astype(str).to_numpy()
    vectors = np.load(out / "vectors.npy", mmap_mode="r")
    if len(ids) != len(vectors) or vectors.shape[1] != semantic.DIM:
        raise SystemExit(f"ids {len(ids):,} vs vectors {vectors.shape}: not the same backfill")

    con = connect()
    slug = semantic._slug(semantic.MODEL)
    done = set(con.sql(f"SELECT DISTINCT platform_post_id FROM {embeddings_source('x', slug)}").df()
               ["platform_post_id"].astype(str))
    todo = np.flatnonzero(~np.isin(ids, list(done)))
    print(f"{len(ids):,} backfilled, {len(ids) - len(todo):,} embedded meanwhile, {len(todo):,} to write")
    if args.dry_run:
        return

    for start in range(0, len(todo), args.chunk):
        part = todo[start : start + args.chunk]
        now = datetime.now(timezone.utc)
        table = pa.table({
            "platform_post_id": ids[part].tolist(),
            "model": [slug] * len(part),
            "dim": [semantic.DIM] * len(part),
            "embedding": [np.asarray(vectors[k], dtype=np.float64).tolist() for k in part],
            "embedded_at": [now] * len(part),
        })
        key = f"embeddings/platform=x/model={slug}/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.parquet"
        con.register("_emb_buf", table)
        try:
            con.execute(f"COPY _emb_buf TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)")
        finally:
            con.unregister("_emb_buf")
        print(f"  wrote {len(part):,} -> {key}", flush=True)
        time.sleep(1.1)  # run ids are per second; never two runs in one key


if __name__ == "__main__":
    main()

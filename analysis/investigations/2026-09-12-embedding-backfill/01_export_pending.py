"""Export every post the embeddings prefix has not covered, for encoding off-R2.

    cd analysis && uv run python investigations/2026-09-12-embedding-backfill/01_export_pending.py

Only 62% of text-eligible posts in the 2026-09-05 snapshot had an embedding
(textsim investigation), so the text trace could not see 38% of what people
wrote. The enrich worker embeds with a per-cycle cap and has fallen behind; this
drains the backlog on mike-pc instead, the way `modal_backfill.py` drains the
hate-speech one.

Uses `db.pending_posts` - the enrich worker's own definition of "not yet
embedded" - so the backfill covers exactly the posts production would have.
Also exports `check.parquet`: posts that ARE embedded, with their persisted
vectors, so `02_encode.py` can show the GPU box reproduces production's vectors
before anything is written back.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kma import semantic
from kma.db import connect, embeddings_source, pending_posts, posts_source

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0, help="0 for the whole backlog")
    ap.add_argument("--check", type=int, default=2000, help="embedded posts to re-encode as a check")
    args = ap.parse_args()

    con = connect()
    source = embeddings_source("x", semantic._slug(semantic.MODEL))
    out = HERE / "out"
    out.mkdir(exist_ok=True)

    start = time.perf_counter()
    pending = pending_posts(con, source, "x", args.limit or None)
    pending.to_parquet(out / "pending.parquet")
    print(f"pending: {len(pending):,} posts in {time.perf_counter() - start:.0f}s -> {out / 'pending.parquet'}")

    check = con.sql(
        f"""SELECT e.platform_post_id, e.embedding FROM {source} e
            USING SAMPLE reservoir({int(args.check)} ROWS) REPEATABLE (0)"""
    ).df()
    con.register("_check_ids", check[["platform_post_id"]])
    texts = con.sql(
        f"""SELECT platform_post_id, any_value(text) AS text
            FROM {posts_source('x')} SEMI JOIN _check_ids USING (platform_post_id)
            GROUP BY 1"""
    ).df()
    check = check.merge(texts, on="platform_post_id")
    check["embedding"] = check["embedding"].map(lambda v: np.asarray(v, dtype=np.float32))
    check.to_parquet(out / "check.parquet")
    print(f"check: {len(check):,} embedded posts with text -> {out / 'check.parquet'}")


if __name__ == "__main__":
    main()

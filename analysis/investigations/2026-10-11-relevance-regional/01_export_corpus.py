"""Export every post with text, projected, to one local parquet.

    cd analysis && uv run python investigations/2026-10-11-relevance-regional/01_export_corpus.py

Read-only on R2. One row per post (latest collected copy), only the columns the
regional work needs. The same file feeds the hard-negative search here and the
whole-corpus rescore on mike-pc, so the corpus is read from R2 once.
"""

import argparse
import time
from pathlib import Path

from kma.db import connect, posts_source

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    con = connect()
    con.execute("SET preserve_insertion_order=false")
    target = HERE / "out" / ("corpus_limit.parquet" if args.limit else "corpus.parquet")
    start = time.perf_counter()
    con.execute(
        f"""COPY (
            SELECT platform_post_id, author_id, author_handle, created_at, text, lang, type,
                   mentions, hashtags, place_name, in_reply_to_id, is_repost
            FROM {posts_source('x')}
            WHERE text IS NOT NULL AND length(trim(text)) > 0
            QUALIFY row_number() OVER (PARTITION BY platform_post_id ORDER BY collected_at DESC) = 1
            {f'LIMIT {int(args.limit)}' if args.limit else ''}
        ) TO '{target}' (FORMAT parquet, COMPRESSION zstd)"""
    )
    n = con.sql(f"SELECT count(*) FROM '{target}'").fetchone()[0]
    print(f"{n:,} posts in {time.perf_counter() - start:.0f}s -> {target}")


if __name__ == "__main__":
    main()

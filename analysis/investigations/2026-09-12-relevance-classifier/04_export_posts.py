"""Export every post with text, for scoring off-R2 on the GPU box.

    cd analysis && uv run python investigations/2026-09-12-relevance-classifier/04_export_posts.py

The live corpus, not the pinned snapshot: the score is meant for production
`kenya_share`, which reads whatever has been collected. One row per post, the
latest collected copy, the same dedup every reader applies.
"""

import argparse
import time
from pathlib import Path

from kma.db import connect, posts_source

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0, help="first N posts only, for a timing run")
    args = ap.parse_args()

    con = connect()
    start = time.perf_counter()
    posts = con.sql(
        f"""SELECT platform_post_id, text FROM {posts_source('x')}
            WHERE text IS NOT NULL AND length(trim(text)) > 0
            QUALIFY row_number() OVER (PARTITION BY platform_post_id ORDER BY collected_at DESC) = 1
            {f'LIMIT {int(args.limit)}' if args.limit else ''}"""
    ).df()
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    posts.to_parquet(out / "corpus.parquet")
    print(f"{len(posts):,} posts in {time.perf_counter() - start:.0f}s -> {out / 'corpus.parquet'}")


if __name__ == "__main__":
    main()

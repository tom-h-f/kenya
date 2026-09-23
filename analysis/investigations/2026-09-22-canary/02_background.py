"""Cut a small background sample of the pinned corpus to a local parquet.

    cd analysis && uv run python investigations/2026-09-22-canary/02_background.py OUT_DIR

The detection-strength curve runs locally on this sample because the Modal
workspace is over its spend limit, and the mac is shared: a full-corpus pass
does not fit a ~4 GB budget. One week of the `2026-09-14-deep500` snapshot, in
the canonical coord2 view, plus the persisted text-similarity edges restricted
to accounts in that week.
"""

import resource
import sys
from pathlib import Path

from kma import bench, coord2, coord2_run
from kma.db import BUCKET, connect

SNAPSHOT = "2026-09-14-deep500"
SINCE, UNTIL = "2026-09-07", "2026-09-14"


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    con = connect()
    con.execute("SET memory_limit='2GB'")
    view = coord2.posts_view(bench.pinned_source(bench.load(SNAPSHOT, con=con), "posts"))
    posts = out / "bg_posts.parquet"
    con.execute(
        f"COPY (SELECT * FROM ({view}) WHERE created_at >= TIMESTAMPTZ '{SINCE}' "
        f"AND created_at < TIMESTAMPTZ '{UNTIL}') TO '{posts}' (FORMAT parquet)"
    )
    key = coord2_run.textsim_key(SNAPSHOT, 0.85, overlap=0.10)
    text = out / "bg_text.parquet"
    con.execute(
        f"""COPY (
            SELECT CAST(t.source AS VARCHAR) AS source, CAST(t.target AS VARCHAR) AS target, t.weight
            FROM read_parquet('r2://{BUCKET}/{key}') t
            WHERE CAST(t.source AS VARCHAR) IN (SELECT CAST(user_id AS VARCHAR) FROM '{posts}')
              AND CAST(t.target AS VARCHAR) IN (SELECT CAST(user_id AS VARCHAR) FROM '{posts}')
        ) TO '{text}' (FORMAT parquet)"""
    )
    print(con.sql(f"SELECT count(*), count(DISTINCT user_id), sum(is_retweet::INT) FROM '{posts}'").fetchall())
    print(con.sql(f"SELECT count(*) FROM '{text}'").fetchall())
    print(f"peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

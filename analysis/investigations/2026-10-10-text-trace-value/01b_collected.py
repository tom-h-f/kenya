"""Last collection day of every text-eligible post in a window.

A rolling window is pinned by the day a row was COLLECTED (`dt`), so a "7 day
text window" is the posts collected in the last 7 days, deep-timeline history
included, not the posts created in the last 7 days. Q6 needs both; `rows`
already has created_at, this adds the collection day.

Read-only on R2. Output: out/<window>/collected.parquet (post_id, last_dt).
"""

import argparse
import logging

import pandas as pd

from kma import bench
from kma.db import connect

from common import WINDOW, wdir

log = logging.getLogger("collected")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", default=WINDOW)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    base = wdir(args.window)
    manifest = pd.read_parquet(base / "manifest.parquet")
    source = bench.pinned_source(manifest, "posts")
    con = connect()
    con.execute("SET enable_progress_bar = false")
    df = con.sql(
        f"""SELECT platform_post_id AS post_id, max(dt) AS last_dt, min(dt) AS first_dt
            FROM {source} GROUP BY 1"""
    ).df()
    df.to_parquet(base / "collected.parquet")
    log.info("%d posts, last_dt %s .. %s", len(df), df["last_dt"].min(), df["last_dt"].max())


if __name__ == "__main__":
    main()

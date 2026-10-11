"""Export one window's text rows and four bipartite networks to local parquet.

    cd analysis && uv run python investigations/2026-10-10-text-trace-value/01_export.py \
        [--window daily-2026-10-09-30d]

Read-only on R2. Everything lands under out/<window>/ (untracked): the window
manifest, the text-eligible rows (post_id, user_id, created_at, text, clean),
per-user counts for the text activity floor, and one parquet per bipartite
trace as `coord2_run.build_networks` builds it. Later scripts fuse and rank
from these without touching R2 again.
"""

import argparse
import logging
import time

from kma import bench, coord2, coord2_run
from kma.db import connect

from common import WINDOW, wdir

log = logging.getLogger("export")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", default=WINDOW)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out = wdir(args.window)
    con = connect()
    con.execute("SET enable_progress_bar = false")
    manifest = bench.load(args.window, con=con)
    manifest.to_parquet(out / "manifest.parquet")
    log.info("manifest: %s", manifest.groupby("prefix")["rows"].agg(["size", "sum"]).to_dict())

    view = coord2.posts_view(bench.pinned_source(manifest, "posts"))

    t = time.monotonic()
    counts = con.sql(
        f"""SELECT CAST(user_id AS VARCHAR) AS user_id, count(*) AS n
            FROM ({view}) WHERE NOT coalesce(is_retweet, false) AND text IS NOT NULL
            GROUP BY 1"""
    ).df()
    counts.to_parquet(out / "floor_counts.parquet")
    log.info("floor counts: %d users (%.0fs)", len(counts), time.monotonic() - t)

    t = time.monotonic()
    rows = coord2.text_rows(con, view)
    rows = rows.assign(user_id=rows["user_id"].astype(str))
    rows.to_parquet(out / "rows.parquet")
    log.info("text rows: %d (%.0fs)", len(rows), time.monotonic() - t)

    t = time.monotonic()
    nets, sizes = coord2_run.build_networks(con, view)
    (out / "nets").mkdir(exist_ok=True)
    for name, edges in nets.items():
        edges.to_parquet(out / "nets" / f"{name}.parquet")
    sizes.to_csv(out / "bipartite_sizes.csv", index=False)
    log.info("networks done (%.0fs)\n%s", time.monotonic() - t, sizes.to_string(index=False))


if __name__ == "__main__":
    main()

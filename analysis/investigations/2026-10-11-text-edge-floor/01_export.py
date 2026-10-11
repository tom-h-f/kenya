"""Window manifest and text-eligible rows for the 21d daily window, read-only on R2."""

import argparse
import logging
import time

from kma import bench, coord2
from kma.db import connect

from common import WINDOW, wdir

log = logging.getLogger("export")


def main() -> None:
    ap = argparse.ArgumentParser()
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
    rows = coord2.text_rows(con, view)
    rows = rows.assign(user_id=rows["user_id"].astype(str))
    rows.to_parquet(out / "rows.parquet")
    log.info("text rows: %d (%.0fs)", len(rows), time.monotonic() - t)


if __name__ == "__main__":
    main()

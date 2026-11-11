"""Weekly canary: would the detector and the leads rule catch a planted group?

    cd analysis
    uv run python -m kma.canary_check                   # dry: prints the result
    uv run python -m kma.canary_check --persist         # what the cloud unit runs

Plants a synthetic coordinated group (`canary.new`, shaped on a real IO
archive operation) in memory, reruns the daily pass's rank on the same window
with it injected (`v2_daily.rank(persist=False)`), and asks the same question
the production rule asks: did the planted accounts land together in one kept
community that `leads_rule` flags?

Nothing reaches the public tables. The planted posts are never written, the
rank persists nothing, and the only outputs are a private result row under
`leads/platform=x/kind=canary_checks/` and `kma_canary.prom`. That is why it
does not use the daily's `--canary`: a planted group in the live run would be
counted on the public dashboard and explorer.

What it does not test is the diff against the previous run and the memory,
because a canary is new by construction; `kma_leads.prom` freshness covers
that the leads pass itself runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd

from kma import bench, canary, coord2, leads_rule, textfile, v2_daily, v2_daily_run, window
from kma.db import BUCKET

log = logging.getLogger(__name__)

SIZE = 20
ACTIVITY = "normal"
PLANT_DAYS = 7
# A canary split across communities is not one detected group.
MIN_TOGETHER = 0.5

GAUGE_FILE = "kma_canary.prom"
GAUGE_RUN = "kma_canary_last_run_timestamp_seconds"
GAUGE_DETECTED = "kma_canary_detected"
RESULT_PREFIX = "leads/platform=x/kind=canary_checks"


def evaluate(planted: list[str], members: pd.DataFrame, communities: pd.DataFrame) -> dict:
    """Where the planted accounts landed, and whether the rule flags it."""
    ids = members["user_id"].astype(str)
    found = members[ids.isin(set(planted))]
    out = {"planted": len(planted), "in_graph": len(found), "community": None,
           "together": 0.0, "community_size": None, "kept": False, "flagged": False,
           "signals": None, "band": None, "detected": False}
    if found.empty:
        return out
    top = int(found["community"].mode().iloc[0])
    out["community"] = top
    out["together"] = round(float((found["community"] == top).sum()) / len(planted), 3)
    row = communities[communities["community"] == top]
    if row.empty:
        return out
    sig = leads_rule.signals(row).iloc[0]
    size = int(row["size"].iloc[0])
    out.update(community_size=size, kept=bool(sig["kept"]), flagged=bool(sig["flagged"]),
               signals=sig["signals"], band=leads_rule.band(size))
    out["detected"] = out["flagged"] and out["together"] >= MIN_TOGETHER
    return out


def run(
    con: duckdb.DuckDBPyConnection,
    *,
    end: str | None = None,
    days: int = v2_daily_run.DEFAULT_DAYS,
    persist: bool = False,
    textfile_dir: Path | None = None,
) -> dict:
    w = window.Window.ending(end, days)
    manifest = window.ensure(w, con=con)
    has_text = v2_daily.text_trace_exists(con, w.id)
    planted_at = datetime.combine(w.end + timedelta(days=1), time(0), tzinfo=timezone.utc)
    # The window's pinned objects, not the whole archive: planting reads the
    # last week for retweet targets and tag collisions, and an unpinned view
    # scans every posts object in R2.
    pinned = coord2.posts_view(bench.pinned_source(manifest, "posts"))
    recent = (f"SELECT * FROM ({pinned}) WHERE created_at >= "
              f"TIMESTAMPTZ '{(planted_at - timedelta(days=PLANT_DAYS)).isoformat()}'")
    plant = canary.new(con, recent, canary_id=f"check-{planted_at:%Y-%m-%d}", size=SIZE,
                       activity=ACTIVITY, days=PLANT_DAYS, now=planted_at)
    log.info("canary %s: %d accounts, %d posts", plant.canary_id, len(plant.accounts),
             len(plant.posts))

    result = v2_daily.rank(
        con, w, canary_posts=plant.posts, persist=False, manifest=manifest,
        text_threshold=v2_daily.TEXT_THRESHOLD if has_text else None,
        text_overlap=coord2.TEXT_MIN_OVERLAP if has_text else None,
    )
    out = {
        "canary_id": plant.canary_id, "window_id": w.id, "text_trace": has_text,
        "rule": leads_rule.RULE_VERSION,
        **evaluate(plant.accounts, result.tables["members"], result.tables["communities"]),
        "checked_at": datetime.now(timezone.utc),
    }
    log.info("canary %s detected=%s (%s)", plant.canary_id, out["detected"], out["signals"])

    if persist:
        frame = pd.DataFrame([out])
        key = f"{RESULT_PREFIX}/dt={out['checked_at']:%Y-%m-%d}/run={out['checked_at']:%Y%m%dT%H%M%SZ}.parquet"
        con.register("_canary_check", frame)
        try:
            con.execute(f"COPY _canary_check TO 'r2://{BUCKET}/{key}' (FORMAT parquet)")
        finally:
            con.unregister("_canary_check")
        out["key"] = key
    if textfile_dir is not None:
        out["gauge"] = str(textfile.write(textfile_dir, GAUGE_FILE, textfile.gauges([
            (GAUGE_RUN, {}, int(out["checked_at"].timestamp())),
            (GAUGE_DETECTED, {}, int(out["detected"])),
        ])))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Plant a canary in memory and check the leads rule catches it.")
    ap.add_argument("--end", default="", help="last collection day, ISO; default yesterday UTC")
    ap.add_argument("--days", type=int, default=v2_daily_run.DEFAULT_DAYS,
                    help="window length; the daily pass's, so its text trace is reused")
    ap.add_argument("--persist", action="store_true",
                    help=f"write the result row to R2 and {GAUGE_FILE} to ${textfile.ENV}")
    args = ap.parse_args(argv)
    textfile_dir = Path(os.environ[textfile.ENV]) if os.getenv(textfile.ENV) else None
    if args.persist and textfile_dir is None:
        ap.error(f"--persist needs ${textfile.ENV}: a check that cannot report should not run")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from kma.db import connect

    con = connect()
    con.execute("SET enable_progress_bar = false")
    out = run(con, end=args.end, days=args.days, persist=args.persist,
              textfile_dir=textfile_dir if args.persist else None)
    print(json.dumps(out, default=str, indent=2))


if __name__ == "__main__":
    main()

"""The figures the dashboard renders, and nothing else.

    uv run python -m kma.figures --out figures/figures.json

`docs/analysis/2026-09-13-publishable-statistics.md` audited every candidate
number this project holds and found four publishable. The 2026-09-14 depth
result added a fifth, and it is the strongest of them, because it is a
measurement of our own method failing under a control arm.

This exports exactly those and refuses to invent a sixth.

TWO HALVES, AND THE DISTINCTION IS LOAD BEARING
===============================================
**Live** figures are recomputed from R2 every export - the corpus shape, the
latest v2 run, the depth check. A number that can be computed is computed,
because a transcribed one goes stale silently and nobody finds out.

**Recorded** figures come from `figures/recorded.yaml`: measurements that are
not derivable from a persisted artifact, each carrying the document and date it
came from. They are transcribed because there is no alternative, which is why
`export` refuses any entry missing `source` or `measured`.

WHAT IS DELIBERATELY ABSENT
===========================
Any prevalence rate. Every partition this project holds was collected because
something about it looked interesting, so no rate over it has a denominator.
The control arm (`kenya_monitor.control`) exists to fix that and has not
accumulated yet. The page says so rather than leaving the gap for a reader to
fill with an assumption.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

log = logging.getLogger(__name__)

RECORDED_PATH = Path(__file__).resolve().parents[2] / "figures" / "recorded.yaml"

# Entries here are transcribed from a document rather than computed, so each one
# has to say which document and when. Without both, a reader cannot tell a
# measurement from a remembered number.
REQUIRED_PROVENANCE = ("source", "measured")


def load_recorded(path: Path = RECORDED_PATH) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for name, block in raw.items():
        missing = [k for k in REQUIRED_PROVENANCE if not block.get(k)]
        if missing:
            raise ValueError(
                f"recorded figure {name!r} is missing {', '.join(missing)} - "
                "a transcribed number without its source is not publishable"
            )
    return raw


def corpus_shape(con, snapshot: str) -> dict:
    """What the corpus IS, by collection scope.

    Split by scope rather than totalled, because the totals are the thing that
    misleads: targeted partitions oversample the toxic tail by construction and
    the control arm is neither.
    """
    from kma import bench, db

    manifest = bench.load(snapshot, con=con)
    posts = bench.pinned_source(manifest, "posts")
    rows = con.sql(
        f"""
        WITH first_seen AS (
            SELECT platform_post_id, min_by(type, collected_at) AS first_type
            FROM {posts} GROUP BY platform_post_id
        )
        SELECT first_type, count(*) AS posts FROM first_seen GROUP BY 1 ORDER BY 2 DESC
        """
    ).df()
    scopes = {}
    for scope in ("baseline", "targeted", "control"):
        wanted = set(db._SCOPE_TYPES[scope])
        scopes[scope] = int(rows.loc[rows["first_type"].isin(wanted), "posts"].sum())
    return {
        "snapshot": snapshot,
        "by_type": rows.to_dict("records"),
        "by_scope": scopes,
        "note": (
            "Counted on first-seen type. Targeted partitions oversample the "
            "toxic tail by construction; the control arm is in neither scope."
        ),
    }


def ranking_stability(con, before: str, after: str) -> dict:
    """The depth result: what a v2 ranking does when its accounts are deepened.

    The figure of record is the gap between the treated arm and the holdout,
    not the treated arm alone - a recomputed ranking churns anyway, and without
    the control there is no way to tell the two apart.

    Treatment is windowed on the two runs' SNAPSHOTS, matching
    `investigations/2026-09-14-depth-rerank/01_rank_check.py`. Accounts deepened
    before the earlier snapshot are treated in both runs and belong to neither
    arm; counting them inflates the treated arm and puts a survivor in it, which
    is how this page and the findings disagreed on 391/1 against 390/0.
    """
    from kma import bench, db

    def ranked(run: str) -> pd.DataFrame:
        frame = db.coord2_scores(con, run).df()
        frame["user_id"] = frame["user_id"].astype(str)
        return frame[["user_id", "rank"]]

    def pinned_at(run: str):
        name = con.sql(
            f"SELECT any_value(snapshot) FROM {db.coord2_scores_source()} WHERE run = '{run}'"
        ).fetchone()[0]
        return bench.load(name, con=con)["created_at"].max()

    first, second = ranked(before), ranked(after)
    t0, t1 = pd.Timestamp(pinned_at(before)), pd.Timestamp(pinned_at(after))

    ledger = db.deepened_accounts(con).df()
    ledger["user_id"] = ledger["user_id"].astype(str)
    in_window = (ledger["first_deepened_at"] > t0) & (ledger["first_deepened_at"] <= t1)
    treated = set(ledger.loc[in_window, "user_id"])
    held = set(db.held_out_accounts(con).df()["user_id"].astype(str))

    top_before = set(first["user_id"])
    top_after = set(second["user_id"])

    def retention(arm: set) -> dict:
        was = top_before & arm
        return {
            "in_top_before": len(was),
            "in_top_after": len(was & top_after),
            "retained": round(len(was & top_after) / len(was), 3) if was else None,
        }

    return {
        "before_run": before,
        "after_run": after,
        "churn_floor": round(len(top_before & top_after) / max(len(top_before), 1), 3),
        "treated": retention(treated - held),
        "holdout": retention(held),
        "note": (
            "Treated accounts were deepened between the two runs; the holdout "
            "was drawn from the same ranked selection by the same rule and "
            "deliberately not fetched. The gap between them is the result."
        ),
    }


def export(
    out: Path,
    *,
    snapshot: str,
    before_run: str,
    after_run: str,
    con=None,
    recorded_path: Path = RECORDED_PATH,
) -> dict:
    from kma.db import connect

    con = con or connect()
    figures = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "live": {
            "corpus_shape": corpus_shape(con, snapshot),
            "ranking_stability": ranking_stability(con, before_run, after_run),
        },
        "recorded": load_recorded(recorded_path),
        "absent": {
            "prevalence": (
                "No rate is published. Every partition here was collected "
                "because something about it looked interesting, so no rate over "
                "it has a denominator. The control arm exists to fix that and "
                "has not accumulated enough windows yet."
            )
        },
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(figures, indent=2, default=str))
    log.info("wrote %s", out)
    return figures


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=RECORDED_PATH.parent / "figures.json")
    ap.add_argument("--snapshot", default="2026-09-14-deep500")
    ap.add_argument("--before-run", default="20260914T090118Z")
    ap.add_argument("--after-run", default="20260914T150111Z")
    args = ap.parse_args()

    figures = export(
        args.out,
        snapshot=args.snapshot,
        before_run=args.before_run,
        after_run=args.after_run,
    )
    print(json.dumps(figures["live"]["ranking_stability"], indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

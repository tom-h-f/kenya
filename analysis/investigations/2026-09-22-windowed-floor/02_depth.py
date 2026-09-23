"""Does windowed detection survive deepening? The test route 3 must pass.

    uv run python investigations/2026-09-22-windowed-floor/02_depth.py \\
        --before <cache of 2026-09-14-deepened> --after <cache of 2026-09-14-deep500>

The global ranking failed it at floor 2: of accounts in the top 500, 0 of 390
deepened kept their place against 22.7% of the 97-account holdout, which was
selected by the same rule and deliberately not fetched
(`2026-09-14-depth-rerank/findings.md`). The ranking was scoring accounts for
having little history.

The windowed analogue: an account-window is SHOWN when the account sits in a
community of `windowed.MIN_COMMUNITY`+ in that window. For every window both
snapshots cover completely, take the account-windows shown before, and report
the share still shown after, per arm. Deepening adds history to the treated
accounts only; if windowed detection measures behaviour, treated and holdout
retain at similar rates. `gained` counts account-windows shown after but not
before - new history can put an account into a window it was absent from,
which is the method working, not failing.

Arms come from the `deep_timelines/` ledger exactly as `01_rank_check.py`
builds them: treated = first deepened between the two snapshots' pin times,
holdout = held out in that pass, prior = deepened before (excluded).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep  # noqa: E402

from kma import bench, db, windowed  # noqa: E402

BEFORE = "2026-09-14-deepened"
AFTER = "2026-09-14-deep500"


def arms(con) -> tuple[set, set, set]:
    t0 = pd.Timestamp(bench.load(BEFORE, con=con)["created_at"].max())
    t1 = pd.Timestamp(bench.load(AFTER, con=con)["created_at"].max())
    ledger = db.deepened_accounts(con).df()
    ledger["user_id"] = ledger["user_id"].astype(str)
    held = db.held_out_accounts(con).df()
    held["user_id"] = held["user_id"].astype(str)
    treated = set(ledger.loc[(ledger["first_deepened_at"] > t0)
                             & (ledger["first_deepened_at"] <= t1), "user_id"])
    prior = set(ledger.loc[ledger["first_deepened_at"] <= t0, "user_id"])
    holdout = set(held.loc[(held["held_out_at"] > t0) & (held["held_out_at"] <= t1), "user_id"])
    return treated, holdout, prior


def shown(frame: pd.DataFrame) -> set[tuple[str, str]]:
    keep = frame[frame["community_size"] >= windowed.MIN_COMMUNITY]
    return set(zip(keep["user_id"].astype(str), keep["window"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--floors", default="2,3,5,10,20")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "depth_results.csv")
    args = ap.parse_args()

    con = db.connect()
    treated, holdout, prior = arms(con)
    print(f"treated {len(treated)}, holdout {len(holdout)}, prior (excluded) {len(prior)}")

    before, after = sweep.load(args.before), sweep.load(args.after)
    records = []
    for width in ("day", "week"):
        common = sorted(set(sweep.complete_windows(before, width))
                        & set(sweep.complete_windows(after, width)))
        for floor in (int(f) for f in args.floors.split(",")):
            b = shown(sweep.run_real(before, width=width, floor=floor, windows=common))
            a = shown(sweep.run_real(after, width=width, floor=floor, windows=common))
            for label, ids in (("treated", treated), ("holdout", holdout)):
                was = {k for k in b if k[0] in ids}
                now = {k for k in a if k[0] in ids}
                records.append({
                    "width": width, "floor": floor, "arm": label, "windows": len(common),
                    "accounts": len(ids),
                    "shown_before": len(was),
                    "kept": len(was & now),
                    "retained": round(len(was & now) / len(was), 3) if was else None,
                    "gained": len(now - was),
                    "accounts_shown_before": len({k[0] for k in was}),
                    "accounts_shown_after": len({k[0] for k in now}),
                })
                print(records[-1], flush=True)
    pd.DataFrame(records).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

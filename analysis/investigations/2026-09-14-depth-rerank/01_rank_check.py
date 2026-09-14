"""Did deepening move the accounts it treated, and in which direction?

    cd analysis && uv run python investigations/2026-09-14-depth-rerank/01_rank_check.py \\
        --before 20260914T090118Z --after <run> [--top 500]

The check `docs/plans/2026-09-08-collector-depth.md` requires of any v2 re-run
after a depth pass. It is a join, not archaeology: `deep_timelines/` records
every deepened account with its pre-treatment state, keyed on `user_id`.

Treated accounts are those first deepened between the two runs' SNAPSHOTS, not
between the runs themselves. A snapshot pins an explicit object list, so what a
run could see is fixed when the snapshot is taken, not when the run happens -
the 2026-09-14 check has a before-run computed AFTER the depth pass finished
and still blind to it, because its snapshot predated the new partition.
Accounts already deepened before the earlier snapshot are treated in both and
belong to neither arm, so they are excluded rather than counted as controls.

Persisted runs hold the top 500 only (`coord2_run.run(top=500)`), so an
account missing from a run left that top 500; its true rank is not recorded and
no rank delta can be computed for it.

The comparison a raw "n of the treated left the top 500" invites is the wrong
one, because a ranking recomputed on a changed corpus churns anyway. Two
controls are reported beside it:

  - the CHURN FLOOR: how much of the top N is shared between the two runs at
    all. A treated exit rate below this is noise.
  - a MATCHED control: untreated accounts drawn from the same before-rank
    bands as the treated ones, so "fell out" is measured against accounts that
    started where the treated accounts started. When the depth pass held some
    of its selection back (`deep_timelines.hold_out`), that holdout is the
    control and is reported separately - it is the only control drawn from the
    same population by the same rule.
"""

import argparse

import pandas as pd

from kma import db


def _snapshot_time(con, run: str):
    """When the run's snapshot was taken - the moment its view of the corpus
    was frozen, which is what decides whether a depth pass is visible to it."""
    from kma import bench

    name = con.sql(
        f"SELECT any_value(snapshot) FROM {db.coord2_scores_source()} WHERE run = '{run}'"
    ).fetchone()[0]
    manifest = bench.load(name, con=con)
    return manifest["created_at"].max()


def _ranking(con, run: str) -> pd.DataFrame:
    frame = db.coord2_scores(con, run).df()
    if frame.empty:
        raise SystemExit(f"no rows for run {run}")
    frame["user_id"] = frame["user_id"].astype(str)
    return frame[["user_id", "centrality", "rank"]]


def _band(rank: pd.Series, width: int) -> pd.Series:
    return ((rank - 1) // width) * width


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--before", required=True, help="run id of the earlier v2 pass")
    ap.add_argument("--after", required=True, help="run id of the pass taken after deepening")
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--band", type=int, default=100, help="before-rank band width for matching")
    ap.add_argument("--since", help="override the treated window start (ISO, UTC)")
    ap.add_argument("--until", help="override the treated window end (ISO, UTC)")
    args = ap.parse_args()

    con = db.connect()
    before = _ranking(con, args.before)
    after = _ranking(con, args.after)
    t0 = args.since or _snapshot_time(con, args.before)
    t1 = args.until or _snapshot_time(con, args.after)
    print(f"before {args.before}: {len(before):,} accounts, snapshot pinned {t0}")
    print(f"after  {args.after}: {len(after):,} accounts, snapshot pinned {t1}")
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)

    ledger = db.deepened_accounts(con).df()
    ledger["user_id"] = ledger["user_id"].astype(str)
    holdout = db.held_out_accounts(con).df()
    holdout["user_id"] = holdout["user_id"].astype(str)
    held_ids = set(
        holdout.loc[
            (holdout["held_out_at"] > t0) & (holdout["held_out_at"] <= t1), "user_id"
        ]
    )
    treated = ledger[
        (ledger["first_deepened_at"] > t0) & (ledger["first_deepened_at"] <= t1)
    ]
    earlier = ledger[ledger["first_deepened_at"] <= t0]
    print(
        f"ledger: {len(ledger):,} accounts deepened, {len(treated):,} between the two runs, "
        f"{len(earlier):,} already deepened before the earlier run (excluded)"
    )

    joined = before.merge(after, on="user_id", how="outer", suffixes=("_before", "_after"))
    joined["treated"] = joined["user_id"].isin(set(treated["user_id"]))
    joined["held_out"] = joined["user_id"].isin(held_ids)
    joined["prior"] = joined["user_id"].isin(set(earlier["user_id"]))
    if held_ids:
        print(f"holdout arm: {len(held_ids):,} accounts selected and deliberately not fetched")

    top_before = set(before.loc[before["rank"] <= args.top, "user_id"])
    top_after = set(after.loc[after["rank"] <= args.top, "user_id"])
    shared = len(top_before & top_after)
    print(
        f"\nchurn floor: {shared} of {args.top} accounts are in both top {args.top}s "
        f"({shared / args.top:.1%} shared)"
    )

    pool = joined[~joined["prior"] & joined["rank_before"].notna()].copy()
    pool["in_top_before"] = pool["rank_before"] <= args.top
    pool["in_top_after"] = pool["rank_after"].notna() & (pool["rank_after"] <= args.top)
    pool["band"] = _band(pool["rank_before"], args.band)

    was_top = pool[pool["in_top_before"]]
    arms = [("treated", was_top[was_top["treated"]])]
    if held_ids:
        arms.append(("holdout", was_top[was_top["held_out"]]))
    arms.append(
        ("untreated", was_top[~was_top["treated"] & ~was_top["held_out"]])
    )
    rows = []
    for label, arm in arms:
        rows.append({
            "arm": label,
            "in top before": len(arm),
            "in top after": int(arm["in_top_after"].sum()),
            "retained": f"{arm['in_top_after'].mean():.1%}" if len(arm) else "-",
        })
    print("\nraw retention of the top", args.top)
    print(pd.DataFrame(rows).to_string(index=False))

    # Matched: within each before-rank band, compare treated retention against
    # the untreated accounts that started in the same band. A treated set drawn
    # from the head of the ranking would otherwise be compared against a control
    # drawn from its tail.
    matched = []
    for band, group in was_top.groupby("band"):
        t = group[group["treated"]]
        u = group[group["held_out"]] if held_ids else group[~group["treated"]]
        if t.empty:
            continue
        matched.append({
            "before rank": f"{int(band) + 1}-{int(band) + args.band}",
            "treated": len(t),
            "treated kept": f"{t['in_top_after'].mean():.1%}",
            "control": len(u),
            "control kept": f"{u['in_top_after'].mean():.1%}" if len(u) else "-",
        })
    if matched:
        print("\nmatched on before-rank band")
        print(pd.DataFrame(matched).to_string(index=False))

    moved = was_top[was_top["treated"] & was_top["rank_after"].notna()]
    if not moved.empty:
        delta = moved["rank_after"] - moved["rank_before"]
        print(
            f"\ntreated accounts still ranked after: {len(moved)}; "
            f"rank change median {delta.median():+,.0f}, "
            f"min {delta.min():+,.0f}, max {delta.max():+,.0f}"
        )
    dropped = int((was_top["treated"] & was_top["rank_after"].isna()).sum())
    print(f"treated accounts that left the ranking entirely: {dropped}")


if __name__ == "__main__":
    main()

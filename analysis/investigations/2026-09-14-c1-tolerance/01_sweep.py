"""Sweep the three instrumented tolerance terms and print one table.

    cd analysis && uv run python investigations/2026-09-14-c1-tolerance/01_sweep.py \\
        --snapshot 2026-09-13-c1-replay --passes 10

Each variant is a full replay over the same passes, so this is expensive - a
25-pass run took about 40 minutes against live R2. Sweep at `--passes 10`
first and confirm the winner at 25, rather than paying the full cost per point.

The timing term is a sweep rather than a switch because nothing records the true
selection instant: `census_pass_times` dates a pass by its first engagement
write, and `census_runs.collected_at` is stamped when the pass FINISHES, so it
is later still. Standing progressively earlier and watching recall is the only
measurement available.
"""

import argparse
from datetime import timedelta

import pandas as pd

from kma import bench, replay
from kma.db import connect


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default="2026-09-13-c1-replay")
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--shifts", type=float, nargs="*", default=[0, 2, 5, 10])
    args = ap.parse_args()

    con = connect()
    manifest = bench.load(args.snapshot, con=con)
    base_times = replay.census_pass_times(manifest)[-args.passes :]
    if not base_times:
        raise SystemExit(f"snapshot {args.snapshot!r} holds no engagements objects")
    print(f"{len(base_times)} passes, {base_times[0]} .. {base_times[-1]}")

    variants = [("shift", s, "engagements", False) for s in args.shifts]
    variants += [
        ("ledger ground truth", 0.0, "census_ttl", False),
        ("truncate to fetched", 0.0, "engagements", True),
        ("all three", None, "census_ttl", True),
    ]

    rows = []
    best_shift = 0.0
    for label, shift, ground_truth, truncate in variants:
        if shift is None:
            shift = best_shift
        times = [t - timedelta(minutes=shift) for t in base_times]
        rep = replay.reproduce(
            con,
            manifest=manifest,
            times=times,
            snapshot=args.snapshot,
            policy=replay.MergedCensus(),
            ground_truth=ground_truth,
            truncate=truncate,
        )
        rows.append({
            "variant": label,
            "shift min": shift,
            "ground truth": ground_truth,
            "truncate": truncate,
            "recall": round(rep.recall, 3),
            "precision": round(rep.precision, 3),
            "jaccard": round(rep.jaccard, 3),
            "replayed": int(rep.per_pass["replayed"].sum()),
            "observed": int(rep.per_pass["observed"].sum()),
        })
        print(pd.DataFrame(rows[-1:]).to_string(index=False))
        if label == "shift" and rows[-1]["recall"] >= max(
            r["recall"] for r in rows if r["variant"] == "shift"
        ):
            best_shift = shift

    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print(f"\nbest shift by recall: {best_shift:g} minutes")
    print(f"\n{replay.LIMITATION}")


if __name__ == "__main__":
    main()

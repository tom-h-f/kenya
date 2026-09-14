"""One replay variant, recorded, so a sweep survives being killed.

    cd analysis && uv run python investigations/2026-09-14-c1-tolerance/02_variant.py \\
        --label ledger --ground-truth census_ttl --passes 10

Written after the sweep lost three hours twice over: eight variants fanned out
on Modal queued for capacity rather than running (an over-requested 16 GiB is a
scheduling penalty, not free headroom), and the serial version before it was
killed by the mac's low-memory watchdog with nothing saved.

So: one variant per invocation, appended to `out/variants.json` as soon as it
finishes. Re-running a completed label is a no-op unless `--force`.
"""

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

from kma import bench, replay
from kma.db import connect

HERE = Path(__file__).parent
OUT = HERE / "out"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--snapshot", default="2026-09-13-c1-replay")
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--shift-minutes", type=float, default=0.0)
    ap.add_argument("--ground-truth", choices=("engagements", "census_ttl"),
                    default="engagements")
    ap.add_argument("--truncate", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    path = OUT / "variants.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    key = f"{args.label}@{args.passes}"
    if key in done and not args.force:
        print(f"{key}: already recorded -> {done[key]}")
        return

    con = connect()
    manifest = bench.load(args.snapshot, con=con)
    times = replay.census_pass_times(manifest)[-args.passes:]
    times = [t - timedelta(minutes=args.shift_minutes) for t in times]
    print(f"{key}: {len(times)} passes, {times[0]} .. {times[-1]}", flush=True)

    start = time.perf_counter()
    rep = replay.reproduce(
        con, manifest=manifest, times=times, snapshot=args.snapshot,
        policy=replay.MergedCensus(),
        ground_truth=args.ground_truth, truncate=args.truncate,
    )
    row = {
        "label": args.label, "passes": args.passes, "shift_min": args.shift_minutes,
        "ground_truth": args.ground_truth, "truncate": args.truncate,
        "recall": round(rep.recall, 3), "precision": round(rep.precision, 3),
        "jaccard": round(rep.jaccard, 3),
        "replayed": int(rep.per_pass["replayed"].sum()),
        "observed": int(rep.per_pass["observed"].sum()),
        "seconds": round(time.perf_counter() - start),
    }
    done[key] = row
    path.write_text(json.dumps(done, indent=2))
    print(row)
    print(f"recorded -> {path}")


if __name__ == "__main__":
    main()

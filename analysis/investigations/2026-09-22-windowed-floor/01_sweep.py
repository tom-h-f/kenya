"""Run the windowed `MIN_ENTITIES` sweep.

    cd analysis
    # 1. timed traces for the last 60 days of the snapshot, once
    uv run python investigations/2026-09-22-windowed-floor/01_sweep.py extract \\
        --snapshot 2026-09-14-deep500 --days 60 --cache <dir>
    # 2. the grid, in worker processes
    uv run python investigations/2026-09-22-windowed-floor/01_sweep.py sweep \\
        --cache <dir> --workers 4 [--smoke]

Local rather than Modal: the Modal workspace was over its spend limit on
2026-09-22 when this ran. The extraction is the only R2 read; everything after
it is pandas and igraph over the cached traces.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep  # noqa: E402

from kma import windowed  # noqa: E402

HERE = Path(__file__).resolve().parent
WIDTHS = ("day", "week")
FLOORS = (2, 3, 5, 10, 20)
PLANT_SEEDS = (0, 1, 2)


def extract(snapshot: str, days: int, cache: Path, memory: str, threads: int) -> dict:
    from kma.db import connect

    t0 = time.monotonic()
    con = connect()
    # Bounded so the extraction cannot crowd out the rest of the machine.
    con.execute(f"SET memory_limit='{memory}'")
    con.execute(f"SET threads={threads}")
    rows = windowed.timed_traces(con, windowed.snapshot_view(con, snapshot, days=days))
    cache.mkdir(parents=True, exist_ok=True)
    out = {"snapshot": snapshot, "days": days}
    for name, frame in rows.items():
        frame.to_parquet(cache / f"{name}.parquet", index=False)
        out[name] = len(frame)
        out[f"{name}_users"] = int(frame["user_id"].nunique())
    stamps = [f["created_at"] for f in rows.values() if len(f)]
    out["first"] = str(min(s.min() for s in stamps))
    out["last"] = str(max(s.max() for s in stamps))
    out["seconds"] = round(time.monotonic() - t0, 1)
    (cache / "extract.json").write_text(json.dumps(out, indent=2))
    return out


def point(cache: str, width: str, floor: int, arm: str, shape: str = "", seed: int = 0,
          burn_in: int = 100, max_windows: int = 0) -> dict:
    t0 = time.monotonic()
    rows = sweep.load(cache)
    windows = sweep.complete_windows(rows, width)
    if max_windows:
        windows = windows[-max_windows:]
    out = {"width": width, "floor": floor, "arm": arm, "shape": shape, "seed": seed}
    if arm == "real":
        out.update(sweep.summary(sweep.run_real(rows, width=width, floor=floor, windows=windows)))
        out["windows"] = len(windows)
    elif arm == "null":
        out["burn_in"] = burn_in
        out.update(sweep.summary(sweep.run_null(rows, width=width, floor=floor, windows=windows,
                                                seed=seed, burn_in=burn_in)))
        out["windows"] = len(windows)
    elif arm == "plant":
        days = sweep.complete_windows(rows, "day")
        day = days[int(np.random.default_rng(1000 + seed).integers(len(days)))]
        planted, ids = sweep.plant(rows, sweep.PLANTS[shape], day, seed=seed)
        window = windowed.window_label(pd.Series([pd.Timestamp(day, tz="UTC")]), width).iloc[0]
        frame = sweep.run_real(planted, width=width, floor=floor, windows=[window])
        out.update({"day": day, "window": window, **sweep.plant_metrics(frame, ids, window)})
        out["windows"] = 1
    else:
        raise ValueError(arm)
    out["seconds"] = round(time.monotonic() - t0, 1)
    return out


def grid(burn_in: int):
    for width in WIDTHS:
        for floor in FLOORS:
            yield dict(width=width, floor=floor, arm="real")
            yield dict(width=width, floor=floor, arm="null", burn_in=burn_in)
            for shape in sweep.PLANTS:
                for seed in PLANT_SEEDS:
                    yield dict(width=width, floor=floor, arm="plant", shape=shape, seed=seed)


def smoke(burn_in: int):
    return [
        dict(width="day", floor=5, arm="real", max_windows=2),
        dict(width="day", floor=5, arm="null", max_windows=2, burn_in=burn_in),
        dict(width="week", floor=10, arm="null", max_windows=1, burn_in=burn_in),
        dict(width="day", floor=5, arm="plant", shape="heavy", seed=0),
        dict(width="week", floor=10, arm="plant", shape="light", seed=0),
    ]


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "sweep"])
    ap.add_argument("--snapshot", default="2026-09-14-deep500")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--memory", default="6GB")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--burn-in", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "sweep_results.csv")
    args = ap.parse_args()

    if args.stage == "extract":
        print(json.dumps(extract(args.snapshot, args.days, args.cache, args.memory, args.threads),
                         indent=2))
        return

    calls = smoke(args.burn_in) if args.smoke else list(grid(args.burn_in))
    records = []
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(point, str(args.cache), **c): c for c in calls}
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:  # recorded, not raised: one bad point must not lose the grid
                r = {**c, "error": repr(exc)[:300]}
            records.append(r)
            print(json.dumps(r), flush=True)
    frame = pd.DataFrame(records)
    target = args.out.with_name(args.out.stem + "_smoke.csv") if args.smoke else args.out
    frame.to_csv(target, index=False)
    print(f"wrote {target} ({len(frame)} points, {time.monotonic() - t0:.0f}s wall)")


if __name__ == "__main__":
    main()

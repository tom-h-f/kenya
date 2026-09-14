"""Modal app: score the census replay's tolerance terms off the laptop.

    uv run --with modal modal deploy modal_replay.py

Each variant is a full replay over the same passes and reads a large share of
the snapshot. Three of these running beside each other took down every one of
them on 2026-09-14 - two on the mac and the depth pass on pi0 died within
minutes on DNS and connection failures - and a later single run was killed by
the mac's low-memory watchdog. Neither failure is about the work; both are about
where it ran.
"""

import modal

app = modal.App("kma-replay")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pandas>=2",
        "pyarrow>=18",
        "numpy>=1.26",
        "networkx>=3.3",
        "scikit-learn>=1.5",
        "scipy>=1.13",
        "duckdb>=1.1",
        "python-dotenv>=1.0",
        # duckdb's timestamp casts reach for pytz; pandas no longer pulls it in.
        "pytz",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src", ignore=["**/__pycache__/**", "*.pyc"])
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("kenya-r2")],
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 4,
)
def one(variant: dict) -> dict:
    """One variant, so the sweep can fan out instead of running serially.

    Each variant is an independent replay of the same passes - nothing is shared
    between them but the snapshot - so running them side by side turns a
    multi-hour serial sweep into one variant's wall clock. The serial `sweep`
    below stays for the case where the shift that wins has to pick the shift
    that "all three" uses.
    """
    import logging
    from datetime import timedelta

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from kma import bench, replay
    from kma.db import connect

    con = connect()
    manifest = bench.load(variant["snapshot"], con=con)
    times = replay.census_pass_times(manifest)[-variant["passes"]:]
    times = [t - timedelta(minutes=variant["shift_min"]) for t in times]
    rep = replay.reproduce(
        con, manifest=manifest, times=times, snapshot=variant["snapshot"],
        policy=replay.MergedCensus(),
        ground_truth=variant["ground_truth"], truncate=variant["truncate"],
    )
    out = {
        **{k: variant[k] for k in ("label", "shift_min", "ground_truth", "truncate")},
        "recall": round(rep.recall, 3),
        "precision": round(rep.precision, 3),
        "jaccard": round(rep.jaccard, 3),
        "replayed": int(rep.per_pass["replayed"].sum()),
        "observed": int(rep.per_pass["observed"].sum()),
    }
    print(out, flush=True)
    return out


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("kenya-r2")],
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 6,
)
def sweep(snapshot: str = "2026-09-13-c1-replay", passes: int = 25,
          shifts: tuple = (0.0, 2.0, 5.0, 10.0)) -> list[dict]:
    import logging
    from datetime import timedelta

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from kma import bench, replay
    from kma.db import connect

    con = connect()
    manifest = bench.load(snapshot, con=con)
    base_times = replay.census_pass_times(manifest)[-passes:]
    print(f"{len(base_times)} passes, {base_times[0]} .. {base_times[-1]}", flush=True)

    variants = [("shift", s, "engagements", False) for s in shifts]
    variants += [
        ("ledger ground truth", 0.0, "census_ttl", False),
        ("truncate to fetched", 0.0, "engagements", True),
        ("all three", None, "census_ttl", True),
    ]

    rows: list[dict] = []
    best_shift = 0.0
    for label, shift, ground_truth, truncate in variants:
        if shift is None:
            shift = best_shift
        times = [t - timedelta(minutes=shift) for t in base_times]
        rep = replay.reproduce(
            con, manifest=manifest, times=times, snapshot=snapshot,
            policy=replay.MergedCensus(), ground_truth=ground_truth, truncate=truncate,
        )
        row = {
            "variant": label, "shift_min": shift, "ground_truth": ground_truth,
            "truncate": truncate, "recall": round(rep.recall, 3),
            "precision": round(rep.precision, 3), "jaccard": round(rep.jaccard, 3),
            "replayed": int(rep.per_pass["replayed"].sum()),
            "observed": int(rep.per_pass["observed"].sum()),
        }
        rows.append(row)
        print(row, flush=True)
        if label == "shift" and row["recall"] >= max(
            r["recall"] for r in rows if r["variant"] == "shift"
        ):
            best_shift = shift
    return rows


@app.local_entrypoint()
def main(snapshot: str = "2026-09-13-c1-replay", passes: int = 25):
    for row in sweep.remote(snapshot, passes):
        print(row)

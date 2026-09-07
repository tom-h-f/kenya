"""Modal app: run the IOHunter reproduction gate, six countries in parallel.

The unsupervised half is cheap and runs anywhere. The supervised half is not:
the embedding needs enough epochs to converge, and on the larger graphs a CPU
pass is the bottleneck. Venezuela measured 77.25 Macro-F1 at 5 epochs against
90.64 at 50, so epoch count is not a detail to economise on.

    # once, from analysis/, with the unpacked release on disk:
    uv run --with modal modal volume create iohunter-bench
    uv run --with modal modal volume put iohunter-bench <local>/data/processed /processed

    uv run --with modal modal run modal_iohunter_gate.py                      # all six
    uv run --with modal modal run modal_iohunter_gate.py --country venezuela  # one
    uv run --with modal modal run --detach modal_iohunter_gate.py             # long grids

No secrets: the benchmark is a public CC-BY-4.0 release
(https://zenodo.org/records/13357621) staged on a volume, and nothing here
touches R2 or the Kenya corpus.
"""

import modal

app = modal.App("kma-iohunter-gate")
vol = modal.Volume.from_name("iohunter-bench", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pandas>=2",
        "pyarrow>=18",
        "numpy>=1.26",
        "networkx>=3.3",
        "scikit-learn>=1.5",
        "scipy>=1.13",
        "torch>=2.4",
        "duckdb>=1.1",
        "python-dotenv>=1.0",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src", ignore=["**/__pycache__/**", "*.pyc"])
)

DATA_DIR = "/data/processed"


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="A10G",
    timeout=60 * 60 * 4,
    # One container per country: the graphs differ in size by three orders of
    # magnitude, so sharing a container would leave cuba serialising behind the
    # small ones.
    max_containers=6,
)
def gate_one(country: str, epoch_grid: tuple[int, ...] | None = None, early_stop: bool = False) -> dict:
    import torch

    from kma import iohunter_gate as gate

    if epoch_grid:
        gate.EPOCH_GRID = tuple(epoch_grid)

    loaded = gate.load_country(DATA_DIR, country)
    unsupervised = gate.gate_report(loaded)
    supervised = gate.gate_report_supervised(loaded, early_stop=early_stop)

    return {
        "country": country,
        "cuda": torch.cuda.is_available(),
        "nodepruning": unsupervised,
        "node2vec_rf": supervised,
    }


@app.function(image=image, volumes={"/data": vol}, timeout=600)
def save_results(results: list[dict]) -> str:
    import json
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"/data/results/gate-{stamp}.json"
    import os

    os.makedirs("/data/results", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    vol.commit()
    return path


@app.local_entrypoint()
def main(country: str = "", epochs: str = "", early_stop: bool = False):
    """`--country` limits to one; `--epochs` is a comma-separated grid."""
    import json

    countries = [country] if country else list(
        ("UAE", "cuba", "russia", "venezuela", "iran", "china")
    )
    grid = tuple(int(x) for x in epochs.split(",")) if epochs else None

    results = list(gate_one.starmap([(c, grid, early_stop) for c in countries]))

    # Persist to the volume as well as printing. A detached run's terminal
    # output is not a durable artifact - its logs can come back empty, and a
    # result that took an hour of GPU time should not depend on catching stdout.
    save_results.remote(results)

    for row in results:
        u, s = row["nodepruning"], row["node2vec_rf"]
        print(
            f"{row['country']:>10}  "
            f"NodePruning {u['macro_f1']:6.2f} vs {u['target']:6.2f} "
            f"({'pass' if u['within_2sd'] else 'FAIL'})   "
            f"node2vec+RF {s['macro_f1']:6.2f} vs {s['target']:6.2f} "
            f"({'pass' if s['within_2sd'] else 'FAIL'}, {s['epochs']} epochs)"
        )
    print()
    print(json.dumps(results, indent=2, default=str))

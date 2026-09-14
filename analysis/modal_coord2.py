"""Modal app: run the v2 detector over a pinned snapshot, off the laptop.

    uv run --with modal modal deploy modal_coord2.py
    # then spawn, so the run survives the local client exiting:
    #   modal.Function.from_name("kma-coord2", "rank").spawn(snapshot="...", top=500)

Why this exists. The run holds every trace plus the fused graph in memory at
once and takes over an hour on this corpus. On the mac it competes with whatever
else is open - a 2026-09-14 run was killed at 90 minutes by the low-memory
watchdog while a game held 3.7 GB - and there is no checkpoint to resume from.
Here the memory is declared rather than negotiated.

No GPU: the text trace is built separately by `modal_textsim.py` and read from
R2. This is graph work, and it is CPU and memory bound.
"""

import modal

app = modal.App("kma-coord2")

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
        "python-igraph>=0.11",
        "leidenalg>=0.10",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src", ignore=["**/__pycache__/**", "*.pyc"])
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("kenya-r2")],
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 6,
)
def rank(
    snapshot: str,
    days: int = 0,
    top: int = 500,
    text_threshold: float = 0.85,
    text_overlap: float = 0.10,
    persist: bool = True,
) -> dict:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from kma import coord2_run
    from kma.db import connect

    result = coord2_run.run(
        snapshot,
        days=days or None,
        top=top,
        text_threshold=text_threshold,
        text_overlap=text_overlap or None,
    )
    share = result.scores["kenya_share"]
    out = {
        "snapshot": snapshot,
        "accounts": len(result.scores),
        "traces": result.trace_sizes.to_dict("records"),
        "kenya_share_mean": float(share.mean()),
        "kenya_share_median": float(share.median()),
    }
    if persist:
        out["key"] = coord2_run.persist(
            connect(),
            result.scores,
            snapshot=snapshot,
            trace_sizes=result.trace_sizes,
            text_threshold=text_threshold,
            text_overlap=text_overlap or None,
        )
    return out


@app.local_entrypoint()
def main(snapshot: str, top: int = 500):
    print(rank.remote(snapshot, top=top))

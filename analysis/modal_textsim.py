"""Modal app: build the text-similarity trace over the Kenya corpus on GPU.

The one trace our data can actually support that we never built. It is 72.8% of
the UAE benchmark's signal and 0% of ours, purely because it was never run: we
already hold the embeddings and the text.

    uv run --with modal modal run modal_textsim.py --snapshot 2026-09-05-promotion-off --threshold 0.85
    uv run --with modal modal run --detach modal_textsim.py --limit 50000   # smoke

Writes `textsim/<snapshot>.parquet` to the `iohunter-bench` volume.

Why GPU: 411k eligible posts is 8.4e10 upper-triangle pairs. The CPU
implementation streams every pair for the caller to filter, which cannot finish;
`coord2.gpu_cosine_pairs` pushes the threshold into the block and keeps only
survivors.

`--overlap` is the word-overlap floor on each surviving pair
(`coord2.TEXT_MIN_OVERLAP` by default, `--overlap 0` for the cosine cut alone).
It goes into the R2 key, so a floored and an unfloored trace never share a path.
"""

import modal

app = modal.App("kma-textsim")
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


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("kenya-r2")],
    gpu="A10G",
    timeout=60 * 60 * 6,
)
def build(snapshot: str, limit: int = 0, percentile: float = 96.0, chunk: int = 1024,
          threshold: float = 0.0, overlap: float = -1.0) -> dict:
    import logging
    import os

    from kma import coord2, textsim_run
    from kma.db import connect

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    edges, info = textsim_run.build(
        connect(), snapshot, limit=limit, percentile=percentile, chunk=chunk,
        threshold=threshold, overlap=overlap, similarity_for=coord2.gpu_cosine_pairs,
    )

    os.makedirs("/data/textsim", exist_ok=True)
    # Threshold, floor and row bound belong in the KEY. Without them a 20k-row
    # smoke overwrites a full corpus run at the same path and the two are then
    # distinguishable only by file timestamp, which is how one of these was
    # briefly mistaken for the other.
    scope = "full" if not limit else f"limit{limit}"
    path = f"/data/textsim/{snapshot}__t{info['threshold']:.2f}__o{info['overlap']:.2f}__{scope}.parquet"
    edges.to_parquet(path)
    vol.commit()
    return {**info, "path": path}


@app.local_entrypoint()
def main(snapshot: str = "2026-09-05-promotion-off", limit: int = 0, percentile: float = 96.0,
         threshold: float = 0.0, overlap: float = -1.0):
    print(build.remote(snapshot, limit, percentile, threshold=threshold, overlap=overlap))

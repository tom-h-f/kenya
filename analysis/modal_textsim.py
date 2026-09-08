"""Modal app: build the text-similarity trace over the Kenya corpus on GPU.

The one trace our data can actually support that we never built. It is 72.8% of
the UAE benchmark's signal and 0% of ours, purely because it was never run: we
already hold the embeddings and the text.

    uv run --with modal modal run modal_textsim.py --snapshot 2026-09-05-promotion-off
    uv run --with modal modal run --detach modal_textsim.py --limit 50000   # smoke

Writes `textsim/<snapshot>.parquet` to the `iohunter-bench` volume.

Why GPU: 411k eligible posts is 8.4e10 upper-triangle pairs. The CPU
implementation streams every pair for the caller to filter, which cannot finish;
`coord2.gpu_cosine_pairs` pushes the threshold into the block and keeps only
survivors.
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
          threshold: float = 0.0) -> dict:
    import numpy as np
    import pandas as pd

    from kma import bench, coord2
    from kma.db import connect

    con = connect()
    manifest = bench.load(snapshot, con=con)
    posts = bench.pinned_source(manifest, "posts")
    embeddings = bench.pinned_source(manifest, "embeddings")

    view = coord2.posts_view(posts)
    rows = coord2.text_rows(con, view)
    if limit:
        rows = rows.head(limit)
    print(f"text-eligible rows: {len(rows)}", flush=True)

    con.register("_rows", rows[["post_id"]])
    vecs = con.sql(
        f"""SELECT e.platform_post_id AS post_id, e.embedding
            FROM {embeddings} e JOIN _rows r ON r.post_id = e.platform_post_id
            QUALIFY row_number() OVER (PARTITION BY e.platform_post_id ORDER BY e.embedded_at DESC) = 1"""
    ).df()
    print(f"rows with an embedding: {len(vecs)}", flush=True)

    joined = rows.merge(vecs, on="post_id", how="inner").reset_index(drop=True)
    matrix = np.vstack(joined["embedding"].to_numpy())
    print(f"matrix {matrix.shape}", flush=True)

    # Estimate the cut on a sample with the exact CPU path, THEN push it into
    # the GPU pass. Estimating with a thresholded sampler would bias it.
    window = float(coord2.TEXT_WINDOW_DAYS) * 86400.0
    times = coord2.epoch_seconds(joined["created_at"])
    if threshold > 0:
        # The percentile reading does not survive contact with a real corpus: a
        # 96th percentile over ALL in-window pairs makes 4% of every pair an
        # edge. Measured 2026-09-07 on 16,478 posts, it resolved to 0.5198 and
        # produced 3,116,957 edges. An absolute cut is what the reference
        # implementation ships (`--tweet_sim_threshold`, default 0.7).
        cut = threshold
        print(f"absolute threshold = {cut:.4f}", flush=True)
    else:
        cut = coord2.pair_similarity_percentile(
            matrix.astype("float64"), times, percentile, window_seconds=window, max_rows=4000
        )
        print(f"{percentile}th percentile threshold = {cut:.4f}", flush=True)

    edges = coord2.text_similarity_network(
        joined[["user_id", "created_at"]],
        matrix,
        threshold=float(cut),
        similarity=coord2.gpu_cosine_pairs(float(cut)),
        chunk=chunk,
    )
    print(f"edges: {len(edges)}", flush=True)

    import os

    os.makedirs("/data/textsim", exist_ok=True)
    # Threshold and row bound belong in the KEY. Without them a 20k-row smoke
    # overwrites a full corpus run at the same path and the two are then
    # distinguishable only by file timestamp, which is how one of these was
    # briefly mistaken for the other.
    scope = "full" if not limit else f"limit{limit}"
    path = f"/data/textsim/{snapshot}__t{cut:.2f}__{scope}.parquet"
    edges.to_parquet(path)
    vol.commit()

    users = len(set(edges["source"]) | set(edges["target"])) if len(edges) else 0
    return {
        "snapshot": snapshot,
        "rows": len(joined),
        "threshold": float(cut),
        "edges": len(edges),
        "users": users,
        "path": path,
    }


@app.local_entrypoint()
def main(snapshot: str = "2026-09-05-promotion-off", limit: int = 0, percentile: float = 96.0,
         threshold: float = 0.0):
    print(build.remote(snapshot, limit, percentile, threshold=threshold))

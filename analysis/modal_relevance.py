"""Modal app: keep the relevance scores current, on GPU.

    cd analysis && uv run --with modal modal run modal_relevance.py --limit 5000   # smoke
    uv run --with modal modal run --detach modal_relevance.py                      # full drain
    uv run --with modal modal deploy modal_relevance.py                            # daily schedule

The classifier decides `kenya_share` in the analysis layer, and posts collected
since the last pass have no score - they fall back to the keyword gate, whose
recall is 0.655. This is what stops that gap growing.

The same shape as `modal_backfill.py`: `db.pending_posts` against the relevance
prefix, score, write a run, repeat until nothing is pending. Reads and writes R2
directly; the weights come from R2 too (`relevance.fetch_model`), cached on a
volume so the ~2 GB download happens once. Secrets: `kenya-r2`.

The enrich worker on tf1 is untouched - it keeps writing its own regex `domain`
column, and no CPU inference is added to that host.
"""

import modal

app = modal.App("kma-relevance")
cache = modal.Volume.from_name("kma-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pandas>=2",
        "pyarrow>=18",
        "duckdb>=1.1",
        "python-dotenv>=1.0",
        "torch>=2.4",
        "transformers>=4.56",
        "boto3>=1.34",
    )
    .env({"PYTHONPATH": "/root/src", "KMA_MODEL_CACHE": "/models"})
    .add_local_dir("src", remote_path="/root/src", ignore=["**/__pycache__/**", "*.pyc"])
)


@app.function(
    image=image,
    volumes={"/models": cache},
    secrets=[modal.Secret.from_name("kenya-r2")],
    gpu="A10G",
    timeout=60 * 60 * 4,
    schedule=modal.Period(days=1),
)
def score_pending(limit: int = 0, chunk: int = 100_000, batch: int = 256) -> dict:
    import pandas as pd

    from kma import relevance
    from kma.db import connect, pending_posts, relevance_source

    con = connect()
    model_dir = relevance.fetch_model()
    cache.commit()

    source = relevance_source("x", relevance.MODEL)
    done, keys = 0, []
    while True:
        want = chunk if not limit else min(chunk, limit - done)
        if want <= 0:
            break
        posts = pending_posts(con, source, "x", want)
        if posts.empty:
            print("nothing pending", flush=True)
            break
        scores = relevance.score_texts(model_dir, posts["text"].fillna("").astype(str).tolist(), batch)
        keys.append(relevance.write_scores(con, pd.DataFrame({
            "platform_post_id": posts["platform_post_id"].astype(str), "p_kenya": scores,
        })))
        done += len(posts)
        print(f"scored {done:,} (last run {keys[-1]})", flush=True)
        if len(posts) < want:
            break

    return {"scored": done, "runs": keys, "model": relevance.MODEL}


@app.local_entrypoint()
def main(limit: int = 0, chunk: int = 100_000, batch: int = 256):
    print(score_pending.remote(limit, chunk, batch))

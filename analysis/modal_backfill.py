"""Modal app: backfill the R2 `hatespeech/` prefix on an A100.

Runs the same `kma.hatespeech.score_new` the enrich worker uses, but on GPU and
in a drain loop, so the whole existing corpus gets scored fast without touching
the RAM-tight enrich host. Reads and writes R2 directly (no Modal volume for
data); the HF weight cache lives on a volume so the ~2GB model downloads once.

    uv run --with modal modal run modal_backfill.py --limit 200   # smoke (from analysis/)
    uv run --with modal modal run --detach modal_backfill.py      # full drain

Secrets (both referenced by name, so they resolve identically locally and in the
container - conditional/file-based secrets break with a dependency-count
mismatch): `huggingface` (HF_TOKEN for the private model) and `kenya-r2` (R2
creds). Create the R2 one once from the repo .env:

    set -a; source .env; set +a
    modal secret create kenya-r2 \\
        R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID \\
        R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY \\
        R2_ACCOUNT_ID=$R2_ACCOUNT_ID R2_BUCKET=$R2_BUCKET
"""

import modal

app = modal.App("kma-hatespeech-backfill")
vol = modal.Volume.from_name("hatespeech-finetune", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pandas>=2",
        "pyarrow>=18",
        "duckdb>=1.1",
        "python-dotenv>=1.0",
        "torch>=2.4",
        "transformers>=4.56",
        "huggingface_hub>=0.30",
    )
    .env({"PYTHONPATH": "/root/src", "HF_HOME": "/root/hf-cache"})
    .add_local_dir(
        "src", remote_path="/root/src",
        ignore=["**/__pycache__/**", "*.pyc"],
    )
)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/root/hf-cache": vol},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("kenya-r2"),
    ],
    timeout=6 * 3600,
)
def run(
    limit: int | None = None, batch_size: int = 256, pass_name: str = "hate"
) -> int:
    """Drain one enrichment prefix on GPU.

    `pass_name="incitement"` runs the coded-incitement NLI instead of the hate
    classifier. That pass is four hypotheses per post through mDeBERTa rather
    than one 3-class forward pass, so its default batch is smaller; pass
    `--batch-size` explicitly when tuning.

    Run incitement BEFORE any hate backfill. `hatespeech._measure_frame`
    resolves `coded_suspect` from the `incitement/` prefix at scoring time and
    writes a non-null False when it finds nothing, so hate rows written first
    need `kma.hatespeech.refresh_measure` afterwards to pick the NLI up.
    """
    from kma.db import connect

    if pass_name == "incitement":
        from kma.incitement import backfill, score_new
    elif pass_name == "hate":
        from kma.hatespeech import backfill, score_new
    else:
        raise ValueError(f"unknown pass {pass_name!r} (expected hate|incitement)")

    if limit is not None:  # smoke: one bounded pass, no full drain
        con = connect()
        n = score_new(con, limit=limit, batch_size=batch_size)
        print(f"smoke [{pass_name}]: scored {n}")
        return n
    total = backfill(batch_size=batch_size)
    print(f"backfilled [{pass_name}] {total}")
    return total


@app.function(
    image=image,
    # No GPU: this reuses the persisted model scores and only recomputes the
    # measurement columns, which are regex and threshold work on CPU.
    volumes={"/root/hf-cache": vol},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("kenya-r2"),
    ],
    timeout=6 * 3600,
)
def refresh_measure(limit: int | None = None, chunk: int = 20_000) -> int:
    """Recompute the persisted measurement columns on already-scored hate rows.

    This is what makes an incitement backfill RETROACTIVE. `_measure_frame`
    resolves `coded_suspect` at hate-scoring time and writes a non-null False
    when no NLI row exists yet, so every row scored before the incitement pass
    reached it is marked not-coded permanently - and the collector's
    hate-seeking reads exactly those persisted flags. Run it after any
    incitement backfill.
    """
    from kma.db import connect
    from kma.hatespeech import refresh_measure as refresh

    n = refresh(connect(), limit=limit, chunk=chunk)
    print(f"refresh_measure: rewrote {n} row(s)")
    return n


# RAPIDS cuML, on its own image so the ~3GB CUDA payload stays out of the
# hate/incitement functions. Measured on an A100 (2026-09-04): UMAP over
# 20k x 768 takes 5.1s and HDBSCAN 0.4s, against a CPU run over the full 658k
# that burned ~8 cores for hours without reaching the write.
topics_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("cuml-cu12", extra_index_url="https://pypi.nvidia.com")
    .pip_install(
        "pandas>=2",
        "pyarrow>=18",
        "duckdb>=1.1",
        "python-dotenv>=1.0",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src", ignore=["**/__pycache__/**", "*.pyc"])
)


@app.function(
    image=topics_image,
    gpu="A100",
    secrets=[modal.Secret.from_name("kenya-r2")],
    timeout=6 * 3600,
)
def topics(
    # 60, not the library default of 25: this is the value narratives.py passed
    # when it still fitted UMAP inline, so the persisted topics match what the
    # notebook used to render.
    min_cluster_size: int = 60,
    n_neighbors: int = 15,
    n_components: int = 5,
    random_state: int | None = 42,
    backend: str = "gpu",
) -> int:
    """Cluster the persisted embeddings and write assignments to R2 `topics/`.

    `notebooks/narratives.py`, `coordination.py` and `desk_brief.py` read the
    result via `semantic.load_topics` rather than fitting UMAP themselves - at
    658k x 768 that is hours of CPU and tens of GB of RSS, which is why this is
    a batch job on GPU.

    cuML and umap-learn are different implementations, so a `backend="cpu"` run
    does not reproduce a GPU one. Do not mix them within a series.
    """
    from kma.db import connect
    from kma.semantic import persist_topics

    n = persist_topics(
        connect(),
        min_cluster_size=min_cluster_size,
        n_neighbors=n_neighbors,
        n_components=n_components,
        random_state=random_state,
        backend=backend,
    )
    print(f"topics: assigned {n} post(s)")
    return n


@app.local_entrypoint()
def main(
    limit: int | None = None,
    batch_size: int = 256,
    pass_name: str = "hate",
    spawn: bool = False,
    refresh: bool = False,
    topic: bool = False,
):
    """`--spawn` for anything long. `run.remote()` BLOCKS on the input, and
    `--detach` only keeps the app alive - it does not stop a client disconnect
    from cancelling the in-flight call. A full drain killed that way dies with
    `InputCancellation: Input was cancelled by user` partway through.

    `run.spawn()` is fire-and-forget: it returns a FunctionCall id immediately
    and the work continues server-side regardless of the client. Same pattern as
    investigations/2026-07-18-hatespeech-finetune/modal_train.py.

        modal run --detach modal_backfill.py --pass-name incitement --spawn
        modal run --detach modal_backfill.py --topic --spawn
        # then poll: modal.FunctionCall.from_id(<id>).get(timeout=0)
    """
    if topic:
        fn, kwargs = topics, {}
    elif refresh:
        fn, kwargs = refresh_measure, {"limit": limit}
    else:
        fn = run
        kwargs = {"limit": limit, "batch_size": batch_size, "pass_name": pass_name}
    if spawn:
        print(f"spawned: {fn.spawn(**kwargs).object_id}")
        return
    print(fn.remote(**kwargs))

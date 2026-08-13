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


@app.local_entrypoint()
def main(
    limit: int | None = None, batch_size: int = 256, pass_name: str = "hate"
):
    print(run.remote(limit=limit, batch_size=batch_size, pass_name=pass_name))

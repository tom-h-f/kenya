"""Modal app: the v2 daily's text trace, built on a GPU before the rank runs.

    cd analysis
    uv run --with modal modal run modal_text_trace.py --days 2 --scratch tt-test --no-embed   # nothing in production
    uv run --with modal modal run modal_text_trace.py                                         # the real window
    uv run --with modal modal deploy modal_text_trace.py                                      # app, NO schedule
    KMA_TEXT_TRACE_SCHEDULE=1 uv run --with modal modal deploy modal_text_trace.py            # app + 02:30 UTC cron

Why this exists: the rank runs on host `cloud` (4 vCPU, 15 GiB, `kma.v2_daily_run`)
and the text trace does not fit there at full embedding coverage - the pair
search is quadratic in the window's vectors. This runs the trace on an A10G
and leaves it in R2 under the key the cloud runner already looks for, so the
cloud job reuses it. R2 is the only coupling: the cloud job has no Modal
credentials and imports no `modal`.

What it does (all of it is `kma.text_trace.run`, tested offline):

1. embeds the window's text-eligible posts that have no vector;
2. pins the window manifest, rebuilding it only if step 1 wrote vectors that an
   existing manifest cannot see;
3. builds the trace at the production cut unless it already exists.

`end` defaults to yesterday UTC and `days` to `v2_daily_run.DEFAULT_DAYS`, read
at run time from the code deployed here. The cloud job's window id has to match
this one, so redeploy this app when `DEFAULT_DAYS` changes; a mismatch is not
silent - the cloud run reports `text_trace=missing`.

Not scheduled by default. The cron is gated by `KMA_TEXT_TRACE_SCHEDULE=1` at
deploy time and ships off, so a plain `modal deploy` creates the app and bills
nothing until someone runs it or re-deploys with the flag. 02:30 UTC leaves an
hour before the cloud timer at 03:30.

`--scratch <prefix>` holds the manifest in memory and writes the trace (and the
manifest it read, for an equivalence run) under `<prefix>/` in the bucket.
"""

import os

import modal

app = modal.App("kma-text-trace")
models = modal.Volume.from_name("kma-models", create_if_missing=True)

SCHEDULE_ENV = "KMA_TEXT_TRACE_SCHEDULE"
CRON = "30 2 * * *"
schedule = modal.Cron(CRON) if os.environ.get(SCHEDULE_ENV) == "1" else None

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
        "sentence-transformers>=5.6",
        "duckdb>=1.1",
        "python-dotenv>=1.0",
        "boto3>=1.35",
    )
    .env({"PYTHONPATH": "/root/src", "HF_HOME": "/models/hf"})
    .add_local_dir("src", remote_path="/root/src", ignore=["**/__pycache__/**", "*.pyc"])
)


@app.function(
    image=image,
    volumes={"/models": models},
    secrets=[modal.Secret.from_name("kenya-r2")],
    gpu="A10G",
    timeout=60 * 60 * 2,
    retries=modal.Retries(max_retries=1, initial_delay=60.0),
    schedule=schedule,
)
def build_trace(end: str = "", days: int = 0, scratch: str = "", embed: bool = True,
                force: bool = False) -> dict:
    import logging
    import time

    import torch

    from kma import text_trace, v2_daily_run
    from kma.db import connect

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.monotonic()
    con = connect()
    con.execute("SET enable_progress_bar = false")
    out = text_trace.run(
        end or None, days or v2_daily_run.DEFAULT_DAYS, con=con,
        scratch=scratch, embed=embed, force=force,
    )
    models.commit()
    return {
        **out,
        "wall_s": round(time.monotonic() - started, 1),
        "peak_rss_mb": v2_daily_run.peak_rss_mb(),
        "gpu_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "gpu": torch.cuda.get_device_name(0),
    }


@app.local_entrypoint()
def main(end: str = "", days: int = 0, scratch: str = "", no_embed: bool = False,
         force: bool = False):
    import json

    print(json.dumps(
        build_trace.remote(end, days, scratch, not no_embed, force), indent=2, default=str
    ))

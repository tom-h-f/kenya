"""Modal app: v2 over a rolling window of the live corpus, every day.

    cd analysis
    uv run --with modal modal run modal_v2_daily.py --days 7 --no-persist   # smoke
    uv run --with modal modal run modal_v2_daily.py                         # one real run
    uv run --with modal modal deploy modal_v2_daily.py                      # the schedule

Until this existed v2 ran only by hand, against frozen snapshots, while the
continuous detector was v1 (18.6% recall on the IO archive, 0 of 11
Kenya-relevant in the size-matched blind read). See
`docs/plans/2026-09-22-detection-readiness.md`, workstream A.

Stages, each on the hardware it needs:

1. `run_daily` (small CPU) pins the window - `kma.window`, a bench snapshot
   named `daily-<end>-<days>d` - and drives the rest. It is the scheduled
   function: 03:30 UTC, so yesterday's `dt` partition is closed.
2. Relevance: calls the deployed `kma-relevance` app's `score_pending` first.
   Its own daily schedule stopped writing after 2026-09-14 15:57 (last
   `scored_at`, measured 2026-09-22), and the community report filters on it.
   A failure here is recorded and does not stop the run - unscored communities
   are kept and flagged, not dropped.
3. `textsim` (A10G) builds the text trace for the window, unless it exists.
4. `rank_window` (8 CPU, 32 GB) - traces, fusion, centrality, the community
   report, and every `coord2/kind=daily_*` table. Sized like `kma-coord2.rank`,
   which holds every trace plus the fused graph in memory at once.

Windowed detection (workstream D) plugs in as an `extra_passes` entry in
`rank_window`; see `kma.v2_daily`.

`--no-persist` suppresses the `daily_*` result tables only. The window
manifest and the text trace are still written: both are keyed by window id,
reused by the next run over the same window, and the manifest is what makes a
run reproducible at all.

## Measured, 2026-09-22

The Modal workspace was over its spend limit, so the 7-day smoke ran the same
orchestration on the mac (torch CPU for the pair search): window
`daily-2026-09-21-7d`, 741,613 post rows, 353s end to end (manifest 84s, text
trace 168s, rank 100s), 3.5 GB peak RSS. 5,054 accounts, 101,464 edges, 30
communities of 4+, 27 kept, 500 accounts listed from 25.

Two upstream gaps it exposed, both outside this app:

- **Embeddings lag collection.** 26,788 of 190,398 text-eligible posts in the
  window (14%) had a vector, so the text trace held 909 edges. The tf1 enrich
  worker embedded 500 posts per pass, five passes on 2026-09-21. The text
  trace is only as current as the embedding backlog.
- **Relevance stopped scoring on 2026-09-14** (1,523 of 9,989 listed posts
  scored). `kma-relevance` last billed on 2026-09-14, consistent with the
  spend limit stopping its schedule too. Stage 2 above re-runs it; until it
  can, new communities come out `relevance_scored=False`.
"""

import modal

app = modal.App("kma-v2-daily")

_base = (
    "pandas>=2",
    "pyarrow>=18",
    "numpy>=1.26",
    "networkx>=3.3",
    "scikit-learn>=1.5",
    "scipy>=1.13",
    "duckdb>=1.1",
    "python-dotenv>=1.0",
    "boto3>=1.35",
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*_base, "python-igraph>=0.11", "leidenalg>=0.10")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src", ignore=["**/__pycache__/**", "*.pyc"])
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*_base, "torch>=2.4")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src", ignore=["**/__pycache__/**", "*.pyc"])
)

secrets = [modal.Secret.from_name("kenya-r2")]


def _logging() -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@app.function(image=gpu_image, secrets=secrets, gpu="A10G", timeout=60 * 60 * 6)
def textsim(window_id: str, threshold: float = 0.85, overlap: float = -1.0) -> dict:
    _logging()
    from kma import coord2, textsim_run
    from kma.db import connect

    _, info = textsim_run.build(
        connect(), window_id, threshold=threshold, overlap=overlap,
        similarity_for=coord2.gpu_cosine_pairs,
    )
    return info


@app.function(image=image, secrets=secrets, cpu=8, memory=32768, timeout=60 * 60 * 6)
def rank_window(end: str, days: int, persist: bool = True) -> dict:
    _logging()
    from kma import v2_daily, window
    from kma.db import connect

    result = v2_daily.rank(connect(), window.Window.ending(end, days), persist=persist)
    return {**result.summary, "keys": result.keys}


@app.function(
    image=image,
    secrets=secrets,
    cpu=1,
    memory=4096,
    timeout=60 * 60 * 20,
    schedule=modal.Cron("30 3 * * *"),
)
def run_daily(end: str = "", days: int = 90, persist: bool = True,
              score_relevance: bool = True) -> dict:
    _logging()
    import time

    from kma import v2_daily
    from kma.db import connect

    relevance: dict = {"skipped": True}
    if score_relevance:
        t = time.monotonic()
        try:
            got = modal.Function.from_name("kma-relevance", "score_pending").remote()
            relevance = {"scored": got.get("scored"), "model": got.get("model")}
        except Exception as exc:  # noqa: BLE001 - recorded, and the rank still runs
            relevance = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        relevance["s"] = round(time.monotonic() - t, 1)

    out = v2_daily.orchestrate(
        end or None,
        days,
        textsim=lambda window_id: textsim.remote(window_id),
        rank_fn=lambda w: rank_window.remote(w.end.isoformat(), w.days, persist),
        con=connect(),
    )
    return {"relevance": relevance, **out}


@app.local_entrypoint()
def main(end: str = "", days: int = 90, persist: bool = True, score_relevance: bool = True):
    print(run_daily.remote(end, days, persist, score_relevance))

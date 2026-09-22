"""Modal app: the daily leads pass. Diff, judge what changed, persist.

    uv run --with modal modal run modal_leads.py --dry-run
    uv run --with modal modal run modal_leads.py --source v1 --max-judged 5 --mode validation
    uv run --with modal modal deploy modal_leads.py      # schedules the daily pass

Everything but the reader lives in `kma.leads_run`, so the decisions are tested
without Modal or a key. This file is only where it runs, on what schedule, and
with which secrets: the reader needs `ANTHROPIC_API_KEY`, which by the
2026-09-08 decision stays off tf1 and pi0, so judging happens here and tf1 only
ever reads the verdicts back out of R2 (`kma.leads_notify`).

The schedule sits after the daily v2 pass, which writes the community listing
this diffs. If the listing has not moved since the last pass, the diff finds
nothing new and nothing is spent.

## Secrets

Both referenced unconditionally by name. Conditional or file-based secrets break
with a local/remote dependency-count mismatch, measured on `modal_backfill.py`.
"""

import modal

app = modal.App("kma-leads")

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
        "anthropic>=0.40",
        "python-igraph>=1.0",
        "leidenalg>=0.12",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src", ignore=["**/__pycache__/**", "*.pyc"])
)

SECRETS = [
    modal.Secret.from_name("kenya-r2"),
    modal.Secret.from_name("anthropic"),
]


def _pass(source: str, limit: int, dry_run: bool, max_judged: int, mode: str) -> dict:
    import logging
    import os

    from kma import adjudicate, leads_run
    from kma.db import connect

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not dry_run and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is absent. Create the secret once:\n"
            "    modal secret create anthropic ANTHROPIC_API_KEY=sk-...\n"
            "Or pass --dry-run to plan the queue without judging."
        )
    return leads_run.execute(
        connect(),
        source=source,
        limit=limit,
        judge=None if dry_run else adjudicate.judge_all,
        persist=not dry_run,
        mode=mode,
        max_judged=max_judged or None,
    )


@app.function(image=image, secrets=SECRETS, timeout=60 * 60 * 2)
def leads_pass(
    source: str = "v2",
    limit: int = 10,
    dry_run: bool = False,
    max_judged: int = 0,
    mode: str = "production",
) -> dict:
    return _pass(source, limit, dry_run, max_judged, mode)


# 06:30 UTC daily: after the daily v2 listing (workstream A) has landed. Only
# active once this app is deployed.
@app.function(image=image, secrets=SECRETS, timeout=60 * 60 * 2,
              schedule=modal.Cron("30 6 * * *"))
def daily() -> dict:
    return _pass("v2", 10, False, 0, "production")


@app.local_entrypoint()
def main(source: str = "v2", limit: int = 10, dry_run: bool = False,
         max_judged: int = 0, mode: str = "production"):
    import json

    print(json.dumps(leads_pass.remote(source=source, limit=limit, dry_run=dry_run,
                                       max_judged=max_judged, mode=mode),
                     indent=2, default=str))

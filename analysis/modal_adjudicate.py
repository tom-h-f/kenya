"""Modal app: layer three. Read v2 output, build dossiers, adjudicate, persist.

    uv run --with modal modal run modal_adjudicate.py --dry-run --limit 5
    uv run --with modal modal run modal_adjudicate.py --limit 20

Why this exists on Modal rather than tf1: the decision of 2026-09-08 is that no
`ANTHROPIC_API_KEY` goes on the collector or enrichment hosts and all model work
runs here. The tf1 service it replaces sat dormant for two weeks behind a 401
that looked like a no-op, which is why this one fails loudly instead.

Statistics answer *do these accounts act together* and cannot answer *why* -
settled 2026-08-17 as a property of the method, not a tuning gap. This is the
step that answers why, and its output is a reader's judgement recorded with the
evidence that produced it, never a label.

## The unit is the campaign, the score is the account

Per A3 of `docs/plans/2026-09-08-finishing-the-revamp.md`. Adjudicating
confirmed operations on 2026-08-15, a reader called 4 of 6 correctly and BOTH
misses were the operation's pure engagement-bait clusters, whose content is
indistinguishable from our own reciprocal pods. A bare per-account verdict is
therefore unanswerable for exactly the accounts that matter most. Verdicts
render on campaigns and inherit one-directionally to members.

`--unit` takes `campaign` (default) or `cluster` so the choice stays visible
rather than baked in.

## Secrets

Both referenced unconditionally by name. Conditional or file-based secrets break
with a local/remote dependency-count mismatch, measured on `modal_backfill.py`.

    modal secret create anthropic ANTHROPIC_API_KEY=sk-...
"""

import modal

app = modal.App("kma-adjudicate")
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


@app.function(image=image, volumes={"/data": vol}, secrets=SECRETS, timeout=60 * 60 * 3)
def adjudicate_run(
    limit: int = 20,
    unit: str = "campaign",
    model: str = "",
    dry_run: bool = False,
) -> dict:
    import os

    import pandas as pd

    from kma import adjudicate, coordination, db, dossier
    from kma.db import connect

    if not dry_run and not os.getenv("ANTHROPIC_API_KEY"):
        # Loudly, and naming the fix. The tf1 predecessor's failure read as a
        # no-op for two weeks; that is the failure mode being designed out.
        raise RuntimeError(
            "ANTHROPIC_API_KEY is absent. Create the secret once:\n"
            "    modal secret create anthropic ANTHROPIC_API_KEY=sk-...\n"
            "Or pass --dry-run to build dossiers without judging."
        )

    con = connect()
    members = db.coordination_run_latest(con, kind="clusters").df()
    if members.empty:
        raise RuntimeError("no persisted clustering to adjudicate")

    if unit == "campaign":
        acts = coordination.traces(con, "co_retweet").df()
        groups = coordination.campaigns(members, acts)
        # Campaign ids replace cluster ids so a verdict is rendered on the unit
        # a reader can actually judge, then inherited by members for scoring.
        members = members.merge(groups, on="cluster_id", how="left")
        key = "campaign_id" if "campaign_id" in members.columns else "cluster_id"
    elif unit == "cluster":
        key = "cluster_id"
    else:
        raise ValueError(f"unit must be campaign or cluster, got {unit!r}")

    ranked = (
        members.groupby(key).size().sort_values(ascending=False).head(limit).index.tolist()
    )
    packets = dossier.build(con, members, cluster_ids=ranked)
    print(f"built {len(packets)} dossier(s) on unit={unit} (key={key})", flush=True)

    if dry_run:
        return {
            "unit": unit,
            "packets": len(packets),
            "ids": [str(p.get("cluster_id")) for p in packets],
            "judged": 0,
        }

    chosen = model or adjudicate.MODEL
    verdicts = adjudicate.judge_all(packets, model=chosen)
    frame = pd.DataFrame(verdicts)
    # Provenance travels as columns, not as prose in a log: a verdict without
    # its model, prompt and dossier run cannot be audited or superseded.
    frame["adjudicator"] = chosen
    frame["unit"] = unit
    key_written = coordination.persist_verdicts(con, frame, members, adjudicator=chosen)

    return {
        "unit": unit,
        "packets": len(packets),
        "judged": len(frame),
        "model": chosen,
        "verdicts_key": key_written,
        "types": frame["cluster_type"].value_counts().to_dict() if "cluster_type" in frame else {},
    }


@app.local_entrypoint()
def main(limit: int = 20, unit: str = "campaign", model: str = "", dry_run: bool = False):
    print(adjudicate_run.remote(limit=limit, unit=unit, model=model, dry_run=dry_run))

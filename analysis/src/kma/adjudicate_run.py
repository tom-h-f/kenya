"""Adjudicate the latest coordination run: dossiers in, verdicts to R2.

    python -m kma.adjudicate_run                # dry run, prints the queue
    python -m kma.adjudicate_run --persist      # write verdicts to R2

Layer three on a timer. Layers one and two - detect coordination, score it -
already run unattended and produce a cluster count. That count means
"coordination happened", which on this corpus is dominated by reciprocal
engagement pods, so on its own it is not a finding. This turns it into a triage
queue with a reason attached to every row.

Why candidates are narrowed before adjudication: reading is the expensive step
and most clusters are not worth a reader's attention. The default is
corroborated clusters (>= 2 channels), which is the population the dashboard
already publishes, plus anything a Kenya-share above `--min-kenya` drags in.

Requires ANTHROPIC_API_KEY. Without one it writes the prompts instead and exits
non-zero, so a scheduled run fails loudly rather than silently producing no
verdicts.

Not a verdict about people. `kma.adjudicate` returns a triage judgement about a
CLUSTER; nothing here may be published as a claim about a named account, and
the verdict prefix is private for the same reason `kma.dossier` is.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import duckdb
import pandas as pd

from kma import adjudicate as adj
from kma import coordination as co
from kma import dossier
from kma.db import connect, coordination_run_latest

log = logging.getLogger("kma")

DEFAULT_MAX_CLUSTERS = int(os.getenv("ADJUDICATE_MAX_CLUSTERS", "60"))


def candidates(
    members: pd.DataFrame,
    scorecards: pd.DataFrame | None,
    min_kenya: float = 0.0,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
) -> list:
    """Which clusters are worth a reader, largest first.

    Size-ordered rather than inauthenticity-ordered on purpose: the index is a
    within-run percentile blend that ranks tiny clusters highly (a 3-account
    group trivially looks concealed and homogeneous), and the ground-truth work
    found the substantive clusters are the large ones."""
    if members.empty:
        return []
    sizes = members.groupby("cluster_id")["author_id"].nunique()
    if "n_channels" in members.columns:
        ch = members[["cluster_id", "n_channels"]].drop_duplicates().set_index("cluster_id")
        keep = set(ch[ch["n_channels"] >= 2].index)
    else:
        keep = set(sizes.index)
    if min_kenya > 0 and scorecards is not None and "kenya_share" in scorecards.columns:
        keep |= set(
            scorecards.loc[scorecards["kenya_share"] >= min_kenya, "cluster_id"]
        )
    ranked = sizes[sizes.index.isin(keep)].sort_values(ascending=False)
    return ranked.head(max_clusters).index.tolist()


def _cluster_source(
    con: duckdb.DuckDBPyConnection, platform: str
) -> tuple[pd.DataFrame, str]:
    """The looser triage clustering when it exists, else the published one."""
    from kma import db

    for kind in ("triage_clusters", "clusters"):
        try:
            if not db.prefix_readable(con, db.coordination_source(kind, platform)):
                continue
            frame = coordination_run_latest(con, kind, platform).df()
        except duckdb.Error:
            continue
        if len(frame):
            return frame, kind
    return pd.DataFrame(columns=["cluster_id", "author_id"]), "none"


def run(
    con: duckdb.DuckDBPyConnection,
    persist: bool = False,
    platform: str = "x",
    min_kenya: float = 0.0,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    campaigns: bool = True,
    client=None,
) -> dict:
    # Prefer the looser triage clustering. It is where the recall is - 18.6% ->
    # ~40% against confirmed operations, at zero measured null yield - and it
    # exists precisely so a reader sees what the published strict setting misses.
    # Falls back silently to `kind=clusters`, which is what every run before the
    # dual-resolution change produced.
    members, source = _cluster_source(con, platform)
    out: dict = {"n_candidates": 0, "n_verdicts": 0, "verdicts_key": None,
                 "cluster_source": source, "n_campaigns": None,
                 "n_inherited": 0, "by_type": {}}
    if members.empty:
        log.info("no persisted clusters to adjudicate")
        return out

    try:
        cards = coordination_run_latest(con, "scorecards", platform).df()
    except duckdb.Error:
        cards = None
        log.info("no scorecards yet; adjudicating on dossiers alone")

    ids = candidates(members, cards, min_kenya=min_kenya, max_clusters=max_clusters)
    out["n_candidates"] = len(ids)
    if not ids:
        log.info("no candidate clusters")
        return out
    log.info("adjudicating %d cluster(s)", len(ids))

    # ONE call for every candidate. A dossier build is three fixed whole-prefix
    # R2 scans whose cost does not grow with the number of clusters requested -
    # measured 1,333s for four - so per-cluster builds would pay that repeatedly.
    t0 = time.monotonic()
    packets = dossier.build(
        con, members[["cluster_id", "author_id"]], scorecards=cards,
        platform=platform, cluster_ids=ids,
    )
    log.info("built %d dossier(s) in %.0fs", len(packets), time.monotonic() - t0)
    if not packets:
        return out

    if not os.getenv("ANTHROPIC_API_KEY") and client is None:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Verdicts cannot be produced.\n"
            "Run `python -m kma.adjudicate_run --prompts DIR` to write the "
            "packets for a human reader instead."
        )

    verdicts = adj.judge_all(packets, client=client)
    frame = pd.DataFrame(verdicts)
    out["n_verdicts"] = int(len(frame))

    # Campaign inheritance. A cluster of pure engagement bait cannot be judged
    # on its own content - measured, it is the exact case a reader misses - but
    # if it pushes the same accounts as a cluster carrying a political payload,
    # they are one operation and it inherits. One-directional: an operation is
    # never downgraded by the pods around it.
    if campaigns and len(frame):
        try:
            acts = co._amplification_acts(con, platform)
            cmap = co.campaigns(members[["cluster_id", "author_id"]], acts)
            before = frame["cluster_type"].value_counts().to_dict()
            frame = co.inherit_verdicts(frame, cmap)
            out["n_campaigns"] = int(cmap["campaign_id"].nunique())
            out["n_inherited"] = int(frame["inherited_from"].notna().sum())
            log.info(
                "campaigns: %d over %d clusters; %d verdict(s) inherited (%s -> %s)",
                out["n_campaigns"], len(cmap), out["n_inherited"],
                before, frame["cluster_type"].value_counts().to_dict(),
            )
        except Exception:
            log.exception("campaign linking failed; verdicts stand as judged")

    if "cluster_type" in frame.columns:
        out["by_type"] = frame["cluster_type"].value_counts().to_dict()

    if persist and len(frame):
        out["verdicts_key"] = co.persist_verdicts(
            con, frame, members, platform=platform, adjudicator=adj.MODEL
        )
        log.info("persisted %s", out["verdicts_key"])
    return out


def write_prompts(con, directory: str, platform: str = "x", **kw) -> int:
    """The no-key path: packets to disk for a human reader."""
    from pathlib import Path

    members, _ = _cluster_source(con, platform)
    if members.empty:
        return 0
    try:
        cards = coordination_run_latest(con, "scorecards", platform).df()
    except duckdb.Error:
        cards = None
    ids = candidates(members, cards, **kw)
    packets = dossier.build(
        con, members[["cluster_id", "author_id"]], scorecards=cards,
        platform=platform, cluster_ids=ids,
    )
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    for item in adj.prompts(packets):
        (out / f"cluster-{item['cluster_id']}.txt").write_text(item["prompt"])
    return len(packets)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--persist", action="store_true", help="write verdicts to R2")
    p.add_argument("--platform", default="x")
    p.add_argument(
        "--min-kenya", type=float, default=0.0,
        help="also adjudicate clusters whose Kenya share is at least this",
    )
    p.add_argument("--max-clusters", type=int, default=DEFAULT_MAX_CLUSTERS)
    p.add_argument(
        "--no-campaigns", action="store_true",
        help="do not let a cluster inherit its campaign's verdict",
    )
    p.add_argument(
        "--prompts", metavar="DIR",
        help="write packets for a human reader instead of calling a model",
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    con = connect()
    con.execute("SET enable_progress_bar=false")
    __import__("kma.coordination_run", fromlist=["tune"]).tune(con)

    if args.prompts:
        n = write_prompts(
            con, args.prompts, platform=args.platform,
            min_kenya=args.min_kenya, max_clusters=args.max_clusters,
        )
        print(json.dumps({"prompts_written": n, "dir": args.prompts}, indent=2))
        return

    print(json.dumps(run(
        con, persist=args.persist, platform=args.platform,
        min_kenya=args.min_kenya, max_clusters=args.max_clusters,
        campaigns=not args.no_campaigns,
    ), indent=2, default=str))


if __name__ == "__main__":
    main()

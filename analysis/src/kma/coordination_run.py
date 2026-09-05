"""Headless coordination run: build multiplex layers, cluster, optionally persist.

The pipeline in `kma.coordination` was only ever reachable from a checkbox in
`notebooks/coordination.py`, so nothing was ever written to the R2
`coordination/` prefix. That left `adaptive.cluster_accounts` on the collector
side returning an empty list forever - the analysis -> collection handoff was
wired up but dead. This module is the missing entrypoint.

    python -m kma.coordination_run                      # dry run, prints summary
    python -m kma.coordination_run --persist            # write to R2

Persisted clusters feed the collector's targeting. That is a real feedback
loop: promoted accounts get collected more, so their post counts and co-action
degree inflate, and they look more coordinated next run. The degree-corrected
null corrects for object popularity, not for author over-sampling. The
collector mitigates this by routing promoted accounts to quarantined
`type=cib_timeline` / `type=hate_*` partitions that are excluded from every
rate computation (see `kma.db.TARGETED_TYPES`); do not undo that by adding
promoted accounts to the baseline target list.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import duckdb

from kma import coordination as co
from kma.db import connect

log = logging.getLogger("kma")

# Not WAVE_A. On the full corpus (manipulation sweep, 2026-07-17) `text_sim`
# and `fast_co_share` validated zero edges under the degree-corrected null -
# copypasta exists but sits below significance. Running them costs a full
# projection and Monte-Carlo shuffle per pass for a guaranteed empty layer.
DEFAULT_CHANNELS = ["co_retweet", "co_reply"]

EDGE_METHODS = [("svn_fdr", "sig_fdr"), ("svn_bonf", "sig_bonferroni"), ("pct", "sig_percentile")]

# DuckDB sizes its memory budget from the HOST, not the container cgroup, so on a
# small box with a generous `mem_limit` it happily plans past physical RAM and
# dies mid-projection instead of spilling. Observed on tf1 (3.7Gi host, 6g
# mem_limit): OutOfMemoryException at `projected_edges`, "2.7 GiB/2.9 GiB used".
# An explicit limit makes it spill to disk instead - slower, but it finishes.
COORD_MEMORY_LIMIT = os.getenv("COORD_MEMORY_LIMIT", "1500MB")
COORD_THREADS = int(os.getenv("COORD_THREADS", "2"))


def tune(con: duckdb.DuckDBPyConnection | None) -> None:
    """Constrain DuckDB so the projection spills rather than OOMs.

    Best-effort: settings names drift between DuckDB versions, and a tuning
    failure must not take down a run that would otherwise have succeeded."""
    if con is None:
        return
    for stmt in (
        f"SET memory_limit='{COORD_MEMORY_LIMIT}'",
        f"SET threads={int(COORD_THREADS)}",
        # The projection is a large self-join; row order is irrelevant to the
        # result and preserving it costs memory.
        "SET preserve_insertion_order=false",
    ):
        try:
            con.execute(stmt)
        except Exception:
            log.warning("coordination: could not apply %r", stmt)


def run(
    con: duckdb.DuckDBPyConnection,
    channels: list[str] | None = None,
    persist: bool = False,
    platform: str = "x",
    resolution: float = co.DEFAULT_RESOLUTION,
    min_size: int = 3,
    method: str = co.DEFAULT_EDGE_METHOD,
    scorecards: bool = False,
    triage_resolution: float = co.TRIAGE_RESOLUTION,
) -> dict:
    """One coordination pass. Returns a summary dict; writes nothing unless
    `persist`. Channels default to the two that validate on live data."""
    channels = channels or DEFAULT_CHANNELS
    tune(con)
    channel_stats: dict[str, dict] = {}
    layers = co.build_layers(
        con, channels=channels, platform=platform, method=method, stats=channel_stats
    )
    members, summary = co.clusters(layers, resolution=resolution, min_size=min_size)

    # Corroboration (a cluster supported by >= 2 channels) is the outcome
    # measure for the census work, and until now it was computed nowhere:
    # `adaptive.cluster_accounts` and `netviz` only read `n_channels` back out
    # of persisted parquet, so the "14 corroborated (60 accounts)" baseline in
    # docs/analysis/census-tuning.md was derived by hand and never reproduced.
    corr_ids: set = set()
    if len(summary) and "n_channels" in summary.columns:
        corr_ids = set(summary.loc[summary["n_channels"] >= 2, "cluster_id"])
    n_corr_accounts = (
        int(members["cluster_id"].isin(corr_ids).sum()) if len(members) else 0
    )
    # Read next to n_corroborated_clusters, always. Until bridge accounts are in
    # the tens that number means "cannot be evaluated" rather than "nothing is
    # coordinated" - a single bridge account can produce a corroborated cluster.
    overlap = co.layer_overlap(layers)
    for ch, edges in layers.items():
        if ch in channel_stats:
            channel_stats[ch]["layer_weight_max"] = co.layer_weight_max(edges)

    log.info(
        "layers: %s; clusters: %d (%d accounts); corroborated: %d (%d accounts); "
        "bridge accounts: %d; shared pairs: %d",
        {ch: len(e) for ch, e in layers.items()},
        len(summary),
        len(members),
        len(corr_ids),
        n_corr_accounts,
        overlap["bridge_accounts"],
        overlap["shared_pairs"],
    )

    out: dict = {
        "channels": channels,
        "edges": {ch: int(len(e)) for ch, e in layers.items()},
        "n_clusters": int(len(summary)),
        "n_accounts": int(len(members)),
        "n_corroborated_clusters": len(corr_ids),
        "n_corroborated_accounts": n_corr_accounts,
        **overlap,
        "channel_stats": channel_stats,
        "metrics_key": None,
        "scorecards_key": None,
        "triage_clusters_key": None,
        "persisted": [],
    }
    if not persist:
        return out

    # Metrics first, deliberately. A run that detects nothing returns at
    # `members.empty` below, and a run whose edge write fails never reaches the
    # end - both are exactly the runs whose counters you want. The persisted
    # keys are left out of the row because an R2 listing recovers them; the
    # counters are not recoverable after the fact.
    try:
        out["metrics_key"] = co.persist_run_metrics(
            con,
            channel_stats,
            {
                "channels": channels,
                "method": method,
                "resolution": float(resolution),
                "min_size": int(min_size),
                "n_clusters": int(len(summary)),
                "n_accounts": int(len(members)),
                "n_corroborated_clusters": len(corr_ids),
                "n_corroborated_accounts": n_corr_accounts,
                **overlap,
            },
            platform=platform,
        )
        log.info("persisted %s", out["metrics_key"])
    except Exception:
        # `enrich._coordination_pass` reschedules the whole 6-hourly pass on a
        # non-zero rc. Re-running a full projection because a ~4KB metrics write
        # hit a transient R2 error is a bad trade on this host; a missing run
        # row shows up as a gap in the series, which is visible and harmless.
        log.exception("coordination: run metrics write failed (continuing)")

    if members.empty:
        log.info("nothing to persist: no clusters detected")
        return out

    keys = []
    for ch, edges in layers.items():
        for name, col in EDGE_METHODS:
            subset = edges[edges[col]]
            if len(subset):
                keys.append(co.persist_edges(con, subset, ch, name, platform=platform))
    names = co.cluster_names(con, members, summary, platform=platform)
    named = summary.merge(names, on="cluster_id", how="left") if len(names) else summary
    keys.append(co.persist_clusters(con, members, named, platform=platform))

    # A SECOND clustering of the same validated edges at a lower resolution,
    # written to its own prefix. Cheap - Leiden over edges already in memory, no
    # re-projection - and it is where the recall is: 18.6% -> ~40% against
    # confirmed operations, at zero measured null yield. Kept out of
    # `kind=clusters` so the published count and the collector's targeting are
    # untouched, and wrapped so a failure cannot cost the run of record.
    if triage_resolution and triage_resolution != resolution:
        try:
            t_members, t_summary = co.clusters(
                layers, resolution=triage_resolution, min_size=min_size
            )
            if len(t_members):
                key = co.persist_clusters(
                    con, t_members, t_summary, platform=platform,
                    kind="triage_clusters",
                )
                keys.append(key)
                out["triage_clusters_key"] = key
            log.info(
                "triage clustering at resolution %.4f: %d clusters (%d accounts) "
                "against %d (%d) published",
                triage_resolution, len(t_summary), len(t_members),
                len(summary), len(members),
            )
        except Exception:
            log.exception("coordination: triage clustering failed (continuing)")

    # Off by default, and that is a measurement not a preference: scoring the
    # 2026-08-14 run took 2,276s on an idle laptop, against a 6h timer on a
    # contended 2-core host. `kma.scorecard_run` is the production path - it
    # reads the run back out of R2 instead of recomputing the projection, so it
    # can sit on its own slower timer. This stays for ad-hoc single-command use.
    #
    # Written LAST and never allowed to take the run with it. Edges and clusters
    # are the artifacts of record - the collector's targeting reads them -
    # whereas a missing scorecard run is a gap in a series nothing blocks on.
    if scorecards:
        try:
            t0 = time.monotonic()
            cards = co.scorecards(con, members, layers, platform=platform)
            if len(cards):
                key = co.persist_scorecards(con, cards, members, platform=platform)
                keys.append(key)
                out["scorecards_key"] = key
            log.info(
                "scorecards: %d cluster(s) scored in %.1fs",
                len(cards), time.monotonic() - t0,
            )
        except Exception:
            log.exception("coordination: scorecard write failed (continuing)")

    for k in keys:
        log.info("persisted %s", k)
    out["persisted"] = keys
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--channels",
        default=",".join(DEFAULT_CHANNELS),
        help=f"comma-separated (default {','.join(DEFAULT_CHANNELS)}; "
        f"available: {','.join(co.WAVE_A + co.WAVE_B)})",
    )
    p.add_argument("--persist", action="store_true", help="write results to R2")
    p.add_argument("--platform", default="x")
    p.add_argument("--resolution", type=float, default=co.DEFAULT_RESOLUTION)
    p.add_argument("--min-size", type=int, default=3)
    p.add_argument(
        "--method",
        default=co.DEFAULT_EDGE_METHOD,
        choices=["bonferroni", "fdr", "percentile"],
        help="edge filter feeding community detection (see co.DEFAULT_EDGE_METHOD)",
    )
    p.add_argument(
        "--triage-resolution", type=float, default=co.TRIAGE_RESOLUTION,
        help="second, looser clustering for adjudication only (0 disables); "
             "never feeds the published count or collector targeting",
    )
    p.add_argument(
        "--scorecards",
        action="store_true",
        help="also score the clusters (~40min; normally left to kma.scorecard_run)",
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    unknown = set(channels) - set(co.CHANNELS)
    if unknown:
        p.error(f"unknown channel(s): {', '.join(sorted(unknown))}")

    con = connect()
    con.execute("SET enable_progress_bar=false")
    out = run(
        con,
        channels=channels,
        persist=args.persist,
        platform=args.platform,
        resolution=args.resolution,
        min_size=args.min_size,
        method=args.method,
        scorecards=args.scorecards,
        triage_resolution=args.triage_resolution,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

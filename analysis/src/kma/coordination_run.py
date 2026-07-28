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


def run(
    con: duckdb.DuckDBPyConnection,
    channels: list[str] | None = None,
    persist: bool = False,
    platform: str = "x",
    resolution: float = co.DEFAULT_RESOLUTION,
    min_size: int = 3,
    method: str = "fdr",
) -> dict:
    """One coordination pass. Returns a summary dict; writes nothing unless
    `persist`. Channels default to the two that validate on live data."""
    channels = channels or DEFAULT_CHANNELS
    layers = co.build_layers(con, channels=channels, platform=platform, method=method)
    members, summary = co.clusters(layers, resolution=resolution, min_size=min_size)
    log.info(
        "layers: %s; clusters: %d (%d accounts)",
        {ch: len(e) for ch, e in layers.items()},
        len(summary),
        len(members),
    )

    out: dict = {
        "channels": channels,
        "edges": {ch: int(len(e)) for ch, e in layers.items()},
        "n_clusters": int(len(summary)),
        "n_accounts": int(len(members)),
        "persisted": [],
    }
    if not persist:
        return out
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
    p.add_argument("--method", default="fdr", choices=["fdr", "bonferroni", "percentile"])
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
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

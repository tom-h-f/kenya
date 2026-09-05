"""Score the latest persisted coordination run, without recomputing it.

    python -m kma.scorecard_run                 # dry run, prints the top clusters
    python -m kma.scorecard_run --persist       # write to R2

Until this existed the triage layer was computed only when a human opened
`notebooks/coordination.py` and was then discarded: `persist_clusters` stores
membership and channel counts, so `inauthenticity_index`, `near_dup_rate` and
`hate_index` - the signals that separate an engagement pod from an influence
operation - reached no automated consumer.

Why its own entrypoint rather than a step inside `kma.coordination_run`:
scoring the 2026-08-14 run took 2,276s on an idle laptop, dominated by reads of
the whole `posts/` and `authors/` prefixes, against a coordination pass on a
6-hourly timer sharing two contended cores. Reading the run back out of R2
instead of rebuilding it costs ~20s of edge and cluster loads, which buys the
freedom to schedule scoring as slowly as it deserves.

Consequence to keep in mind: the scorecards describe whichever run was newest
when this started, so after a coordination pass they lag by up to one scoring
interval. `stable_cluster_id` is what joins them back together, not
`cluster_id`, which Leiden reissues every pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import duckdb

from kma import coordination as co
from kma import coordination_run as cr
from kma.db import connect, coordination_run_latest

log = logging.getLogger("kma")

DEFAULT_METHOD = "svn_bonf"


def _layers(
    con: duckdb.DuckDBPyConnection,
    channels: list[str],
    platform: str,
    method: str,
) -> dict:
    """Persisted edges per channel, skipping any that were never written.

    A channel that validated no edges has no partition at all, and the glob for
    it raises rather than returning nothing - so a missing layer has to be
    absorbed here or one dead channel takes down the whole scoring pass."""
    out = {}
    for ch in channels:
        try:
            edges = coordination_run_latest(
                con, "edges", platform, channel=ch, method=method
            ).df()
        except duckdb.Error:
            log.info("scorecards: no persisted %s/%s layer, skipping", ch, method)
            continue
        if len(edges):
            out[ch] = edges
    return out


def run(
    con: duckdb.DuckDBPyConnection,
    channels: list[str] | None = None,
    persist: bool = False,
    platform: str = "x",
    method: str = DEFAULT_METHOD,
) -> dict:
    """Score the newest persisted run. Writes nothing unless `persist`."""
    cr.tune(con)
    channels = channels or cr.DEFAULT_CHANNELS

    members = coordination_run_latest(con, "clusters", platform).df()
    out: dict = {
        "n_clusters": 0,
        "n_accounts": 0,
        "channels": [],
        "scorecards_key": None,
    }
    if not len(members):
        log.info("no persisted clusters to score")
        return out
    members = members[["author_id", "cluster_id"]].drop_duplicates()

    layers = _layers(con, channels, platform, method)
    out["channels"] = sorted(layers)

    t0 = time.monotonic()
    cards = co.scorecards(con, members, layers, platform=platform)
    out["n_clusters"] = int(len(cards))
    out["n_accounts"] = int(members["author_id"].nunique())
    log.info(
        "scored %d cluster(s) over %d account(s) in %.0fs",
        len(cards), out["n_accounts"], time.monotonic() - t0,
    )
    if not len(cards):
        return out

    top = cards.nlargest(1, "inauthenticity_index").iloc[0]
    out["top_cluster"] = {
        "cluster_id": int(top["cluster_id"]),
        "inauthenticity_index": float(top["inauthenticity_index"]),
        "size": int(top["size"]),
    }
    if persist:
        out["scorecards_key"] = co.persist_scorecards(
            con, cards, members, platform=platform
        )
        log.info("persisted %s", out["scorecards_key"])
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--channels", default=",".join(cr.DEFAULT_CHANNELS))
    p.add_argument("--persist", action="store_true", help="write results to R2")
    p.add_argument("--platform", default="x")
    p.add_argument("--method", default=DEFAULT_METHOD)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    unknown = set(channels) - set(co.CHANNELS)
    if unknown:
        p.error(f"unknown channel(s): {', '.join(sorted(unknown))}")

    con = connect()
    con.execute("SET enable_progress_bar=false")
    print(json.dumps(run(
        con, channels=channels, persist=args.persist,
        platform=args.platform, method=args.method,
    ), indent=2))


if __name__ == "__main__":
    main()

"""A8: run the v2 detector over the Kenya corpus and report who it surfaces.

    uv run kma-coord2-run --snapshot 2026-09-05-promotion-off --days 14

Always reads a PINNED snapshot rather than the live glob, so a run is
reproducible from its snapshot id alone and cannot silently widen as the
collector writes.

## The threshold problem, stated rather than papered over

On the benchmark the centrality cut is a percentile selected on labelled
validation data. Kenya has no labels, so no such selection is possible, and the
operating point becomes a TRIAGE BUDGET decision: how many accounts is a human
willing to look at. `--top` expresses that honestly. A fixed 1e-2 would be worse
than arbitrary - eigenvector centrality is L2-normalised across nodes, so on a
million-author graph almost nothing clears it.

Nothing here is a verdict. It is a ranked list for a reader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb
import pandas as pd

from kma import bench, coord2
from kma.db import connect

log = logging.getLogger(__name__)

# The bipartite traces. Text similarity is deliberately separate: it needs
# embeddings and is quadratic, so it is opt-in rather than part of the default
# pass.
BIPARTITE_TRACES = {
    "co_retweet": coord2.co_retweet_traces,
    "co_url": coord2.co_url_traces,
    "hashtag_sequence": coord2.hashtag_sequence_traces,
    "fast_retweet": coord2.fast_retweet_traces,
}


@dataclass
class RunResult:
    networks: dict[str, pd.DataFrame]
    scores: pd.DataFrame
    trace_sizes: pd.DataFrame


def build_networks(
    con: duckdb.DuckDBPyConnection, view: str, *, traces: dict | None = None
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Every bipartite trace, from corpus rows to a filtered similarity network."""
    wanted = traces or BIPARTITE_TRACES
    networks: dict[str, pd.DataFrame] = {}
    sizes = []

    for name, extractor in wanted.items():
        percentile = coord2.EDGE_PERCENTILE[name]
        rows = extractor(con, view)
        if rows.empty:
            log.info("%s: no trace rows", name)
            sizes.append({"trace": name, "trace_rows": 0, "users": 0, "edges": 0})
            continue
        edges = coord2.similarity_network(rows, percentile=percentile)
        networks[name] = edges
        users = len(set(edges["source"]) | set(edges["target"])) if not edges.empty else 0
        sizes.append(
            {"trace": name, "trace_rows": len(rows), "users": users, "edges": len(edges)}
        )
        log.info("%s: %d trace rows -> %d edges over %d users", name, len(rows), len(edges), users)

    return networks, pd.DataFrame(sizes)


def run(
    snapshot: str,
    *,
    days: int | None = 14,
    top: int = 500,
    con: duckdb.DuckDBPyConnection | None = None,
) -> RunResult:
    """One v2 pass over a pinned snapshot."""
    con = con or connect()
    manifest = bench.load(snapshot, con=con)
    source = bench.pinned_source(manifest, "posts")

    view = coord2.posts_view(source)
    if days:
        # Bound the window before anything expensive touches it. The corpus is
        # ~2.5M post rows and every trace is a self-join over it.
        view = f"SELECT * FROM ({view}) WHERE created_at >= now() - INTERVAL {int(days)} DAY"

    networks, sizes = build_networks(con, view)
    if not networks:
        raise RuntimeError("no similarity networks were built; nothing to detect on")

    scores = coord2.detect(networks, threshold=0.0)
    scores = scores.sort_values("centrality", ascending=False).head(top).reset_index(drop=True)
    return RunResult(networks=networks, scores=scores, trace_sizes=sizes)


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Run the v2 detector over a pinned snapshot.")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--days", type=int, default=14, help="0 for the whole snapshot")
    ap.add_argument("--top", type=int, default=500, help="triage budget, not a verdict")
    args = ap.parse_args()

    result = run(args.snapshot, days=args.days or None, top=args.top)
    print(result.trace_sizes.to_string(index=False))
    print()
    print(result.scores.head(25).to_string(index=False))


if __name__ == "__main__":
    main()

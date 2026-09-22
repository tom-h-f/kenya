"""Detect campaigns as events: the v2 detector run per time window.

    uv run kma-windowed --snapshot 2026-09-14-deep500 --width day --days 60

WHY
===
The global v2 run scores three months at once, so a campaign that ran for six
hours is averaged into ordinary behaviour and disappears. Kenyan hashtag
campaigns run for hours to days. Here the same traces, the same TF-IDF cosine
projection, the same percentile filters and the same fused graph are built once
per window, and each window is split into communities the way the production
community report splits the global graph (`2026-09-12-component-ranking`).

Two widths, set by the `MIN_ENTITIES` sweep of 2026-09-22
(`investigations/2026-09-22-windowed-floor/findings.md`):

- `DETECTION` - the daily run, which is what catches an event while it is live.
- `SERIES` - the weekly run, the stable series a trend is read from.

WHAT IS NOT WINDOWED
====================
The text-similarity trace. Its persisted edges are user pairs with no
timestamps (`coord2/kind=textsim`), so they cannot be assigned to a window
without re-deriving the post pairs. Windowed runs use the four bipartite traces
only, and say so in `trace_sizes`.

Nothing here is a verdict. Each row is an account in a community in a window,
for a reader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import duckdb
import networkx as nx
import numpy as np
import pandas as pd

from kma import burst, coord2

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowSpec:
    width: str
    min_entities: int


# Set by the 2026-09-22 sweep; the findings carry the numbers behind each.
DETECTION = WindowSpec("day", 5)
SERIES = WindowSpec("week", 10)

WIDTHS = ("day", "week")

# The communities a reader is shown. Three accounts is a retweet ring, not a
# campaign; the production community report counted "four or more" too.
MIN_COMMUNITY = 4

TIMED_TRACES = {
    "co_retweet": coord2.co_retweet_traces,
    "co_url": coord2.co_url_traces,
    "hashtag_sequence": coord2.hashtag_sequence_traces,
    "fast_retweet": coord2.fast_retweet_traces,
}


@dataclass
class WindowedResult:
    per_window: pd.DataFrame
    burst: pd.DataFrame
    trace_sizes: pd.DataFrame


def timed_traces(
    con: duckdb.DuckDBPyConnection, view: str, traces: dict | None = None
) -> dict[str, pd.DataFrame]:
    """Every bipartite trace with the timestamp of the action, extracted once.

    Extracting once and windowing in memory is the design: re-reading a pinned
    snapshot per window costs one R2 scan of thousands of objects each time.
    """
    out = {}
    for name, extractor in (traces or TIMED_TRACES).items():
        rows = extractor(con, view, with_time=True)
        rows["created_at"] = pd.to_datetime(rows["created_at"], utc=True)
        out[name] = rows.dropna(subset=["created_at"]).reset_index(drop=True)
        log.info("%s: %d timed trace rows", name, len(out[name]))
    return out


def window_label(when: pd.Series, width: str) -> pd.Series:
    """ISO date of the window a timestamp falls in: the day, or the Monday that
    starts its week. Labels sort lexically, which is what `burst.stack` needs."""
    if width not in WIDTHS:
        raise ValueError(f"width must be one of {WIDTHS}, got {width!r}")
    day = pd.to_datetime(when, utc=True).dt.floor("D")
    if width == "week":
        day = day - pd.to_timedelta(day.dt.weekday, unit="D")
    return day.dt.strftime("%Y-%m-%d")


def window_networks(
    rows: dict[str, pd.DataFrame], *, min_entities: int
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """One window's traces to similarity networks, exactly as the global run
    builds them: the paper's percentile per trace, the activity floor inside."""
    networks, sizes = {}, []
    for name, trace in rows.items():
        users = 0
        edges = pd.DataFrame({"source": [], "target": [], "weight": []})
        if not trace.empty:
            edges = coord2.similarity_network(
                trace[["user_id", "entity"]],
                percentile=coord2.EDGE_PERCENTILE[name],
                min_entities=min_entities,
            )
            users = len(set(edges["source"]) | set(edges["target"])) if not edges.empty else 0
        if not edges.empty:
            networks[name] = edges
        sizes.append({"trace": name, "trace_rows": len(trace), "users": users, "edges": len(edges)})
    return networks, pd.DataFrame(sizes)


def communities(graph: nx.Graph, *, resolution: float = 1.0, seed: int = 0) -> dict[str, int]:
    """Leiden (modularity) communities, as in the production community report."""
    import igraph as ig
    import leidenalg as la

    names = list(graph.nodes)
    if not names:
        return {}
    index = {n: k for k, n in enumerate(names)}
    g = ig.Graph(n=len(names), edges=[(index[a], index[b]) for a, b in graph.edges()])
    part = la.find_partition(
        g, la.RBConfigurationVertexPartition, resolution_parameter=resolution, seed=seed
    )
    return {names[v]: c for c, members in enumerate(part) for v in members}


def _leading(graph: nx.Graph, nodes: list) -> tuple[float, np.ndarray]:
    sub = graph.subgraph(nodes)
    if sub.number_of_edges() == 0:
        return 0.0, np.zeros(len(nodes))
    order = list(sub.nodes)
    if len(order) <= 500:
        values, vectors = np.linalg.eigh(nx.to_numpy_array(sub, nodelist=order, weight=None))
        value, vec = values[-1], vectors[:, -1]
    else:
        from scipy.sparse.linalg import eigsh

        adj = nx.to_scipy_sparse_array(sub, nodelist=order, weight=None, dtype=float)
        values, vectors = eigsh(adj, k=1, which="LA")
        value, vec = values[0], vectors[:, 0]
    vec = np.abs(vec)
    by_node = dict(zip(order, vec / (np.linalg.norm(vec) or 1.0), strict=True))
    return float(value), np.array([by_node[n] for n in nodes])


def score_window(
    networks: dict[str, pd.DataFrame],
    *,
    resolution: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per account in the window's fused graph.

    `centrality` is the paper's global eigenvector score over the window, kept
    on every row because it is what reproduced the benchmark. `community`,
    `community_size` and `community_eigenvalue` are the report's unit: a
    community ranks by its leading eigenvalue, an account within it by its own
    leading eigenvector (`community_centrality`)."""
    graph = coord2.fuse(networks)
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=["user_id", "centrality", "community", "community_size",
                     "community_eigenvalue", "community_centrality"]
        )
    global_score = coord2.centrality(graph)
    member = communities(graph, resolution=resolution, seed=seed)
    frame = pd.DataFrame({"user_id": list(member), "community": list(member.values())})
    frame["centrality"] = frame["user_id"].map(global_score)
    frame["community_size"] = frame.groupby("community")["user_id"].transform("size")

    eig, within = {}, {}
    for cid, group in frame[frame["community_size"] >= 2].groupby("community"):
        nodes = group["user_id"].tolist()
        eig[cid], vec = _leading(graph, nodes)
        within.update(zip(nodes, vec, strict=True))
    frame["community_eigenvalue"] = frame["community"].map(eig).fillna(0.0)
    frame["community_centrality"] = frame["user_id"].map(within).fillna(0.0)
    return frame.sort_values(
        ["community_eigenvalue", "community_centrality"], ascending=False
    ).reset_index(drop=True)


def run_rows(
    rows: dict[str, pd.DataFrame],
    *,
    width: str,
    min_entities: int,
    resolution: float = 1.0,
    seed: int = 0,
) -> WindowedResult:
    """Windowed detection over already-extracted timed traces."""
    labelled = {
        name: trace.assign(window=window_label(trace["created_at"], width))
        for name, trace in rows.items()
    }
    windows = sorted(set().union(*(set(t["window"]) for t in labelled.values())))
    per_window, sizes = [], []
    for window in windows:
        in_window = {name: t[t["window"] == window] for name, t in labelled.items()}
        networks, window_sizes = window_networks(in_window, min_entities=min_entities)
        sizes.append(window_sizes.assign(window=window))
        if not networks:
            continue
        scores = score_window(networks, resolution=resolution, seed=seed)
        per_window.append(scores.assign(window=window))
        log.info(
            "%s %s: %d accounts, %d communities of %d+",
            width, window, len(scores),
            scores.loc[scores["community_size"] >= MIN_COMMUNITY, "community"].nunique(),
            MIN_COMMUNITY,
        )

    frame = pd.concat(per_window, ignore_index=True) if per_window else pd.DataFrame(
        columns=["user_id", "centrality", "community", "community_size",
                 "community_eigenvalue", "community_centrality", "window"]
    )
    bursts = burst.concentration(frame[["user_id", "window", "centrality"]]) if len(frame) else (
        burst.concentration(pd.DataFrame(columns=["user_id", "window", "centrality"]))
    )
    return WindowedResult(
        per_window=frame,
        burst=bursts,
        trace_sizes=pd.concat(sizes, ignore_index=True) if sizes else pd.DataFrame(),
    )


def snapshot_view(
    con: duckdb.DuckDBPyConnection,
    snapshot: str,
    *,
    days: int | None = None,
    since: str | None = None,
    until: str | None = None,
) -> str:
    """The canonical posts view over a pinned snapshot, optionally cut to a
    period. `days` counts back from the snapshot's newest post, not from now,
    so a rerun of an old snapshot addresses the windows it did when it ran."""
    from kma import bench

    view = coord2.posts_view(bench.pinned_source(bench.load(snapshot, con=con), "posts"))
    if days:
        newest = con.sql(f"SELECT max(created_at) FROM ({view})").fetchone()[0]
        view = (
            f"SELECT * FROM ({view}) WHERE created_at >= "
            f"TIMESTAMPTZ '{newest.isoformat()}' - INTERVAL {int(days)} DAY"
        )
    if since:
        view = f"SELECT * FROM ({view}) WHERE created_at >= TIMESTAMPTZ '{since}'"
    if until:
        view = f"SELECT * FROM ({view}) WHERE created_at < TIMESTAMPTZ '{until}'"
    return view


def run(
    snapshot: str,
    *,
    spec: WindowSpec = DETECTION,
    days: int | None = None,
    since: str | None = None,
    until: str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> WindowedResult:
    """The entry point for a scheduled pipeline: windowed v2 over a pinned
    snapshot, reproducible from the snapshot id and the spec alone."""
    from kma.db import connect

    con = con or connect()
    rows = timed_traces(con, snapshot_view(con, snapshot, days=days, since=since, until=until))
    return run_rows(rows, width=spec.width, min_entities=spec.min_entities)


def persist(
    con: duckdb.DuckDBPyConnection,
    result: WindowedResult,
    *,
    snapshot: str,
    spec: WindowSpec,
    platform: str = "x",
) -> str:
    """Write one windowed pass under its own prefix, `kind=windowed`.

    Separate from `kind=scores` for the reason `coord2_run.persist` keeps v1 and
    v2 apart: a windowed row is (account, window), a scores row is an account,
    and a reader unioning the two would count every account once per window."""
    from kma.db import BUCKET

    now = datetime.now(timezone.utc)
    buf = result.per_window.copy()
    buf["snapshot"] = snapshot
    buf["width"] = spec.width
    buf["min_entities"] = spec.min_entities
    buf["computed_at"] = now
    key = (
        f"coord2/platform={platform}/kind=windowed/width={spec.width}"
        f"/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.parquet"
    )
    con.register("_windowed_buf", buf)
    try:
        con.execute(f"COPY _windowed_buf TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)")
    finally:
        con.unregister("_windowed_buf")
    log.info("wrote %s (%d account-windows)", key, len(buf))
    return key


def curveball(rows: pd.DataFrame, *, burn_in: int = 100, seed: int = 0) -> pd.DataFrame:
    """A degree-preserving randomisation of one trace's user x entity incidence.

    Both degree sequences survive, so anything the detector still reports is
    explained by who is active and what is popular, not by co-action. Burn-in
    counts trades per user; v1's null reported 60 clusters at 5 and 2 at 100,
    so 100 is the floor, not a nicety. Term frequency is dropped: the null is
    over distinct incidences, which is what the curveball swap preserves."""
    from kma.coordination import _curveball_trade

    distinct = rows[["user_id", "entity"]].drop_duplicates()
    if distinct.empty:
        return distinct
    sets = {u: set(g) for u, g in distinct.groupby("user_id")["entity"]}
    rng = np.random.default_rng(seed)
    _curveball_trade(sets, burn_in * len(sets), rng)
    return pd.DataFrame(
        [(u, e) for u, entities in sets.items() for e in entities], columns=["user_id", "entity"]
    )


def main() -> None:
    import argparse

    from kma.db import connect

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Windowed v2 detection over a pinned snapshot.")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--width", choices=WIDTHS, default=DETECTION.width)
    ap.add_argument("--min-entities", type=int, default=None,
                    help="activity floor; defaults to the spec for the width")
    ap.add_argument("--days", type=int, default=0, help="0 for the whole snapshot")
    ap.add_argument("--persist", action="store_true")
    args = ap.parse_args()

    default = DETECTION if args.width == DETECTION.width else SERIES
    spec = WindowSpec(args.width, args.min_entities or default.min_entities)
    result = run(args.snapshot, spec=spec, days=args.days or None)
    shown = result.per_window[result.per_window["community_size"] >= MIN_COMMUNITY]
    print(shown.groupby("window").agg(accounts=("user_id", "size"),
                                      communities=("community", "nunique")).to_string())
    if args.persist:
        print("persisted:", persist(connect(), result, snapshot=args.snapshot, spec=spec))


if __name__ == "__main__":
    main()

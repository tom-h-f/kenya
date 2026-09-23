"""v2's community report: judge the campaign, score the account.

Promoted from `investigations/2026-09-12-component-ranking/03_communities.py`,
which is where it was measured. The global eigenvector score stays the paper's
(the only variant that passed the reproduction gate, 6/6 against 3/6 for
per-component and PageRank), but it hands the whole top 500 to one dense block.
So what is REPORTED changes:

1. Leiden communities over the fused graph (modularity, resolution 1, seed 0 -
   the library defaults, not a tuned value).
2. Within each community of at least `min_size` accounts, the paper's measure
   applied to the group: the leading eigenvector of its induced subgraph.
3. Communities in order of that subgraph's leading eigenvalue, each giving its
   top `per_group` accounts, until `budget` accounts are listed.
4. Communities whose members the relevance model puts below `min_kenya` are
   dropped - a filter, not a re-sort.

Measured on `2026-09-05-promotion-off__emb20260912` at a 0.5 floor: 17
communities, 328 accounts, 12 of 17 read Kenya-relevant in A2 against 3 of 17
for v1. The global score is kept on every row, so the paper's ranking is one
sort away.

## One departure from the investigation script: unscored communities stay

The script filled a missing relevance score with 0, so a community none of
whose posts the model had scored was dropped as off-domain. On a frozen
snapshot scored once, that never happened. On a daily window it is the normal
case for anything NEW - the relevance app scores on its own schedule, and
`relevance/` held no partition after `dt=2026-09-14` when this was written
(listed 2026-09-22) - so the rule would drop precisely the communities a daily
run exists to catch. Here a community is dropped only when it HAS a score and
the score is below the floor; one without a score is kept and flagged
`relevance_scored=False`, for the adjudicator to judge on its text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb
import networkx as nx
import numpy as np
import pandas as pd

from kma import relevance
from kma.db import relevance_source

log = logging.getLogger(__name__)

# The promoted model, not the one the listing was measured with: the 0.5 floor
# was set on `kenya-relevance-afroxlmr-2026-09-12`, and `relevance.MODEL` has
# since moved to -09-13, which is the only one still being scored.
RELEVANCE_MODEL = relevance.MODEL
RELEVANCE_THRESHOLD = relevance.THRESHOLD
MIN_KENYA = 0.5
BUDGET = 500
PER_GROUP = 25
MIN_SIZE = 4
RESOLUTION = 1.0
SEED = 0


@dataclass
class Report:
    members: pd.DataFrame
    """Every account in every community of `min_size`+, with its scores."""
    communities: pd.DataFrame
    """One row per community of `min_size`+, kept or dropped, and why."""
    listed: pd.DataFrame
    """The accounts the triage budget buys, in listing order."""


def communities(graph: nx.Graph, resolution: float = RESOLUTION, seed: int = SEED) -> dict[str, int]:
    """Leiden (modularity) membership, account -> community id.

    Ids are arbitrary and NOT stable across runs; match communities between
    runs by membership overlap, never by id.
    """
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


def leading(graph: nx.Graph, nodes: list) -> tuple[float, np.ndarray]:
    """Leading eigenvalue and |eigenvector| of the unweighted adjacency."""
    if len(nodes) <= 500:
        adj = nx.to_numpy_array(graph, nodelist=nodes, weight=None)
        values, vectors = np.linalg.eigh(adj)
        return float(values[-1]), np.abs(vectors[:, -1])
    from scipy.sparse.linalg import eigsh

    adj = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight=None, dtype=float)
    values, vectors = eigsh(adj, k=1, which="LA")
    return float(values[0]), np.abs(vectors[:, 0])


def model_kenya_share(
    con: duckdb.DuckDBPyConnection,
    view: str,
    ids: list[str],
    model: str = RELEVANCE_MODEL,
    threshold: float = RELEVANCE_THRESHOLD,
    source: str | None = None,
) -> pd.DataFrame:
    """Per account: share of its SCORED posts the model calls Kenyan, and how
    many of its posts were scored at all.

    Accounts with no scored post are absent from `kenya_model` (NaN), which is
    the distinction the module docstring's departure depends on.
    """
    source = source or relevance_source("x", model)
    con.register("_rel_ids", pd.DataFrame({"user_id": ids}))
    try:
        got = con.sql(
            f"""
            WITH scored AS (
                SELECT platform_post_id, p_kenya FROM {source}
                QUALIFY row_number() OVER (
                    PARTITION BY platform_post_id ORDER BY scored_at DESC) = 1
            )
            SELECT CAST(p.user_id AS VARCHAR) AS user_id,
                   count(*) AS posts,
                   count(s.p_kenya) AS scored,
                   avg(CASE WHEN s.p_kenya IS NULL THEN NULL
                            WHEN s.p_kenya >= {float(threshold)} THEN 1.0 ELSE 0.0 END) AS kenya_model
            FROM ({view}) p
            JOIN _rel_ids i ON CAST(p.user_id AS VARCHAR) = i.user_id
            LEFT JOIN scored s ON s.platform_post_id = p.post_id
            GROUP BY 1
            """
        ).df()
    finally:
        con.unregister("_rel_ids")
    return got.set_index("user_id")


def report(
    graph: nx.Graph,
    global_scores: pd.Series,
    *,
    kenya: pd.DataFrame | None = None,
    budget: int = BUDGET,
    per_group: int = PER_GROUP,
    min_size: int = MIN_SIZE,
    min_kenya: float = MIN_KENYA,
    resolution: float = RESOLUTION,
    seed: int = SEED,
) -> Report:
    """Communities, within-community scores, relevance filter, listing.

    `kenya` is `model_kenya_share`'s frame, indexed by user_id; None skips the
    relevance filter entirely (every community kept, flagged unscored).
    """
    member = communities(graph, resolution, seed)
    groups = pd.Series(member, name="community").rename_axis("user_id").reset_index()
    sizes = groups["community"].value_counts()

    rows, table = [], []
    for community, size in sizes[sizes >= min_size].items():
        nodes = groups.loc[groups["community"] == community, "user_id"].tolist()
        sub = graph.subgraph(nodes)
        value, vector = leading(sub, nodes)
        vector = vector / np.linalg.norm(vector)
        # `coord2.fuse` labels every edge it builds, but a caller can hand this
        # a graph assembled another way - the canary's synthetic cliques do -
        # and an unlabelled edge should cost that community its trace shares,
        # not the whole report.
        traces = pd.Series(
            [t for _, _, d in sub.edges(data=True) for t in d.get("traces", ())],
            dtype="object",
        ).value_counts()
        total = max(int(traces.sum()), 1)
        table.append({
            "community": int(community),
            "size": int(size),
            "edges": sub.number_of_edges(),
            "eigenvalue": float(value),
            **{f"share_{t}": float(traces.get(t, 0) / total)
               for t in sorted(set(traces.index))},
        })
        rows += [{
            "user_id": str(n),
            "community": int(community),
            "community_size": int(size),
            "community_eigenvalue": float(value),
            "within_score": float(v),
            "centrality": float(global_scores.get(n, 0.0)),
        } for n, v in zip(nodes, vector, strict=True)]

    members = pd.DataFrame(rows, columns=[
        "user_id", "community", "community_size", "community_eigenvalue",
        "within_score", "centrality",
    ])
    table = pd.DataFrame(table)
    if table.empty:
        empty = members.assign(rank_in_group=pd.Series(dtype="int64"))
        return Report(members=empty, communities=table, listed=empty)
    table = table.sort_values("eigenvalue", ascending=False).reset_index(drop=True)
    members = members.sort_values(
        ["community_eigenvalue", "within_score"], ascending=[False, False]
    ).reset_index(drop=True)

    if kenya is not None and not kenya.empty:
        members["kenya_model"] = members["user_id"].map(kenya["kenya_model"])
        members["posts_scored"] = members["user_id"].map(kenya["scored"]).fillna(0).astype(int)
        members["posts_in_window"] = members["user_id"].map(kenya["posts"]).fillna(0).astype(int)
    else:
        members["kenya_model"] = np.nan
        members["posts_scored"] = 0
        members["posts_in_window"] = 0

    per = members.groupby("community").agg(
        kenya_model=("kenya_model", "mean"),
        posts_scored=("posts_scored", "sum"),
        posts_in_window=("posts_in_window", "sum"),
    )
    table["kenya_model"] = table["community"].map(per["kenya_model"])
    table["relevance_coverage"] = table["community"].map(
        per["posts_scored"] / per["posts_in_window"].where(per["posts_in_window"] > 0)
    )
    table["relevance_scored"] = table["kenya_model"].notna()
    table["kept"] = ~(table["relevance_scored"] & (table["kenya_model"] < min_kenya))

    kept = set(table.loc[table["kept"], "community"])
    candidates = members[members["community"].isin(kept)].copy()
    candidates["rank_in_group"] = candidates.groupby("community").cumcount()
    listed = candidates[candidates["rank_in_group"] < per_group].head(budget).reset_index(drop=True)
    table["listed"] = table["community"].map(listed["community"].value_counts()).fillna(0).astype(int)

    log.info(
        "communities: %d accounts, %d communities of %d+; %d kept (%d unscored), %d listed "
        "accounts from %d communities",
        graph.number_of_nodes(), len(table), min_size, len(kept),
        int((~table["relevance_scored"]).sum()), len(listed), listed["community"].nunique(),
    )
    return Report(members=members, communities=table, listed=listed)

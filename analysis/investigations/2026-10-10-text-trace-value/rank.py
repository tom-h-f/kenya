"""Rank a window from cached traces, with or without the text trace.

The same calls `v2_daily.rank` makes after the networks exist: `coord2.fuse`,
`coord2.centrality`, `coord2_communities.report`. Nothing here re-implements
them; the trace dicts are simply built from cached parquet.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from kma import coord2, coord2_communities

from common import BIPARTITE, wdir


@dataclass
class Arm:
    name: str
    nets: dict[str, pd.DataFrame]
    graph: nx.Graph
    glob: pd.Series
    report: coord2_communities.Report

    @property
    def listed(self) -> pd.DataFrame:
        return self.report.listed

    def text_edge_set(self) -> set[tuple[str, str]]:
        t = self.nets.get("text_similarity")
        if t is None or t.empty:
            return set()
        return {tuple(sorted(p)) for p in zip(t["source"].astype(str), t["target"].astype(str))}


def floor_text(edges: pd.DataFrame, counts: pd.DataFrame,
               minimum: int = coord2.MIN_ENTITIES_PER_USER) -> pd.DataFrame:
    """`coord2_run.apply_text_floor` over the cached per-user counts."""
    eligible = set(counts.loc[counts["n"] >= minimum, "user_id"].astype(str))
    edges = edges.assign(source=edges["source"].astype(str), target=edges["target"].astype(str))
    kept = edges[edges["source"].isin(eligible) & edges["target"].isin(eligible)]
    return kept.reset_index(drop=True)


def nets_for(window: str, text: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    nets = {}
    for name in BIPARTITE:
        path = wdir(window) / "nets" / f"{name}.parquet"
        if path.exists():
            nets[name] = pd.read_parquet(path).astype({"source": str, "target": str})
    if text is not None:
        counts = pd.read_parquet(wdir(window) / "floor_counts.parquet")
        nets["text_similarity"] = floor_text(text, counts)
    return nets


def rank(name: str, nets: dict[str, pd.DataFrame], kenya: pd.DataFrame | None,
         seed: int = coord2_communities.SEED) -> Arm:
    graph = coord2.fuse(nets)
    glob = coord2.centrality(graph)
    rep = coord2_communities.report(graph, glob, kenya=kenya, seed=seed)
    return Arm(name, nets, graph, glob, rep)


def best_match(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """For each community of `a`, the community of `b` sharing most members.

    Ids are not stable across runs, so this is membership overlap, as
    `v2_daily` documents. Frames are `Report.members`."""
    b_sets = b.groupby("community")["user_id"].agg(set)
    owner: dict[str, list[int]] = {}
    for c, s in b_sets.items():
        for u in s:
            owner.setdefault(u, []).append(c)
    rows = []
    for c, members in a.groupby("community")["user_id"].agg(set).items():
        hits: dict[int, int] = {}
        for u in members:
            for bc in owner.get(u, ()):
                hits[bc] = hits.get(bc, 0) + 1
        if hits:
            bc = max(hits, key=lambda k: hits[k] / len(members | b_sets[k]))
            inter = hits[bc]
            jac = inter / len(members | b_sets[bc])
        else:
            bc, inter, jac = -1, 0, 0.0
        rows.append({"community": c, "size": len(members), "match": bc, "shared": inter, "jaccard": jac})
    return pd.DataFrame(rows)


def non_text_component(arm: Arm, members: list[str]) -> int:
    """Largest connected component of a community once its text-only edges are
    removed (an edge that is also behavioural stays)."""
    sub = arm.graph.subgraph(members)
    keep = nx.Graph()
    keep.add_nodes_from(members)
    keep.add_edges_from((u, v) for u, v, d in sub.edges(data=True) if d["traces"] - {"text_similarity"})
    return max((len(c) for c in nx.connected_components(keep)), default=0)


def ensure_kenya(con, window: str, ids: list[str]) -> pd.DataFrame:
    """`coord2_communities.model_kenya_share` over the window's pinned posts,
    cached per user so each arm and each coverage state reuses one query."""
    from kma import bench
    from kma.coord2 import posts_view

    path = wdir(window) / "kenya.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(
        columns=["user_id", "posts", "scored", "kenya_model"])
    have["user_id"] = have["user_id"].astype(str)
    missing = sorted(set(map(str, ids)) - set(have["user_id"]))
    if missing:
        manifest = pd.read_parquet(wdir(window) / "manifest.parquet")
        view = posts_view(bench.pinned_source(manifest, "posts"))
        got = coord2_communities.model_kenya_share(con, view, missing).reset_index()
        have = pd.concat([have, got], ignore_index=True)
        have.to_parquet(path)
    return have.drop_duplicates("user_id").set_index("user_id")

"""Why every text-trace variant hands the whole top 500 to one block.

    cd analysis && uv run python investigations/2026-09-12-component-ranking/01_structure.py \\
        --text current=<textsim out>/sweep_min0.75/edges_t0.85.parquet \\
        --text floored=<textsim out>/sweep_lex0.10/edges_t0.85.parquet \\
        --campaign <textsim out>/pair_audit_judged_t0.85.parquet --rows <textsim out>/rows.parquet

The textsim investigation found that the top 500 is always one dense block -
the Sheng-reply mass, one co-retweet block, or a greeting farm - whichever trace
and cut is used. `coord2.centrality` already warns why that can happen: on a
disconnected graph the leading eigenvector sits on one component. This checks
whether that is the mechanism here, or whether the concentration happens inside
one connected component, and compares rankings that cannot concentrate that way:

  global      the current ranking: the fused graph's leading eigenvector
  component   each connected component's own leading eigenvector, scaled by
              that component's leading eigenvalue, so a dense group ranks by how
              dense it is rather than scoring zero for not being the densest
  pagerank    PageRank (alpha 0.85, unweighted), which gives every component
              mass in proportion to its size

The four bipartite traces are built once from the pinned snapshot and cached,
so each text variant costs a fuse and three rankings.
"""

import argparse
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh

from kma import bench, coord2, coord2_run
from kma.db import connect

HERE = Path(__file__).parent
TOP = 500


def cached_networks(con, view, cache: Path) -> dict[str, pd.DataFrame]:
    cache.mkdir(parents=True, exist_ok=True)
    files = sorted(cache.glob("*.parquet"))
    if files:
        return {f.stem: pd.read_parquet(f) for f in files}
    start = time.perf_counter()
    networks, _ = coord2_run.build_networks(con, view)
    for name, edges in networks.items():
        edges = edges.assign(source=edges["source"].astype(str), target=edges["target"].astype(str))
        edges.to_parquet(cache / f"{name}.parquet")
        networks[name] = edges
    print(f"built bipartite traces in {time.perf_counter() - start:.0f}s")
    return networks


def leading(graph: nx.Graph, nodes: list) -> tuple[float, np.ndarray]:
    if len(nodes) <= 500:
        adj = nx.to_numpy_array(graph, nodelist=nodes, weight=None)
        values, vectors = np.linalg.eigh(adj)
        return float(values[-1]), np.abs(vectors[:, -1])
    adj = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight=None, dtype=float)
    values, vectors = eigsh(adj, k=1, which="LA")
    return float(values[0]), np.abs(vectors[:, 0])


def component_centrality(graph: nx.Graph) -> tuple[pd.Series, pd.DataFrame]:
    scores, rows = {}, []
    for k, members in enumerate(sorted(nx.connected_components(graph), key=len, reverse=True)):
        nodes = list(members)
        sub = graph.subgraph(nodes)
        value, vector = leading(sub, nodes)
        vector = vector / np.linalg.norm(vector)
        scores.update(zip(nodes, value * vector, strict=True))
        # Benchmark graphs arrive fused already, without per-edge trace labels.
        traces = pd.Series([t for _, _, d in sub.edges(data=True) for t in d.get("traces", ())]).value_counts()
        rows.append({"component": k, "nodes": len(nodes), "edges": sub.number_of_edges(),
                     "eigenvalue": round(value, 2), **{f"edges:{t}": int(n) for t, n in traces.items()}})
    return pd.Series(scores, name="score"), pd.DataFrame(rows)


def rankings(graph: nx.Graph) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    glob = coord2.centrality(graph)
    comp, table = component_centrality(graph)
    pr = pd.Series(nx.pagerank(graph, alpha=0.85, weight=None), name="score")
    return {"global": glob, "component": comp, "pagerank": pr}, table


def localisation(vector: pd.Series) -> dict:
    v = np.sort(vector.to_numpy() ** 2)[::-1]
    return {"IPR": round(float((v**2).sum()), 5), "nodes for 90% of mass": int(np.searchsorted(np.cumsum(v), 0.9) + 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default="2026-09-05-promotion-off")
    ap.add_argument("--text", action="append", required=True, help="name=path to a text edges parquet")
    ap.add_argument("--campaign", type=Path, help="reader-labelled pairs; same_message authors are tracked")
    ap.add_argument("--rows", type=Path, help="rows.parquet that maps the labelled post ids to authors")
    args = ap.parse_args()

    con = connect()
    view = coord2.posts_view(bench.pinned_source(bench.load(args.snapshot, con=con), "posts"))
    networks = cached_networks(con, view, HERE / "out" / args.snapshot / "networks")

    campaign = set()
    if args.campaign and args.rows:
        labelled = pd.read_parquet(args.campaign)
        rows = pd.read_parquet(args.rows).astype({"post_id": str, "user_id": str})
        posts = set(labelled.loc[labelled["label"] == "same_message", ["a", "b"]].to_numpy().ravel())
        campaign = set(rows.loc[rows["post_id"].isin(posts), "user_id"])

    out = HERE / "out" / args.snapshot
    for spec in args.text:
        name, path = spec.split("=", 1)
        text = coord2_run.apply_text_floor(
            con, pd.read_parquet(path).astype({"source": str, "target": str}), view
        )
        graph = coord2.fuse({**networks, "text_similarity": text})
        start = time.perf_counter()
        ranked, table = rankings(graph)
        sizes = table["nodes"].to_numpy()
        print(f"\n=== {name}: {graph.number_of_nodes():,} accounts, {graph.number_of_edges():,} edges, "
              f"{len(sizes):,} components; largest {sizes[:6].tolist()} "
              f"({time.perf_counter() - start:.0f}s for all three rankings)")
        print("largest components by trace:")
        print(table.head(6).fillna(0).to_string(index=False))
        print(f"global eigenvector localisation: {localisation(ranked['global'])}")

        member = {n: k for k, members in enumerate(sorted(nx.connected_components(graph), key=len, reverse=True))
                  for n in members}
        report = []
        for method, scores in ranked.items():
            top = scores.sort_values(ascending=False).head(TOP)
            ids = top.index.astype(str)
            frame = coord2_run.attach_relevance(con, pd.DataFrame({"user_id": ids, "centrality": top.to_numpy()}), view)
            frame.to_parquet(out / f"top{TOP}_{name}_{method}.parquet")
            comps = pd.Series([member[u] for u in top.index]).value_counts()
            text_linked = sum(1 for u in top.index
                              if any("text_similarity" in d["traces"] for _, _, d in graph.edges(u, data=True)))
            report.append({
                "ranking": method,
                "components in top 500": len(comps),
                "from largest component": int(comps.get(0, 0)),
                "text-linked": text_linked,
                "campaign authors": len(campaign & set(ids)),
                "kenya mean": round(frame["kenya_share"].mean(), 3),
                "kenya median": round(frame["kenya_share"].median(), 3),
                "median posts": int(frame["n_posts"].median()),
            })
        print(pd.DataFrame(report).to_string(index=False))
        if campaign:
            print(f"(campaign authors: {len(campaign)}, of them in the graph: {len(campaign & set(graph.nodes))})")


if __name__ == "__main__":
    main()

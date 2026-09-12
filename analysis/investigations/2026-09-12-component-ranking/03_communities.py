"""Split the fused graph into communities and spend the triage budget across them.

    cd analysis && uv run python investigations/2026-09-12-component-ranking/03_communities.py \\
        --text <edges parquet> --campaign <labelled pairs> --rows <rows.parquet>

`01_structure.py` found the concentration is not across components: the fused
Kenya graph is one giant component (17,551 of 20,480 accounts with the floored
text trace) holding every trace, and the leading eigenvector puts 90% of its
mass on 433 of them - the co-retweet core. `02_gate.py` found that replacing
the global score with a per-component one or PageRank fails the reproduction
gate (3/6 each, against 6/6). So the score stays the paper's and this changes
only what is REPORTED:

1. Leiden communities over the whole fused graph (modularity, resolution 1,
   seed 0 - the library's defaults, not a tuned value).
2. Within each community of at least `--min-size` accounts, the paper's own
   measure applied to the group: the leading eigenvector of its induced
   subgraph.
3. Communities in order of that subgraph's leading eigenvalue - the quantity
   eigenvector centrality rewards - each contributing its top `--per-group`
   accounts, until `--budget` accounts are listed.

The unit a reader judges becomes the community, which is the A3 decision
("judge the campaign, score the account"); the global score is kept on every
row, so the paper's ranking is still one sort away.
"""

import argparse
import importlib.util
from pathlib import Path

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import pandas as pd

from kma import bench, coord2, coord2_run
from kma.db import connect, relevance_source

HERE = Path(__file__).parent
RELEVANCE_MODEL = "kenya-relevance-afroxlmr-2026-09-12"
spec = importlib.util.spec_from_file_location("structure", HERE / "01_structure.py")
structure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(structure)


def model_kenya_share(con, view: str, ids: list[str], model: str, threshold: float) -> pd.Series:
    """Share of each account's posts the relevance classifier calls Kenyan.

    The same shape as `coord2_run.attach_relevance`, from the learned scores
    instead of the keyword gate: A2 found a third of the listed communities
    were K-pop, anime and other off-domain hashtag drives, and the gate's
    recall (0.655) is too leaky to separate them.
    """
    con.register("_rel_ids", pd.DataFrame({"user_id": ids}))
    try:
        got = con.sql(
            f"""
            WITH scored AS (
                SELECT platform_post_id, p_kenya FROM {relevance_source('x', model)}
                QUALIFY row_number() OVER (
                    PARTITION BY platform_post_id ORDER BY scored_at DESC) = 1
            )
            SELECT CAST(p.user_id AS VARCHAR) AS user_id,
                   avg(CASE WHEN s.p_kenya >= {float(threshold)} THEN 1.0 ELSE 0.0 END) AS kenya_model
            FROM ({view}) p
            JOIN _rel_ids i ON CAST(p.user_id AS VARCHAR) = i.user_id
            JOIN scored s ON s.platform_post_id = p.post_id
            GROUP BY 1
            """
        ).df()
    finally:
        con.unregister("_rel_ids")
    return got.set_index("user_id")["kenya_model"]


def communities(graph: nx.Graph, resolution: float, seed: int) -> dict[str, int]:
    names = list(graph.nodes)
    index = {n: k for k, n in enumerate(names)}
    g = ig.Graph(n=len(names), edges=[(index[a], index[b]) for a, b in graph.edges()])
    part = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter=resolution, seed=seed)
    return {names[v]: c for c, members in enumerate(part) for v in members}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default="2026-09-05-promotion-off")
    ap.add_argument("--text", type=Path, required=True, help="text-similarity edges parquet")
    ap.add_argument("--name", default="production")
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--per-group", type=int, default=25)
    ap.add_argument("--min-size", type=int, default=4)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--campaign", type=Path)
    ap.add_argument("--rows", type=Path)
    ap.add_argument("--relevance-model", default=RELEVANCE_MODEL, help="'' to rank without it")
    ap.add_argument("--relevance-threshold", type=float, default=0.5)
    ap.add_argument("--min-kenya", type=float, default=0.2,
                    help="drop communities whose members' model Kenya share is below this")
    args = ap.parse_args()

    con = connect()
    view = coord2.posts_view(bench.pinned_source(bench.load(args.snapshot, con=con), "posts"))
    networks = structure.cached_networks(con, view, HERE / "out" / args.snapshot / "networks")
    text = coord2_run.apply_text_floor(con, pd.read_parquet(args.text).astype({"source": str, "target": str}), view)
    graph = coord2.fuse({**networks, "text_similarity": text})
    glob = coord2.centrality(graph)

    member = communities(graph, args.resolution, seed=0)
    groups = pd.Series(member).rename("community").rename_axis("user_id").reset_index()
    sizes = groups["community"].value_counts()
    print(f"{graph.number_of_nodes():,} accounts, {graph.number_of_edges():,} edges -> {len(sizes):,} communities; "
          f"{int((sizes >= args.min_size).sum())} of {args.min_size}+ accounts; largest {sizes.head(8).tolist()}")

    rows, table = [], []
    for community, size in sizes[sizes >= args.min_size].items():
        nodes = groups.loc[groups["community"] == community, "user_id"].tolist()
        sub = graph.subgraph(nodes)
        value, vector = structure.leading(sub, nodes)
        vector = vector / np.linalg.norm(vector)
        traces = pd.Series([t for _, _, d in sub.edges(data=True) for t in d["traces"]]).value_counts()
        table.append({"community": community, "size": size, "edges": sub.number_of_edges(),
                      "eigenvalue": round(value, 2),
                      "text share": round(float(traces.get("text_similarity", 0) / max(traces.sum(), 1)), 3),
                      "co_retweet share": round(float(traces.get("co_retweet", 0) / max(traces.sum(), 1)), 3)})
        rows += [{"user_id": n, "community": community, "community_size": size, "community_eigenvalue": value,
                  "within_score": float(v), "global_score": float(glob[n])} for n, v in zip(nodes, vector)]

    table = pd.DataFrame(table).sort_values("eigenvalue", ascending=False).reset_index(drop=True)
    scored = pd.DataFrame(rows).sort_values(["community_eigenvalue", "within_score"], ascending=[False, False])

    if args.relevance_model:
        share = model_kenya_share(con, view, scored["user_id"].astype(str).unique().tolist(),
                                  args.relevance_model, args.relevance_threshold)
        scored["kenya_model"] = scored["user_id"].astype(str).map(share)
        per_community = scored.groupby("community")["kenya_model"].mean()
        table["kenya_model"] = table["community"].map(per_community).round(3)
        # A filter, not a re-sort: the eigenvalue order is the coherence the
        # detector fired on, and relevance decides whether it is our business.
        dropped = table[table["kenya_model"].fillna(0) < args.min_kenya]
        print(f"\ndropping {len(dropped)} of {len(table)} communities below Kenya share {args.min_kenya}"
              f" (largest: {dropped.nlargest(5, 'size')[['community', 'size', 'kenya_model']].to_dict('records')})")
        scored = scored[~scored["community"].isin(set(dropped["community"]))]
        table = table[~table["community"].isin(set(dropped["community"]))]
    scored["rank_in_group"] = scored.groupby("community").cumcount()
    listed = scored[scored["rank_in_group"] < args.per_group].head(args.budget)
    scored["listed"] = scored.index.isin(listed.index)

    report = coord2_run.attach_relevance(
        con, listed[["user_id", "global_score"]].rename(columns={"global_score": "centrality"}), view
    ).merge(listed[["user_id", "community"]], on="user_id")
    kenya = report.groupby("community")["kenya_share"].mean()
    table["listed"] = table["community"].map(listed["community"].value_counts()).fillna(0).astype(int)
    table["kenya share (listed)"] = table["community"].map(kenya).round(3)

    campaign = set()
    if args.campaign and args.rows:
        labelled = pd.read_parquet(args.campaign)
        ids = pd.read_parquet(args.rows).astype({"post_id": str, "user_id": str})
        posts = set(labelled.loc[labelled["label"] == "same_message", ["a", "b"]].to_numpy().ravel())
        campaign = set(ids.loc[ids["post_id"].isin(posts), "user_id"])

    top_global = set(glob.sort_values(ascending=False).head(args.budget).index)
    listed_ids = set(listed["user_id"])
    text_linked = {u for u in listed_ids
                   if any("text_similarity" in d["traces"] for _, _, d in graph.edges(u, data=True))}
    print(f"\nlisted {len(listed_ids)} accounts from {listed['community'].nunique()} communities "
          f"(global top {args.budget} shares {len(listed_ids & top_global)} of them)")
    print(f"text-linked {len(text_linked)}; campaign authors {len(campaign & listed_ids)} of {len(campaign)}; "
          f"Kenya share mean {report['kenya_share'].mean():.3f} median {report['kenya_share'].median():.3f}")
    print("\ncommunities in listing order:")
    print(table[table["listed"] > 0].head(30).to_string(index=False))

    out = HERE / "out" / args.snapshot
    scored.to_parquet(out / f"communities_{args.name}.parquet")
    table.to_parquet(out / f"community_table_{args.name}.parquet")
    print(f"\n-> {out}/communities_{args.name}.parquet")


if __name__ == "__main__":
    main()

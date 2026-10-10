"""Q4: what the text trace finds that the behavioural traces do not.

    cd analysis && uv run python investigations/2026-10-10-text-trace-value/07_unique.py \
        --tag manifest|current

"Text-only" account: has an edge in the thresholded text network and in none of
the four thresholded behavioural networks (what the detector actually sees, so
an account with one stray shared retweet that missed the 96th-percentile edge
cut counts as text-only). Reads 03_arms.py's `full` arm and 05_precision.py's
pair scores; writes `q4_sample.parquet`, the pairs the blind read draws from
(text-only edges inside listed communities, seeded).

Read-only; ids stay in out/.
"""

import argparse
import json

import networkx as nx
import numpy as np
import pandas as pd

from common import BIPARTITE, WINDOW, wdir
import rank as rk

N_READ = 20


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", default=WINDOW)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    base = wdir(args.window)
    arms = base / f"arms_{args.tag}"
    pdir = base / f"pairs_{args.tag}"
    listed = pd.read_parquet(arms / "listed_full.parquet")
    members = pd.read_parquet(arms / "members_full.parquet")
    comms = pd.read_parquet(arms / "communities_full.parquet")
    pairs = pd.read_parquet(pdir / "pairs_scored.parquet")
    kenya = pd.read_parquet(base / "kenya.parquet").set_index("user_id")
    counts = pd.read_parquet(base / "floor_counts.parquet").set_index("user_id")["n"]

    nets = rk.nets_for(args.window, pd.read_parquet(pdir / "edges.parquet"))
    behavioural = set()
    for name in BIPARTITE:
        d = nets.get(name)
        if d is not None and len(d):
            behavioural |= set(d["source"]) | set(d["target"])
    text = nets["text_similarity"]
    text_users = set(text["source"]) | set(text["target"])
    text_only = text_users - behavioural
    listed_ids = set(listed["user_id"])

    # pure-text communities: nothing left once text-only edges go
    full = rk.rank("full", nets, None)
    comm_rows = []
    for c in sorted(listed["community"].unique()):
        nodes = members.loc[members["community"] == c, "user_id"].tolist()
        comp = rk.non_text_component(full, nodes)
        comm_rows.append({"community": int(c), "size": len(nodes), "largest_non_text": comp,
                          "pure_text": comp < 4,
                          "listed": int((listed["community"] == c).sum())})
    ct = pd.DataFrame(comm_rows)
    pure = ct[ct["pure_text"]]
    pure_ids = set(listed.loc[listed["community"].isin(pure["community"]), "user_id"])

    def kenya_block(ids: set[str]) -> dict:
        k = kenya.reindex(sorted(ids))
        scored = k["scored"].fillna(0) > 0
        return {
            "accounts": len(ids),
            "with_scored_posts": int(scored.sum()),
            "mean_kenya_model": round(float(k.loc[scored, "kenya_model"].mean()), 3),
            "share_ge_0.5": round(float((k.loc[scored, "kenya_model"] >= 0.5).mean()), 3),
            "median_posts_in_window": float(k["posts"].median()),
            "median_text_posts": float(counts.reindex(sorted(ids)).median()),
        }

    g = nx.Graph()
    g.add_edges_from(zip(text["source"], text["target"]))
    sub = g.subgraph(text_only)
    comps = sorted((len(c) for c in nx.connected_components(sub)), reverse=True)

    res = {
        "text_edge_accounts": len(text_users),
        "text_only_accounts": len(text_only),
        "text_only_share": round(len(text_only) / max(len(text_users), 1), 4),
        "text_only_components_top10": comps[:10],
        "text_only_components_ge4": int(sum(c >= 4 for c in comps)),
        "listed": len(listed_ids),
        "listed_text_only": len(listed_ids & text_only),
        "listed_in_pure_text_communities": len(pure_ids),
        "pure_text_communities": int(len(pure)),
        "pure_text_community_sizes": pure["size"].tolist(),
        "listed_communities": int(len(ct)),
        "kenya_text_only": kenya_block(text_only),
        "kenya_text_only_listed": kenya_block(listed_ids & text_only),
        "kenya_text_with_behavioural": kenya_block(text_users & behavioural),
        "kenya_listed_all": kenya_block(listed_ids),
        "kenya_pure_text_communities": kenya_block(pure_ids),
    }

    # read sample: edges between two text-only accounts, inside pure-text communities if any
    pool_ids = pure_ids or (listed_ids & text_only)
    e = pairs[pairs["source"].isin(pool_ids) & pairs["target"].isin(pool_ids)]
    one = e.sample(frac=1.0, random_state=args.seed).drop_duplicates(["source", "target"])
    rows = pd.read_parquet(base / "rows.parquet", columns=["post_id", "text"]).set_index("post_id")["text"]
    pick = one.sample(n=min(N_READ, len(one)), random_state=args.seed)
    pick = pick.assign(text_a=pick["post_a"].map(rows), text_b=pick["post_b"].map(rows))
    pick.to_parquet(pdir / "q4_sample.parquet")
    res["q4_pool_edges"] = int(len(one))
    res["q4_pool_pairs_token_jaccard_median"] = round(float(e["tok_j"].median()), 3) if len(e) else None
    res["q4_pool_pairs_identical_share"] = round(float(e["identical"].mean()), 3) if len(e) else None
    res["q4_pool_unique_clean_texts"] = int(pd.concat([e["post_a"], e["post_b"]]).map(
        pd.read_parquet(base / "rows.parquet", columns=["post_id", "clean"]).set_index("post_id")["clean"]
    ).nunique()) if len(e) else 0
    ct.to_csv(arms / "listed_communities_text_dependence.csv", index=False)
    (arms / "unique.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(ct.to_string(index=False))


if __name__ == "__main__":
    main()

"""Q1, Q4, Q6: rank the same window with and without the text trace.

    cd analysis && uv run python investigations/2026-10-10-text-trace-value/03_arms.py \
        --tag manifest|current [--window daily-2026-10-09-30d]

Needs 01_export.py (the four bipartite networks, per-user floor counts) and
02_pairs.py with the same tag (the text trace at the coverage under test).
Arms, all over the SAME four bipartite networks:

  none     no text trace
  full     the text trace as production builds it (floor, mention stripping,
           0.85 cosine, 0.10 char-4gram overlap)
  a<N>     text edges whose later post falls in the last N days (the trace's
           own `bucket=day` axis), after the whole-window pair search
  b<N>     text pairs whose BOTH posts were created in the last N days
  c<N>     text pairs whose BOTH posts were collected in the last N days: what a
           real N-day text window would build, since the window pins by `dt`

Each arm runs `coord2.fuse` -> `centrality` -> `coord2_communities.report`
with the production defaults (budget 500, 25 per community, 4+ members, Kenya
floor 0.5 on the relevance model). Output under out/<window>/arms_<tag>/.

Read-only on R2 (the relevance query); ids stay in out/.
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd

from kma.db import connect

from common import WINDOW, wdir
import rank as rk

log = logging.getLogger("arms")
DAYS = (1, 3, 7, 14)


def text_users(edges: pd.DataFrame) -> set[str]:
    return set(edges["source"].astype(str)) | set(edges["target"].astype(str))


def to_edges(pairs: pd.DataFrame) -> pd.DataFrame:
    """Post pairs to the user-level trace, as `text_similarity_network` does
    without a time bucket: mean cosine per account pair."""
    if pairs.empty:
        return pd.DataFrame({"source": [], "target": [], "weight": []})
    return pairs.groupby(["source", "target"], as_index=False).agg(weight=("sim", "mean"))


def pure_text_ids(arm: rk.Arm) -> set[str]:
    """Listed accounts in communities that fall apart (largest component under
    `MIN_SIZE`) once the text-only edges are removed: communities that exist
    only because of the text trace."""
    members = arm.report.members
    out: set[str] = set()
    for c in arm.listed["community"].unique():
        nodes = members.loc[members["community"] == c, "user_id"].tolist()
        if rk.non_text_component(arm, nodes) < 4:
            out |= set(arm.listed.loc[arm.listed["community"] == c, "user_id"])
    return out


def summarise(arm: rk.Arm, other: rk.Arm | None, none: rk.Arm) -> dict:
    listed = arm.listed
    ids = set(listed["user_id"])
    text_edges = arm.text_edge_set()
    with_text = {u for e in text_edges for u in e}
    behavioural = {u for n, d in arm.nets.items() if n != "text_similarity"
                   for u in text_users(d)}
    comm = arm.report.communities
    out = {
        "arm": arm.name,
        "text_edges_after_floor": len(text_edges),
        "graph_nodes": arm.graph.number_of_nodes(),
        "graph_edges": arm.graph.number_of_edges(),
        "communities": len(comm),
        "communities_kept": int(comm["kept"].sum()) if len(comm) else 0,
        "listed": len(listed),
        "listed_communities": int(listed["community"].nunique()) if len(listed) else 0,
        "listed_with_text_edge": len(ids & with_text),
        "listed_text_only": len(ids & (with_text - behavioural)),
        "listed_mean_kenya": float(listed["kenya_model"].mean()) if len(listed) else None,
    }
    pure = pure_text_ids(arm)
    out["listed_pure_text"] = len(pure)
    out["listed_share_pure_text"] = round(len(pure) / max(len(ids), 1), 4)
    if other is not None:
        full_pure = pure_text_ids(other)
        out["pure_text_retained_of_full"] = len(pure & full_pure)
        out["pure_text_full"] = len(full_pure)
        out["listed_text_only_retained_of_full"] = len(
            (ids & (with_text - behavioural)) & {u for e in other.text_edge_set() for u in e} & ids
            & (set(other.listed["user_id"])))
    out["listed_share_with_text_edge"] = round(out["listed_with_text_edge"] / max(len(ids), 1), 4)
    out["listed_share_text_only"] = round(out["listed_text_only"] / max(len(ids), 1), 4)
    ref = {"none": none, "full": other}
    for name, r in ref.items():
        if r is None or r is arm:
            continue
        rids = set(r.listed["user_id"])
        out[f"vs_{name}_overlap"] = len(ids & rids)
        out[f"vs_{name}_jaccard"] = round(len(ids & rids) / max(len(ids | rids), 1), 4)
        top = lambda a: set(a.glob.sort_values(ascending=False).head(500).index)
        out[f"vs_{name}_global_top500_overlap"] = len(top(arm) & top(r))
    return out


def community_table(full: rk.Arm, none: rk.Arm) -> pd.DataFrame:
    """Per listed community of `full`: how much of it is text, and does it
    exist without the text trace."""
    comm = full.report.communities
    members = full.report.members
    listed_c = sorted(full.listed["community"].unique())
    match = rk.best_match(members[members["community"].isin(listed_c)], none.report.members)
    none_listed = set(none.listed["community"].unique())
    rows = []
    for c in listed_c:
        nodes = members.loc[members["community"] == c, "user_id"].tolist()
        info = comm[comm["community"] == c].iloc[0]
        m = match[match["community"] == c].iloc[0]
        rows.append({
            "community": int(c), "size": len(nodes),
            "listed": int((full.listed["community"] == c).sum()),
            "edges": int(info["edges"]),
            "share_text": float(info.get("share_text_similarity", 0.0) or 0.0),
            "largest_non_text_component": rk.non_text_component(full, nodes),
            "best_match_none": int(m["match"]), "match_jaccard": round(float(m["jaccard"]), 3),
            "match_listed_in_none": bool(int(m["match"]) in none_listed),
            "kenya_model": float(info["kenya_model"]) if pd.notna(info["kenya_model"]) else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", default=WINDOW)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--days", type=int, nargs="*", default=list(DAYS))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    base = wdir(args.window)
    src = base / f"pairs_{args.tag}"
    out = base / f"arms_{args.tag}"
    out.mkdir(exist_ok=True)
    pairs = pd.read_parquet(src / "pairs.parquet")
    collected = pd.read_parquet(base / "collected.parquet") if (base / "collected.parquet").exists() else None
    pairs = pairs.assign(
        t_a=pd.to_datetime(pairs["t_a"], utc=True), t_b=pd.to_datetime(pairs["t_b"], utc=True),
        later=lambda d: d[["t_a", "t_b"]].max(axis=1).dt.floor("D"),
        earlier=lambda d: d[["t_a", "t_b"]].min(axis=1),
    )
    end = max(pairs["t_a"].max(), pairs["t_b"].max()).normalize()
    log.info("latest post day in the trace: %s", end.date())
    if collected is not None:
        last = collected.set_index("post_id")["last_dt"]
        pairs = pairs.assign(
            col_a=pairs["post_a"].map(last), col_b=pairs["post_b"].map(last),
        )
        pairs["col_min"] = pairs[["col_a", "col_b"]].min(axis=1)
    cut = lambda n: end - pd.Timedelta(days=n - 1)

    arms_text: dict[str, pd.DataFrame | None] = {"none": None, "full": to_edges(pairs)}
    for n in args.days:
        arms_text[f"a{n}"] = to_edges(pairs[pairs["later"] >= cut(n)])
        arms_text[f"b{n}"] = to_edges(pairs[pairs["earlier"] >= cut(n)])
        if collected is not None:
            floor_day = (cut(n)).strftime("%Y-%m-%d")
            arms_text[f"c{n}"] = to_edges(pairs[pairs["col_min"].astype(str) >= floor_day])

    con = connect()
    con.execute("SET enable_progress_bar = false")
    nets_none = rk.nets_for(args.window, None)
    users = set()
    for d in nets_none.values():
        users |= text_users(d)
    counts = pd.read_parquet(base / "floor_counts.parquet")
    full_nets = rk.nets_for(args.window, arms_text["full"])
    users |= text_users(full_nets["text_similarity"])
    kenya = rk.ensure_kenya(con, args.window, sorted(users))
    log.info("relevance frame: %d users", len(kenya))

    arms: dict[str, rk.Arm] = {}
    for name, text in arms_text.items():
        nets = rk.nets_for(args.window, text)
        arms[name] = rk.rank(name, nets, kenya)
        arm = arms[name]
        arm.listed.to_parquet(out / f"listed_{name}.parquet")
        arm.report.communities.to_parquet(out / f"communities_{name}.parquet")
        arm.report.members.to_parquet(out / f"members_{name}.parquet")
        log.info("%s: %d text edges, %d listed in %d communities", name,
                 len(arm.text_edge_set()), len(arm.listed), arm.listed["community"].nunique())

    summaries = [summarise(a, arms["full"], arms["none"]) for a in arms.values()]
    table = pd.DataFrame(summaries)
    table.to_csv(out / "arms.csv", index=False)
    comm = community_table(arms["full"], arms["none"])
    comm.to_csv(out / "communities_full_vs_none.csv", index=False)
    print(table.to_string(index=False))
    print()
    print(comm.to_string(index=False))

    # Who is listed in one arm and not the other.
    f, n = set(arms["full"].listed["user_id"]), set(arms["none"].listed["user_id"])
    text_nodes = {u for e in arms["full"].text_edge_set() for u in e}
    behavioural = {u for d in nets_none.values() for u in text_users(d)}
    detail = {
        "full_only": len(f - n), "none_only": len(n - f), "both": len(f & n),
        "full_only_with_text_edge": len((f - n) & text_nodes),
        "full_only_text_only_accounts": len((f - n) & (text_nodes - behavioural)),
        "none_only_in_graph_of_full": len((n - f) & set(arms["full"].graph.nodes)),
        "full_only_mean_kenya": float(kenya.reindex(sorted(f - n))["kenya_model"].mean()),
        "none_only_mean_kenya": float(kenya.reindex(sorted(n - f))["kenya_model"].mean()),
        "both_mean_kenya": float(kenya.reindex(sorted(f & n))["kenya_model"].mean()),
        "end_day": str(end.date()),
        "text_trace_users_total": len(text_nodes),
        "text_trace_users_text_only_in_graph": len(text_nodes - behavioural),
    }
    (out / "detail.json").write_text(json.dumps(detail, indent=2))
    print(json.dumps(detail, indent=2))


if __name__ == "__main__":
    main()

"""Overlap-floor sweep on the live 21d window.

    uv run python 03_sweep.py --bundle <dir with communities.json, g<cid>.json>

Inputs: out/<window>/pairs_manifest/pairs_all.parquet (every cross-author post pair at cosine >= 0.85,
with `overlap` = production char-4gram Jaccard of cleaned text), rows.parquet (text), nets/ and
floor_counts.parquet from 01b_nets.py. Each arm keeps the pairs whose chosen overlap metric reaches the
floor, builds the user-level text trace exactly as production does (mean cosine per pair per day of the
later post, then mean over days), and runs fuse -> centrality -> community report with production
defaults. Read-only on R2 (relevance query, cached).
"""

import argparse
import json
import logging
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kma.db import connect

from common import WINDOW, wdir
import rank as rk

log = logging.getLogger("sweep")
OUT = Path(__file__).parent / "out"
CASES = [0, 8, 35, 41, 42, 45, 77, 90, 101]
NOISE = re.compile(r"https?://\S+|@\w+")


def words(text: str) -> frozenset:
    return frozenset(re.findall(r"\w+", NOISE.sub(" ", str(text).lower())))


def word_jaccard(a: frozenset, b: frozenset) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def prod_edges(pairs: pd.DataFrame) -> pd.DataFrame:
    """The persisted trace: mean cosine per (pair, day of the later post), then mean over days."""
    if pairs.empty:
        return pd.DataFrame({"source": [], "target": [], "weight": []})
    lo = np.where(pairs["user_a"] <= pairs["user_b"], pairs["user_a"], pairs["user_b"])
    hi = np.where(pairs["user_a"] <= pairs["user_b"], pairs["user_b"], pairs["user_a"])
    later = pd.to_datetime(np.maximum(pairs["t_a"].to_numpy(), pairs["t_b"].to_numpy()), utc=True).floor("D")
    bucket = pairs.assign(source=lo, target=hi, bucket=later).groupby(
        ["source", "target", "bucket"], as_index=False).agg(weight=("sim", "mean"))
    return bucket.groupby(["source", "target"], as_index=False).agg(weight=("weight", "mean"))


def best(members: pd.DataFrame, ids: set[str]) -> tuple[int, int, float]:
    owner = members.groupby("community")["user_id"].agg(set)
    hit = {c: len(ids & s) for c, s in owner.items() if ids & s}
    if not hit:
        return -1, 0, 0.0
    c = max(hit, key=lambda k: hit[k] / len(ids | owner[k]))
    return int(c), len(owner[c]), round(hit[c] / len(ids | owner[c]), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--window", default=WINDOW)
    ap.add_argument("--arms", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    base = wdir(args.window)
    pairs = pd.read_parquet(base / "pairs_manifest" / "pairs_all.parquet")
    rows = pd.read_parquet(base / "rows.parquet", columns=["post_id", "text"]).set_index("post_id")["text"]
    log.info("pairs at cosine>=0.85 cross-author: %d", len(pairs))

    t = time.monotonic()
    cache = base / "pairs_manifest" / "pairs_wj.parquet"
    if cache.exists():
        pairs = pd.read_parquet(cache)
    else:
        w = {}
        def tok(p):
            if p not in w:
                w[p] = words(rows[p])
            return w[p]
        pairs["wj"] = [word_jaccard(tok(a), tok(b)) for a, b in zip(pairs["post_a"], pairs["post_b"])]
        pairs.to_parquet(cache)
    log.info("word jaccard computed (%.0fs)", time.monotonic() - t)

    known = {}
    for cid in CASES:
        g = json.loads((args.bundle / f"g{cid}.json").read_text())
        known[cid] = set(map(str, g["nodes"]["id"]))

    arms = [(a.split(":")[0], float(a.split(":")[1])) for a in args.arms.split(",") if a] or [("char4", 0.10), ("char4", 0.20), ("char4", 0.30), ("char4", 0.45), ("char4", 0.60),
            ("word", 0.30), ("word", 0.45), ("word", 0.60)]
    con = connect()
    con.execute("SET enable_progress_bar = false")
    nets_base = rk.nets_for(args.window, None)
    full = prod_edges(pairs[pairs["overlap"] >= 0.10])
    users = {u for d in nets_base.values() for u in set(d["source"]) | set(d["target"])}
    users |= set(full["source"]) | set(full["target"])
    kenya = rk.ensure_kenya(con, args.window, sorted(users))

    results, case_rows, result_arms = [], [], {}
    for metric, floor in arms:
        started = time.monotonic()
        col = "overlap" if metric == "char4" else "wj"
        kept = pairs[(pairs[col] >= floor) & (pairs["overlap"] >= (0.10 if metric == "char4" else 0.0))]
        text = prod_edges(kept)
        nets = rk.nets_for(args.window, text)
        arm = rk.rank(f"{metric}{floor:.2f}", nets, kenya)
        tedges = arm.text_edge_set()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
        listed = set(arm.listed["user_id"])
        result_arms[(metric, floor)] = arm
        comm = arm.report.communities
        mem = arm.report.members
        text_only = 0
        for c in arm.listed["community"].unique():
            nodes = mem.loc[mem["community"] == c, "user_id"].tolist()
            if rk.non_text_component(arm, nodes) < 4:
                text_only += 1
        results.append({
            "metric": metric, "floor": floor, "post_pairs": len(kept), "user_text_edges": len(tedges),
            "text_users": len({u for e in tedges for u in e}),
            "communities": len(comm), "kept": int(comm["kept"].sum()),
            "listed": len(listed), "listed_communities": int(arm.listed["community"].nunique()),
            "text_only_listed_communities": text_only,
            "secs": round(time.monotonic() - started, 1), "peak_rss_gb": round(rss, 2),
        })
        for cid, ids in known.items():
            c, size, jac = best(mem, ids)
            row = comm[comm["community"] == c]
            case_rows.append({
                "metric": metric, "floor": floor, "cid": cid, "bundle_size": len(ids),
                "match": c, "match_size": size, "jaccard": jac,
                "kept": bool(row["kept"].iloc[0]) if len(row) else None,
                "listed_in_match": int((arm.listed["community"] == c).sum()),
                "listed_known_members": len(listed & ids),
                "text_edges_inside": sum(1 for a, b in tedges if a in ids and b in ids),
            })
        log.info("%s", results[-1])
        arm.listed.to_parquet(OUT / f"listed_{metric}{floor:.2f}.parquet")
        mem.to_parquet(OUT / f"members_{metric}{floor:.2f}.parquet")
        comm.to_parquet(OUT / f"communities_{metric}{floor:.2f}.parquet")

    ref = set(pd.read_parquet(OUT / "listed_char40.10.parquet")["user_id"])
    for r in results:
        listed = set(result_arms[(r["metric"], r["floor"])].listed["user_id"])
        r["listed_overlap_current"] = len(listed & ref)
        r["listed_jaccard_current"] = round(len(listed & ref) / max(len(listed | ref), 1), 4)
    res = pd.DataFrame(results)
    res.to_csv(OUT / f"sweep{args.tag}.csv", index=False)
    cases = pd.DataFrame(case_rows)
    cases.to_csv(OUT / f"sweep_cases{args.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.to_string(index=False))
    print(cases.to_string(index=False))


if __name__ == "__main__":
    main()

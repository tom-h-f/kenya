"""Score the v2 ranking at each text-similarity threshold.

    cd analysis && uv run python investigations/2026-09-11-textsim-sensitivity/02_score.py \\
        investigations/2026-09-11-textsim-sensitivity/out/2026-09-05-promotion-off/sweep_min0.70

Builds the four threshold-independent bipartite traces once, then for every
`edges_t*.parquet` runs what `coord2_run.run` runs - text floor, fusion,
eigenvector centrality, top 500, Kenya share - so only the text trace varies.

Two checks gate every comparison:
  1. the local 0.85 text trace must match the production trace in R2, edge for edge;
  2. the local 0.85 top 500 must be the same accounts as the persisted production run.
If either fails, the sweep measures the GPU box rather than the threshold.

Also writes the fused network among the 0.85 top 500, which A2's matched sample
splits into groups (`03_a2_sample.py`).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from kma import bench, coord2, coord2_run
from kma.db import BUCKET, connect

BASELINE = 0.85
TOP = 500
PRODUCTION_SCORES = "coord2/platform=x/kind=scores/dt=2026-09-08/run=20260908T081903Z.parquet"


def as_str_ids(edges: pd.DataFrame) -> pd.DataFrame:
    return edges.assign(source=edges["source"].astype(str), target=edges["target"].astype(str))


def same_edges(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    key = ["source", "target"]
    a = a.sort_values(key).reset_index(drop=True)
    b = b.sort_values(key).reset_index(drop=True)
    if not a[key].equals(b[key]):
        return False
    return bool(np.allclose(a["weight"].to_numpy(), b["weight"].to_numpy(), rtol=1e-6))


def rank_top(con, view, networks, text_edges):
    floored = coord2_run.apply_text_floor(con, text_edges, view)
    nets = {**networks, "text_similarity": floored}
    scores = coord2.detect(nets, threshold=0.0)
    top = scores.sort_values("centrality", ascending=False).head(TOP).reset_index(drop=True)
    return coord2_run.attach_relevance(con, top, view), nets, len(floored)


def compare(top: pd.DataFrame, base: pd.DataFrame) -> dict:
    ids = top["user_id"].astype(str).tolist()
    base_ids = base["user_id"].astype(str).tolist()
    rank = {u: i for i, u in enumerate(ids)}
    base_rank = {u: i for i, u in enumerate(base_ids)}
    common = [u for u in ids if u in base_rank]
    rho = (
        spearmanr([base_rank[u] for u in common], [rank[u] for u in common]).statistic
        if len(common) > 2 else float("nan")
    )
    share = top["kenya_share"]
    return {
        "overlap": len(common),
        "jaccard": round(len(common) / len(set(ids) | set(base_ids)), 3),
        "spearman_on_overlap": round(float(rho), 3),
        "top50_overlap": len(set(ids[:50]) & set(base_ids[:50])),
        "kenya_mean": round(float(share.mean()), 3),
        "kenya_median": round(float(share.median()), 3),
        "kenya_ge_15pct": int((share >= 0.15).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--snapshot", default="2026-09-05-promotion-off")
    args = ap.parse_args()

    files = {float(p.stem.split("_t")[1]): p for p in sorted(args.sweep_dir.glob("edges_t*.parquet"))}
    if BASELINE not in files:
        raise SystemExit(f"no edges_t{BASELINE:.2f}.parquet in {args.sweep_dir}")

    con = connect()
    manifest = bench.load(args.snapshot, con=con)
    view = coord2.posts_view(bench.pinned_source(manifest, "posts"))

    start = time.perf_counter()
    networks, _ = coord2_run.build_networks(con, view)
    print(f"bipartite traces in {time.perf_counter() - start:.0f}s: "
          + ", ".join(f"{name} {len(edges):,}" for name, edges in networks.items()))

    local = as_str_ids(pd.read_parquet(files[BASELINE]))
    production = coord2_run.load_textsim(con, args.snapshot, BASELINE)
    check1 = same_edges(local, production)
    print(f"check 1, text trace at {BASELINE}: local {len(local):,} vs production {len(production):,} edges -> "
          f"{'IDENTICAL' if check1 else 'DIFFERENT'}")

    tops, text_edges = {}, {}
    for threshold, path in sorted(files.items()):
        start = time.perf_counter()
        top, nets, text_edges[threshold] = rank_top(con, view, networks, as_str_ids(pd.read_parquet(path)))
        tops[threshold] = top
        top.to_parquet(args.sweep_dir / f"top{TOP}_t{threshold:.2f}.parquet")
        if threshold == BASELINE:
            sub = coord2.fuse(nets).subgraph(set(top["user_id"].astype(str)))
            pd.DataFrame(list(sub.edges()), columns=["source", "target"]).to_parquet(
                args.sweep_dir / f"top{TOP}_fused_edges_t{threshold:.2f}.parquet"
            )
        print(f"  t={threshold:.2f} ranked in {time.perf_counter() - start:.0f}s "
              f"({text_edges[threshold]:,} text edges after the floor)")

    production_top = con.sql(
        f"SELECT CAST(user_id AS VARCHAR) AS user_id FROM read_parquet('r2://{BUCKET}/{PRODUCTION_SCORES}')"
    ).df()
    overlap = len(set(production_top["user_id"]) & set(tops[BASELINE]["user_id"].astype(str)))
    check2 = overlap == TOP
    print(f"check 2, top {TOP} at {BASELINE}: {overlap} of {TOP} accounts shared with production -> "
          f"{'SAME' if check2 else 'DIFFERENT'}")

    summary = pd.DataFrame(
        [{"threshold": t, "text_edges": text_edges[t], **compare(tops[t], tops[BASELINE])} for t in sorted(tops)]
    )
    summary.to_csv(args.sweep_dir / "summary.csv", index=False)
    print()
    print(summary.to_string(index=False))
    if not (check1 and check2):
        print("\nWARNING: a gating check failed; read nothing above until it is explained.")


if __name__ == "__main__":
    main()

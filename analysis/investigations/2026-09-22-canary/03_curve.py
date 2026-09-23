"""Detection-strength curve on a SMALL local sample: the smallest planted
operation the v2 chain surfaces.

    cd analysis && uv run python investigations/2026-09-22-canary/03_curve.py DATA_DIR \\
        [--sizes 5,10,20,50] [--activity low,normal] [--seeds 0]

DATA_DIR is `02_background.py`'s output: one week of `2026-09-14-deep500`
(245,450 posts, 101,558 accounts) and the persisted text trace restricted to
it. A week of one snapshot, not the production window, because the Modal
workspace is over its spend limit and the mac is shared - so this is a
small-sample curve, and the full-scale one is pending (commands in
`findings.md`).
"""

import argparse
import json
import resource
import time
from pathlib import Path

import duckdb
import pandas as pd
from sentence_transformers import SentenceTransformer

from kma import canary, coord2, coord2_run
from kma.semantic import MODEL


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def point(con, base_view: str, bg_text: pd.DataFrame, encoder, *, size: int,
          activity: str, seed: int, start) -> dict:
    t0 = time.monotonic()
    cid = f"curve-{size}-{activity}-{seed}"
    posts = canary.plant(cid, size=size, activity=activity, seed=seed, start=start,
                         targets=canary.retweet_targets(con, base_view))
    view = canary.inject_view(con, base_view, posts)
    networks, sizes = coord2_run.build_networks(con, view)
    text = pd.concat([bg_text, canary.text_edges(posts, encoder)], ignore_index=True)
    networks["text_similarity"] = coord2_run.apply_text_floor(con, text, view)
    graph = coord2.fuse(networks)
    scores = coord2.centrality(graph)
    planted = [canary.account_id(cid, n) for n in range(size)]
    result = canary.evaluate(graph, scores, planted)
    by_trace = {name: len(set(planted) & (set(e["source"]) | set(e["target"])))
                for name, e in networks.items()}
    return {"size": size, "activity": activity, "seed": seed,
            "planted_posts": len(posts), "planted_in_trace": by_trace, **result,
            "graph_nodes": graph.number_of_nodes(), "seconds": round(time.monotonic() - t0, 1),
            "peak_rss_gb": round(rss_gb(), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("--sizes", default="5,10,20,50")
    ap.add_argument("--activity", default="low,normal")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    con = duckdb.connect()
    con.execute("SET memory_limit='1500MB'")
    base = f"SELECT * FROM read_parquet('{args.data / 'bg_posts.parquet'}')"
    canary.check_tags_unused(con, base)
    bg_text = pd.read_parquet(args.data / "bg_text.parquet").astype({"source": str, "target": str})
    start = pd.Timestamp(con.sql(f"SELECT min(created_at) FROM ({base})").fetchone()[0]).to_pydatetime()
    encoder = SentenceTransformer(MODEL, device="cpu")

    rows = []
    for activity in args.activity.split(","):
        for size in (int(s) for s in args.sizes.split(",")):
            for seed in (int(s) for s in args.seeds.split(",")):
                row = point(con, base, bg_text, encoder, size=size, activity=activity,
                            seed=seed, start=start)
                print(json.dumps(row, default=str), flush=True)
                rows.append(row)
    if args.out:
        pd.DataFrame(rows).to_json(args.out, orient="records", indent=1, default_handler=str)


if __name__ == "__main__":
    main()

"""Noise floor for Q1: how much does the listing move with NO text signal change?

    cd analysis && uv run python investigations/2026-10-10-text-trace-value/04_null.py \
        --tag manifest|current [--perturb N]

Adding or removing the text trace moves the listing (03_arms), but so does
anything else: Leiden's membership on a 12,000-account component is sensitive,
and the listing is the top 25 of each community by within-community
eigenvector. This puts a number on that sensitivity so the text arm can be read
against it.

  seed      the same graph, Leiden seed 1..5 instead of 0
  drop<K>   the same graph minus K edges chosen at random from the largest
            bipartite network, K = the text edge count of the arm under test
            (5 draws, seed 0)

Each is compared with the same graph's seed-0 listing (Jaccard of listed
accounts). The text effect is the Jaccard of `none` against `full`.
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd

from kma.db import connect

from common import WINDOW, wdir
import rank as rk

log = logging.getLogger("null")


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", default=WINDOW)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--draws", type=int, default=5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    base = wdir(args.window)
    out = base / f"arms_{args.tag}"

    text = pd.read_parquet(base / f"pairs_{args.tag}" / "edges.parquet")
    con = connect()
    con.execute("SET enable_progress_bar = false")
    nets_full = rk.nets_for(args.window, text)
    nets_none = rk.nets_for(args.window, None)
    k_text = len(nets_full["text_similarity"])
    ids = set()
    for d in nets_full.values():
        ids |= set(d["source"]) | set(d["target"])
    kenya = rk.ensure_kenya(con, args.window, sorted(ids))

    result: dict = {"text_edges": k_text}
    base_none = rk.rank("none", nets_none, kenya)
    base_full = rk.rank("full", nets_full, kenya)
    ln, lf = set(base_none.listed["user_id"]), set(base_full.listed["user_id"])
    result["text_effect_jaccard"] = jaccard(ln, lf)
    result["listed_none"], result["listed_full"] = len(ln), len(lf)

    for name, nets, ref in (("none", nets_none, ln), ("full", nets_full, lf)):
        js = []
        for seed in range(1, args.draws + 1):
            arm = rk.rank(f"{name}-seed{seed}", nets, kenya, seed=seed)
            js.append(jaccard(ref, set(arm.listed["user_id"])))
            log.info("%s Leiden seed %d: listed %d, Jaccard with seed 0 %.3f",
                     name, seed, len(arm.listed), js[-1])
        result[f"{name}_leiden_seed_jaccard"] = js

    rng = np.random.default_rng(0)
    big = max(("co_retweet", "co_url", "hashtag_sequence", "fast_retweet"),
              key=lambda n: len(nets_none.get(n, [])))
    js = []
    for draw in range(args.draws):
        nets = {n: d.copy() for n, d in nets_none.items()}
        e = nets[big]
        drop = rng.choice(len(e), size=min(k_text, len(e)), replace=False)
        nets[big] = e.drop(e.index[drop]).reset_index(drop=True)
        arm = rk.rank(f"none-drop{draw}", nets, kenya)
        js.append(jaccard(ln, set(arm.listed["user_id"])))
        log.info("none minus %d random %s edges (draw %d): Jaccard %.3f",
                 k_text, big, draw, js[-1])
    result["none_random_drop_jaccard"] = js
    result["dropped_from"] = big

    (out / "null.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

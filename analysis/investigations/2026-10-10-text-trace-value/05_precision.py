"""Q3 (bulk): what the live text edges link, by word overlap, per cosine stratum.

    cd analysis && uv run python investigations/2026-10-10-text-trace-value/05_precision.py \
        --tag manifest|current

Takes the post pairs behind the persisted-method trace (02_pairs.py, floors
applied) and measures, for every pair rather than a sample:

  token Jaccard   cleaned-word sets, as 2026-09-11's `06_pair_audit.py`
  char-4gram      the production floor's own statistic (already >= 0.10)
  identical text  cleaned texts equal (templates, greeting farms)

and the same per account pair (edge): its best pair's token Jaccard.
Strata are the post-pair cosine. Also writes the stratified edge sample the
blind read draws from (`sample.parquet`, seeded, one post pair per edge).

Texts and ids stay in out/. Nothing here reads R2.
"""

import argparse
import json

import numpy as np
import pandas as pd

from common import WINDOW, wdir

BINS = [0.85, 0.88, 0.92, 0.96, 1.0001]
LABELS = ["0.85-0.88", "0.88-0.92", "0.92-0.96", "0.96-1.00"]
PER_STRATUM = 15


def tokens(s: str) -> frozenset:
    return frozenset(s.split()) if s else frozenset()


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", default=WINDOW)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    base = wdir(args.window)
    out = base / f"pairs_{args.tag}"
    pairs = pd.read_parquet(out / "pairs.parquet")
    rows = pd.read_parquet(base / "rows.parquet", columns=["post_id", "text", "clean"]).set_index("post_id")
    pairs = pairs.assign(
        clean_a=pairs["post_a"].map(rows["clean"]), clean_b=pairs["post_b"].map(rows["clean"]),
    )
    pairs["tok_j"] = [jaccard(tokens(a), tokens(b)) for a, b in zip(pairs["clean_a"], pairs["clean_b"])]
    pairs["identical"] = pairs["clean_a"] == pairs["clean_b"]
    pairs["stratum"] = pd.cut(pairs["sim"], BINS, labels=LABELS, right=False)

    def block(df: pd.DataFrame) -> dict:
        return {
            "pairs": int(len(df)),
            "median_token_jaccard": round(float(df["tok_j"].median()), 3),
            "p75_token_jaccard": round(float(df["tok_j"].quantile(0.75)), 3),
            "share_tok_j_0": round(float((df["tok_j"] == 0).mean()), 4),
            "share_tok_j_lt_0.1": round(float((df["tok_j"] < 0.1).mean()), 4),
            "share_tok_j_ge_0.5": round(float((df["tok_j"] >= 0.5).mean()), 4),
            "share_identical_clean": round(float(df["identical"].mean()), 4),
            "median_char4gram": round(float(df["overlap"].median()), 3),
        }

    res = {"all_pairs": block(pairs), "by_cosine": {}}
    for label in LABELS:
        sub = pairs[pairs["stratum"] == label]
        if len(sub):
            res["by_cosine"][label] = block(sub) | {"share_of_pairs": round(len(sub) / len(pairs), 4)}

    edge = (pairs.groupby(["source", "target"])
            .agg(best_tok_j=("tok_j", "max"), n_pairs=("sim", "size"), mean_sim=("sim", "mean"),
                 any_identical=("identical", "max")).reset_index())
    res["edges"] = {
        "edges": int(len(edge)),
        "accounts": int(len(set(edge["source"]) | set(edge["target"]))),
        "median_best_token_jaccard": round(float(edge["best_tok_j"].median()), 3),
        "share_best_tok_j_lt_0.1": round(float((edge["best_tok_j"] < 0.1).mean()), 4),
        "share_best_tok_j_ge_0.5": round(float((edge["best_tok_j"] >= 0.5).mean()), 4),
        "share_with_identical_pair": round(float(edge["any_identical"].mean()), 4),
        "edges_single_pair": round(float((edge["n_pairs"] == 1).mean()), 4),
    }
    deg = pd.concat([edge["source"], edge["target"]]).value_counts()
    res["edges"]["max_account_degree"] = int(deg.iloc[0])
    res["edges"]["accounts_degree_ge_10"] = int((deg >= 10).sum())
    res["edges"]["share_edges_touching_top10_accounts"] = round(float(
        (edge["source"].isin(deg.index[:10]) | edge["target"].isin(deg.index[:10])).mean()), 4)

    rng = np.random.default_rng(args.seed)
    one = pairs.sample(frac=1.0, random_state=args.seed).drop_duplicates(["source", "target"])
    parts = []
    for label in LABELS:
        sub = one[one["stratum"] == label]
        parts.append(sub.sample(n=min(PER_STRATUM, len(sub)), random_state=args.seed))
    sample = pd.concat(parts)
    sample = sample.assign(
        text_a=sample["post_a"].map(rows["text"]), text_b=sample["post_b"].map(rows["text"]),
        population_share=sample["stratum"].map(
            {k: v["share_of_pairs"] for k, v in res["by_cosine"].items()}).astype(float),
    )
    sample.to_parquet(out / "sample.parquet")
    pairs.drop(columns=["clean_a", "clean_b"]).to_parquet(out / "pairs_scored.parquet")
    (out / "precision.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

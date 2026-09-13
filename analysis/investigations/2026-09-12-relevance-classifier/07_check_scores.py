"""Sanity-check the corpus-wide relevance scores before anything trusts them.

    cd analysis && uv run python investigations/2026-09-12-relevance-classifier/07_check_scores.py

Reads `out/corpus.parquet` and `out/scores.parquet` - no R2 - and reports what
the model says about the whole corpus next to what the keyword gate says: the
share it calls Kenyan overall and per gate bucket, the shape of the probability
distribution, and samples of the two disagreements that matter (gate says
off-domain, model says Kenya; gate says Kenya, model does not).

The B2 evaluation was 91 posts. This is the check that the same model behaves
sanely on 1.28M, where a small false-positive rate in the `ambiguous` bucket -
74.7% of the corpus - is what would quietly inflate every Kenya share.
"""

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from kma import measure

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = HERE / "out"
    posts = pd.read_parquet(out / "corpus.parquet")
    scores = pd.read_parquet(out / "scores.parquet")
    frame = posts.merge(scores, on="platform_post_id")
    if len(frame) != len(posts):
        raise SystemExit(f"scores cover {len(frame):,} of {len(posts):,} posts")

    frame["bucket"] = [measure.domain_bucket(t) for t in frame["text"]]
    frame["model"] = frame["p_kenya"] >= args.threshold
    frame["gate"] = frame["bucket"] == "kenya"

    print(f"{len(frame):,} posts; gate calls {frame['gate'].mean():.3f} Kenyan, "
          f"model calls {frame['model'].mean():.3f} at p>={args.threshold}")
    print("\np_kenya quantiles:",
          np.round(np.quantile(frame["p_kenya"], [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]), 3).tolist())
    print(f"confident either way: p<0.1 on {float((frame['p_kenya'] < 0.1).mean()):.3f}, "
          f"p>0.9 on {float((frame['p_kenya'] > 0.9).mean()):.3f}")

    by_bucket = frame.groupby("bucket").agg(
        posts=("model", "size"), corpus_share=("model", lambda s: len(s) / len(frame)),
        model_says_kenya=("model", "mean"), mean_p=("p_kenya", "mean"))
    print("\nby gate bucket:")
    print(by_bucket.round(3).to_string())

    print("\nat other thresholds:", {t: round(float((frame["p_kenya"] >= t).mean()), 3)
                                     for t in (0.5, 0.7, 0.9, 0.95)})

    rng = np.random.default_rng(args.seed)
    for title, mask in (
        ("gate says off-domain, model says Kenya", (frame["bucket"] == "offdomain") & frame["model"]),
        ("gate says ambiguous, model says Kenya", (frame["bucket"] == "ambiguous") & frame["model"]),
        ("gate says Kenya, model does not", frame["gate"] & ~frame["model"]),
    ):
        rows = frame[mask]
        print(f"\n== {title}: {len(rows):,} posts")
        for r in rows.sample(min(args.samples, len(rows)), random_state=int(rng.integers(1 << 31))).itertuples():
            print(f"  p={r.p_kenya:.2f}  {textwrap.shorten(str(r.text), 150)}")


if __name__ == "__main__":
    main()

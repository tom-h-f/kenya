# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
#     "torch>=2.4",
#     "transformers>=4.48",
# ]
# ///
"""Score the whole 2026 corpus with the Plan D classifier (Modal A100).

    modal run modal_train.py --cmd "python 12_score_corpus.py --limit 500"
    modal run modal_train.py --cmd "python 12_score_corpus.py"

Reads out/dapt_corpus.parquet, writes out/corpus_scored.parquet. Scores are a
ranking signal for candidate mining, never labels (see PLAN-A-HANDOFF.md).
"""

from __future__ import annotations

import argparse
import time
from importlib import import_module

import pandas as pd

from _common import OUT

infer = import_module("04_infer")

MODEL_DIR = OUT / "model-d3-s1337"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="smoke subsample")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    corpus = pd.read_parquet(OUT / "dapt_corpus.parquet")
    total = len(corpus)
    df = corpus.sample(min(args.limit, total), random_state=42) if args.limit else corpus
    df = df.reset_index(drop=True)
    print(f"scoring {len(df)} of {total} rows with {args.model_dir}")

    start = time.time()
    scored = infer.predict(
        df["text"].astype(str).tolist(),
        model_dir=args.model_dir,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - start
    rate = len(df) / elapsed
    print(f"{elapsed:.1f}s for {len(df)} rows ({rate:.0f} rows/s)")
    print(f"projection for full {total}: {total / rate / 60:.1f} min")

    out = pd.concat(
        [df[["platform_post_id", "text"]], scored.drop(columns=["label_id"])], axis=1
    )
    name = "corpus_scored_sample.parquet" if args.limit else "corpus_scored.parquet"
    out.to_parquet(OUT / name, index=False)

    print(out["label"].value_counts().to_string())
    print(out[["p_neither", "p_offensive", "p_hate"]].describe().to_string())
    print(f"p_hate >= 0.20: {(out['p_hate'] >= 0.20).sum()}")
    print(f"wrote {OUT / name} ({len(out)} rows)")


if __name__ == "__main__":
    main()

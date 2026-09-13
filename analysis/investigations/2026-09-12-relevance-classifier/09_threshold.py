"""Where to put the cut, against a precision target rather than a default.

    cd analysis && uv run python investigations/2026-09-12-relevance-classifier/09_threshold.py \\
        --labels out --scores out/scores.parquet

0.5 is where the classifier was evaluated, not a tuned value, and it now gates
what the collector chases. This sweeps it on the labelled sets - Tom's 100
first, because those labels are his - and shows what each cut costs in recall
and what share of the corpus it admits.

Corpus-weighted, as `measure_eval.score` is: the `ambiguous` bucket is 75% of
the corpus, so unweighted numbers understate what a false positive there costs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def labelled(data: Path, path: str) -> pd.DataFrame:
    frame = pd.read_csv(data / path)
    if "bucket" not in frame:
        frame = frame.merge(
            pd.read_csv(data / "measure_sample.csv")[["post_id", "bucket", "stratum_share"]],
            on="post_id")
    return frame[frame["label"].isin(("kenya", "offdomain"))].reset_index(drop=True)


def sweep(frame: pd.DataFrame, p: np.ndarray, thresholds) -> pd.DataFrame:
    actual = (frame["label"] == "kenya").to_numpy()
    weight = (frame["stratum_share"].to_numpy()
              / frame.groupby("bucket")["bucket"].transform("size").to_numpy())
    rows = []
    for t in thresholds:
        predicted = p >= t
        tp = weight[predicted & actual].sum()
        fp = weight[predicted & ~actual].sum()
        fn = weight[~predicted & actual].sum()
        rows.append({
            "threshold": t,
            "precision": round(float(tp / (tp + fp)), 3) if tp + fp else None,
            "recall": round(float(tp / (tp + fn)), 3) if tp + fn else None,
            "flagged (unweighted)": int(predicted.sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, default=HERE / "out")
    ap.add_argument("--scores", type=Path, default=HERE / "out" / "scores.parquet")
    ap.add_argument("--model-dir", type=Path, help="score the labelled posts with this model")
    args = ap.parse_args()

    thresholds = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    scores = pd.read_parquet(args.scores)
    scores["platform_post_id"] = scores["platform_post_id"].astype(str)
    corpus = {t: round(float((scores["p_kenya"] >= t).mean()), 3) for t in thresholds}

    for name, path in (("Tom's 100", "measure_human.csv"), ("model-labelled 300", "measure_sample.csv")):
        frame = labelled(args.labels, path)
        if args.model_dir:
            from kma import relevance
            p = relevance.score_texts(args.model_dir, frame["text"].astype(str).tolist())
        else:
            merged = frame.assign(post_id=frame["post_id"].astype(str)).merge(
                scores, left_on="post_id", right_on="platform_post_id", how="left")
            if merged["p_kenya"].isna().any():
                raise SystemExit(f"{int(merged['p_kenya'].isna().sum())} labelled posts have no "
                                 "persisted score; pass --model-dir to score them directly")
            frame, p = merged, merged["p_kenya"].to_numpy()
        print(f"\n== {name} (n={len(frame)}), corpus-weighted")
        print(sweep(frame, p, thresholds).to_string(index=False))

    print("\nshare of the 1.28M-post corpus flagged at each cut:", corpus)


if __name__ == "__main__":
    main()

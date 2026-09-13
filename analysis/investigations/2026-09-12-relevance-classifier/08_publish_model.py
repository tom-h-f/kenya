"""Publish the trained relevance model to R2, and check it loads from there.

    cd analysis && uv run python investigations/2026-09-12-relevance-classifier/08_publish_model.py \\
        ~/data/relevance-model/model [--verify]

The model was trained on mike-pc and lived only there, so nothing else could
score. R2 rather than the HF Hub: no HF token exists on tac2 or in `.env`, and
every host that would score already holds R2 credentials - Modal through its
`kenya-r2` secret.

`--verify` downloads it back into a clean cache and re-scores the B2 label sets,
which is the acceptance test: a host with only R2 credentials reproduces the
numbers the model was accepted on.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from kma import relevance
from kma.db import connect


def b2_report(model_dir: Path, data: Path, threshold: float) -> dict:
    """Precision and recall against Tom's 100 and the model-labelled 300,
    corpus-weighted by the gate's buckets, as `measure_eval.score` does."""
    out = {}
    for name, path in (("human 100", "measure_human.csv"), ("model 300", "measure_sample.csv")):
        frame = pd.read_csv(data / path)
        if "bucket" not in frame:
            frame = frame.merge(
                pd.read_csv(data / "measure_sample.csv")[["post_id", "bucket", "stratum_share"]],
                on="post_id")
        frame = frame[frame["label"].isin(("kenya", "offdomain"))].reset_index(drop=True)
        predicted = relevance.score_texts(model_dir, frame["text"].astype(str).tolist()) >= threshold
        actual = (frame["label"] == "kenya").to_numpy()
        weight = (frame["stratum_share"].to_numpy()
                  / frame.groupby("bucket")["bucket"].transform("size").to_numpy())
        tp = weight[predicted & actual].sum()
        fp = weight[predicted & ~actual].sum()
        fn = weight[~predicted & actual].sum()
        out[name] = {"precision": round(float(tp / (tp + fp)), 3),
                     "recall": round(float(tp / (tp + fn)), 3), "n": int(len(frame))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path)
    ap.add_argument("--name", default=relevance.MODEL)
    ap.add_argument("--labels", type=Path, default=Path("out"), help="dir holding the B2 csvs")
    ap.add_argument("--threshold", type=float, default=relevance.THRESHOLD)
    ap.add_argument("--verify", action="store_true", help="download it back and re-score the B2 sets")
    args = ap.parse_args()

    keys = relevance.publish_model(args.source, args.name)
    print(f"published {len(keys)} files to {relevance.model_key(args.name)}")

    if not args.verify:
        return
    with tempfile.TemporaryDirectory() as cache:
        fetched = relevance.fetch_model(args.name, cache=Path(cache))
        print(f"fetched back to {fetched}: {sorted(p.name for p in fetched.iterdir())}")
        report = b2_report(fetched, args.labels, args.threshold)
    for name, scores in report.items():
        print(f"  {name}: precision {scores['precision']} recall {scores['recall']} (n={scores['n']})")


if __name__ == "__main__":
    main()

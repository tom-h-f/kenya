# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
#     "scikit-learn>=1.4",
#     "torch>=2.4",
#     "transformers>=4.48",
#     "matplotlib>=3.8",
# ]
# ///
"""Test-set report for the fine-tuned model: macro-F1, per-class metrics,
confusion matrix PNG, worst misclassifications, hate-class threshold sweep."""

from __future__ import annotations

import argparse
import json

import os

os.environ.pop("MPLBACKEND", None)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from _common import LABELS, OUT, load_split
from importlib import import_module

predict = import_module("04_infer").predict


def confusion_png(cm: np.ndarray, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=str(OUT / "model"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--prefix", default="03")
    args = ap.parse_args()

    test = load_split(args.split)
    scored = predict(test["text"].tolist(), model_dir=args.model_dir)
    pred = scored["label_id"].values
    true = test["label"].values

    macro = f1_score(true, pred, average="macro")
    print(classification_report(true, pred, target_names=LABELS, digits=3))
    print(f"macro-F1: {macro:.4f}")

    cm = confusion_matrix(true, pred)
    confusion_png(cm, OUT / f"{args.prefix}_confusion.png")

    errors = test.assign(
        pred=[LABELS[p] for p in pred],
        true=[LABELS[t] for t in true],
        p_pred=scored[["p_neither", "p_offensive", "p_hate"]].max(axis=1).values,
    )
    errors = errors[errors["pred"] != errors["true"]]
    errors = errors.sort_values("p_pred", ascending=False)
    errors[["text", "true", "pred", "p_pred"]].to_csv(
        OUT / f"{args.prefix}_errors.csv", index=False
    )
    print(f"{len(errors)} errors -> out/{args.prefix}_errors.csv (worst = most confident)")

    rows = []
    for t in np.arange(0.5, 1.0, 0.05):
        flag = scored["p_hate"].values >= t
        if flag.sum() == 0:
            continue
        rows.append(
            {
                "threshold": round(t, 2),
                "flagged": int(flag.sum()),
                "precision": float((true[flag] == 2).mean()),
                "recall": float(flag[true == 2].mean()),
            }
        )
    sweep = pd.DataFrame(rows)
    print("\nhate-class threshold sweep (for triage tuning):")
    print(sweep.to_string(index=False))
    sweep.to_csv(OUT / f"{args.prefix}_hate_thresholds.csv", index=False)

    (OUT / f"{args.prefix}_metrics.json").write_text(
        json.dumps(
            {
                "macro_f1": macro,
                "report": classification_report(
                    true, pred, target_names=LABELS, output_dict=True
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

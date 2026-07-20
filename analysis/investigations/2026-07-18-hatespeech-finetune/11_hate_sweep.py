# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
#     "torch>=2.4",
#     "transformers>=4.56",
# ]
# ///
"""Sweep the hate decision threshold below argmax, for one model + split.

03_eval.py only sweeps 0.5-1.0, but argmax fires on p_hate < 0.5 whenever
hate is the top class, so recall above the argmax point is invisible there.
This sweeps the full range and reports the operating points that matter.
"""

from __future__ import annotations

import argparse
from importlib import import_module

import pandas as pd

from _common import OUT, load_split

predict = import_module("04_infer").predict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--split", default="test_unanimous")
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    df = load_split(args.split)
    scored = predict(df["text"].tolist(), model_dir=args.model_dir)
    y_true = (df["label"].values == 2)
    p_hate = scored["p_hate"].values

    rows = []
    for thr in [round(0.02 * i, 2) for i in range(1, 50)]:
        pred = p_hate >= thr
        tp = int((pred & y_true).sum())
        flagged = int(pred.sum())
        rows.append({
            "threshold": thr,
            "flagged": flagged,
            "precision": tp / flagged if flagged else 0.0,
            "recall": tp / int(y_true.sum()),
        })
    out = pd.DataFrame(rows)
    path = OUT / f"{args.prefix}_full_sweep.csv"
    out.to_csv(path, index=False)

    argmax_pred = scored["label_id"].values == 2
    tp = int((argmax_pred & y_true).sum())
    print(f"argmax: flagged {int(argmax_pred.sum())} "
          f"P {tp / max(argmax_pred.sum(), 1):.3f} R {tp / y_true.sum():.3f}")
    gate = out[out["recall"] >= 0.80]
    if len(gate):
        best = gate.loc[gate["precision"].idxmax()]
        print(f"best P at R>=0.80: thr {best['threshold']} "
              f"P {best['precision']:.3f} R {best['recall']:.3f} "
              f"flagged {int(best['flagged'])}")
    else:
        print("no threshold reaches recall 0.80")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

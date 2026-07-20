# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
#     "scikit-learn>=1.4",
# ]
# ///
"""Turn Plan A's labelled 2026 batch into train/val/test splits for round 2.

Reads out/labels_2026_full_final.parquet and writes, in the shape the rest
of the pipeline expects (text, label int, agreement):

  train2026.parquet  - the train split minus a val carve-out
  val2026.parquet    - model selection on 2026 data, NOT 2013 data
  gold.parquet       - random control (prevalence-honest test)
  challenge.parquet  - lexicon + NLI-tail rows (coded-term recall test)

`agreement` is written as 1.0 throughout: these labels come from a single
labeller, so no agreement signal exists. It is present only because
02_train.py's --extra-data path expects the column; never train with
--weight-by-agreement on this data, the number is a placeholder.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from _common import LABEL2ID, OUT, SEED

SOURCE = OUT / "labels_2026_full_final.parquet"


def shape(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["post_id", "text"]].copy()
    out["label"] = df["label"].map(LABEL2ID).astype(int)
    out["agreement"] = 1.0
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-size", type=int, default=300)
    args = ap.parse_args()

    df = pd.read_parquet(SOURCE)
    if df["label_source"].nunique() > 1:
        print(f"label sources: {df['label_source'].value_counts().to_dict()}")

    parts = {name: shape(g) for name, g in df.groupby("split")}
    train_all = parts["train"]
    train, val = train_test_split(
        train_all,
        test_size=args.val_size,
        random_state=SEED,
        stratify=train_all["label"],
    )

    written = {}
    for name, part in [
        ("train2026", train),
        ("val2026", val),
        ("gold", parts["gold"]),
        ("challenge", parts["challenge"]),
    ]:
        part = part.reset_index(drop=True)
        part.to_parquet(OUT / f"{name}.parquet", index=False)
        counts = part["label"].value_counts().sort_index().to_dict()
        written[name] = {"n": len(part), "by_label": {str(k): int(v) for k, v in counts.items()}}
        print(f"{name:12s} n={len(part):5d}  labels(0/1/2)={counts}")

    (OUT / "17_round2_splits.json").write_text(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()

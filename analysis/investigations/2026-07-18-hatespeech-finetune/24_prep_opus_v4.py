# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
#     "scikit-learn>=1.4",
# ]
# ///
"""Build Opus v4 train/val/challenge(/gold) splits from prior membership.

    uv run 24_prep_opus_v4.py
    uv run 24_prep_opus_v4.py --source out/labels_2026_opus-v4-full.parquet \\
        --suffix -full --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from _common import LABEL2ID, OUT, SEED

SOURCE = OUT / "labels_2026_opus-v4-partial.parquet"
ALLOWED_SPLITS = {"train", "challenge", "gold"}


def shape(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["post_id", "text"]].copy()
    out["label"] = df["label"].map(LABEL2ID).astype(int)
    out["agreement"] = 1.0
    return out.reset_index(drop=True)


def build_splits(
    df: pd.DataFrame, *, val_size: int, seed: int, suffix: str = ""
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if "split" not in df.columns:
        raise ValueError("labels are missing prior split membership")
    if df["split"].isna().any():
        missing = df.loc[df["split"].isna(), "post_id"].astype(str).tolist()[:5]
        raise ValueError(f"rows missing prior split: {missing}")

    unexpected = sorted(set(df["split"]) - ALLOWED_SPLITS)
    if unexpected:
        raise ValueError(f"unexpected prior splits: {unexpected}")

    challenge = df[df["split"] == "challenge"].copy()
    train_pool = df[df["split"] == "train"].copy()
    gold = df[df["split"] == "gold"].copy()

    if len(train_pool) <= val_size:
        raise ValueError(
            f"train pool too small for val_size={val_size}: n={len(train_pool)}"
        )

    train, val = train_test_split(
        train_pool,
        test_size=val_size,
        random_state=seed,
        stratify=train_pool["label"],
    )
    parts = {
        f"train2026_opus-v4{suffix}": shape(train),
        f"val2026_opus-v4{suffix}": shape(val),
        f"challenge_opus-v4{suffix}": shape(challenge),
    }
    if len(gold):
        parts[f"gold_opus-v4{suffix}"] = shape(gold)

    report = {
        name: {
            "n": len(frame),
            "by_label": {
                str(key): int(value)
                for key, value in frame["label"].value_counts().sort_index().items()
            },
        }
        for name, frame in parts.items()
    }
    report["notes"] = {
        "gold_opus-v4": (
            f"n={len(gold)}" if len(gold) else "unavailable: zero prior gold IDs"
        ),
        "source_rows": len(df),
        "val_size": val_size,
        "seed": seed,
        "suffix": suffix,
    }
    return parts, report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--val-size", type=int, default=250)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--suffix",
        default="",
        help="output name suffix, e.g. -full -> train2026_opus-v4-full.parquet",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.source)
    parts, report = build_splits(
        df, val_size=args.val_size, seed=args.seed, suffix=args.suffix
    )

    outputs = {name: OUT / f"{name}.parquet" for name in parts}
    report_path = OUT / f"24_opus_v4{args.suffix or ''}_splits.json".replace(
        "opus_v4_splits", "opus_v4_splits"
    )
    if args.suffix:
        report_path = OUT / f"24_opus_v4{args.suffix.lstrip('-')}_splits.json"
        if not str(report_path).endswith("_splits.json"):
            report_path = OUT / f"24_opus_v4{args.suffix}_splits.json"
    else:
        report_path = OUT / "24_opus_v4_splits.json"
    # normalize: suffix -full -> 24_opus_v4_full_splits.json
    if args.suffix:
        tag = args.suffix.lstrip("-")
        report_path = OUT / f"24_opus_v4_{tag}_splits.json"

    for path in [*outputs.values(), report_path]:
        if path.exists() and not args.force:
            raise SystemExit(f"{path} exists; pass --force to replace")

    for name, frame in parts.items():
        frame.to_parquet(outputs[name], index=False)
        print(
            f"{name:28s} n={len(frame):5d}  "
            f"labels(0/1/2)={frame['label'].value_counts().sort_index().to_dict()}"
        )
    report["artifacts"] = {name: str(path.name) for name, path in outputs.items()}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2", "pyarrow>=18", "datasets>=3", "datasketch>=1.6"]
# ///
"""AfriHate Swahili intake (Plan C): download, map labels, dedupe vs our
corpus, write out/afrihate_swa.parquet with source tags.

Label mapping: Hate -> 2 (hate), Abuse -> 1 (offensive), Normal -> 0
(neither). AfriHate "Abuse" = abusive/offensive language not targeting a
protected group - same boundary as Davidson offensive. License: CC BY-NC-SA
4.0 (non-commercial, cite paper arXiv:2501.08284).

Requires HF_TOKEN env var (gated dataset - accept terms on HF first).
"""

from __future__ import annotations

import os
import re

import pandas as pd

from _common import LABELS, OUT

LABEL_MAP = {"Normal": 0, "Abuse": 1, "Hate": 2}
NORM_RE = re.compile(r"[^a-z0-9 ]+")


def normalise(t: str) -> str:
    return NORM_RE.sub("", t.lower())


def main() -> None:
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set")

    parts = []
    for split in ["train", "validation", "test"]:
        ds = load_dataset("afrihate/afrihate", "swa", split=split, token=token)
        part = ds.to_pandas()
        part["afrihate_split"] = split
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)

    df["label"] = df["label"].map(LABEL_MAP)
    if df["label"].isna().any():
        raise SystemExit(f"unmapped labels: {df[df['label'].isna()]['label']}")
    df["label"] = df["label"].astype(int)
    df = df.rename(columns={"tweet": "text"})
    df["text"] = df["text"].str.strip()
    df["source"] = "afrihate_swa"
    df["agreement"] = 1.0

    before = len(df)
    df = df[df["text"].str.len() > 0].drop_duplicates("text")

    ours = pd.read_parquet(OUT / "clean.parquet")
    our_norms = set(ours["text"].map(normalise))
    overlap = df["text"].map(normalise).isin(our_norms)
    df = df[~overlap]
    print(f"{before} rows -> {len(df)} after dedup ({overlap.sum()} overlapped ours)")

    counts = df["label"].value_counts().sort_index()
    print(", ".join(f"{LABELS[i]}={counts.get(i, 0)}" for i in range(3)))
    df = df[["text", "label", "agreement", "source", "afrihate_split"]]
    df.to_parquet(OUT / "afrihate_swa.parquet", index=False)
    print(f"wrote out/afrihate_swa.parquet ({len(df)} rows)")


if __name__ == "__main__":
    main()

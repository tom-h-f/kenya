# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
# ]
# ///
"""Freeze the 2026 gold test ids and emit the training set (Plan A5/A6).

    uv run 16_gold_split.py --tag full

Gold = the **random control stratum only**. It is the one stratum no model
selected, so it is the only split on which a score means anything about the
corpus rather than about the miner. Model-mined rows are training data.

The lexicon and NLI strata are written as a separate `challenge` split: their
miners are rule-based and independent of the classifier, so they are not
circular, but they are not a random sample either - useful for coded-term
recall, useless for prevalence.

Caveat recorded in the output: with human adjudication dropped, every label
here is LLM-derived. A model scored on this set is measured against LLM
judgement, not human ground truth, and the gold set cannot detect a blind
spot the labellers share.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from _common import OUT

GOLD_STRATA = ["random_control"]
CHALLENGE_STRATA = ["lexicon", "nli_tail"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    adjudicated = OUT / f"labels_2026_{args.tag}_adjudicated.parquet"
    path = adjudicated if adjudicated.exists() else OUT / f"labels_2026_{args.tag}.parquet"
    print(f"labels: {path.name}")
    df = pd.read_parquet(path)
    df = df[df["label"].notna()].copy()

    split = pd.Series("train", index=df.index)
    split[df["stratum"].isin(GOLD_STRATA)] = "gold"
    split[df["stratum"].isin(CHALLENGE_STRATA)] = "challenge"
    df["split"] = split.values

    manifest = {
        "tag": args.tag,
        "labelled_rows": len(df),
        "gold_strata": GOLD_STRATA,
        "challenge_strata": CHALLENGE_STRATA,
        "human_verified": False,
        "labels_file": path.name,
        "caveat": (
            "All labels are LLM-derived. This run used a SINGLE labeller "
            "(Gemini 3.1 Pro) with no second opinion, no agreement metric and "
            "no adjudication, so there is no reliability estimate for them at "
            "all. Scores on this gold set measure agreement with one model's "
            "judgement, not human ground truth."
        ),
        "counts": {
            name: {
                "n": len(g),
                "labels": g["label"].value_counts().to_dict(),
                "sources": g["label_source"].value_counts().to_dict(),
            }
            for name, g in df.groupby("split", sort=False)
        },
        "gold_ids": sorted(df.loc[df["split"] == "gold", "post_id"].tolist()),
        "challenge_ids": sorted(df.loc[df["split"] == "challenge", "post_id"].tolist()),
    }

    df.to_parquet(OUT / f"labels_2026_{args.tag}_final.parquet", index=False)
    (OUT / "gold_2026_ids.json").write_text(json.dumps(manifest, indent=2))

    summary = {k: v for k, v in manifest.items() if not k.endswith("_ids")}
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT / f'labels_2026_{args.tag}_final.parquet'}")


if __name__ == "__main__":
    main()

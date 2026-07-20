# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "duckdb>=1",
#     "python-dotenv>=1",
#     "pandas>=2",
#     "pyarrow>=18",
#     "datasketch>=1.6",
#     "scikit-learn>=1.4",
# ]
# ///
"""Export the DAPT corpus: all collected X posts, cleaned and deduped.

Runs on tac2 only (needs R2 creds from the monorepo .env). Writes
out/dapt_corpus.parquet and copies it to the Drive working dir for Colab.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from importlib import import_module
from pathlib import Path

import pandas as pd

from _common import OUT, load_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from kma.db import connect, latest_posts  # noqa: E402

prep = import_module("00_prep")

DRIVE_OUT = Path.home() / "Drive/Colab/hatespeech-finetune/out"
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
NORM_RE = re.compile(r"[^a-z0-9 ]+")
EVAL_SPLITS = ["val", "test", "test_unanimous"]


def clean(text: str) -> str:
    return prep.WS_RE.sub(" ", URL_RE.sub(" ", text)).strip()


def norm_key(text: str) -> str:
    return NORM_RE.sub("", MENTION_RE.sub(" ", text.lower())).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="subsample for smoke; writes dapt_corpus_sample.parquet")
    ap.add_argument("--no-drive", action="store_true")
    args = ap.parse_args()

    counts: dict[str, int] = {}
    df = latest_posts(connect(), "x").df()[
        ["platform_post_id", "text", "collected_at"]
    ]
    counts["pulled"] = len(df)
    print(f"pulled {len(df)} latest posts from R2")

    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=42)
        counts["limited"] = len(df)

    df["text"] = df["text"].astype(str).map(clean)
    df = df[df["text"].str.len() >= 10]
    counts["after_clean_minlen"] = len(df)

    df["_norm"] = df["text"].map(norm_key)
    df = df.sort_values("collected_at").drop_duplicates("_norm", keep="last")
    counts["after_exact_dedupe"] = len(df)

    eval_keys = {
        norm_key(t) for name in EVAL_SPLITS for t in load_split(name)["text"]
    }
    before = len(df)
    df = df[~df["_norm"].isin(eval_keys)]
    counts["eval_overlap_dropped"] = before - len(df)
    print(f"eval-split overlap dropped: {before - len(df)}")

    clusters = prep.near_dup_clusters(df["text"])
    df = (
        df.assign(_cluster=clusters)
        .sort_values("collected_at")
        .drop_duplicates("_cluster", keep="last")
    )
    counts["after_near_dedupe"] = len(df)

    df = df[["text", "platform_post_id", "collected_at"]].reset_index(drop=True)
    name = "dapt_corpus_sample.parquet" if args.limit else "dapt_corpus.parquet"
    out_path = OUT / name
    df.to_parquet(out_path, index=False)
    (OUT / "08_export_report.json").write_text(json.dumps(counts, indent=2))
    print(f"wrote {out_path} ({len(df)} rows)")
    print(json.dumps(counts, indent=2))

    if not args.no_drive:
        DRIVE_OUT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, DRIVE_OUT / name)
        print(f"copied to {DRIVE_OUT / name}")


if __name__ == "__main__":
    main()

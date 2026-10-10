"""Blind read of text pairs, calibration for Q3 and Q4.

    cd analysis && uv run python investigations/2026-10-10-text-trace-value/06_read.py \
        --tag current --sample sample|q4_sample [--limit N] [--dry-run]

The reader, prompt and labels are 2026-09-11's `06_pair_audit.py` unchanged
(headless `claude -p`, claude-sonnet-5, no tools, no MCP, no user settings, one
fresh process in an empty directory; `same_message` / `same_topic` /
`unrelated`), so the counts are comparable with 26 / 15 / 59. There is no
Anthropic API key in this project. Total budget for this study: 80 pairs
(60 for Q3, 20 for Q4); the script refuses to exceed `--limit`, and the pairs
go out in one call per sample.

Pair order is shuffled and stripped of everything but the two texts.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

from common import WINDOW, wdir

AUDIT = (Path(__file__).resolve().parents[1] / "2026-09-11-textsim-sensitivity" / "06_pair_audit.py")


def load_audit():
    spec = importlib.util.spec_from_file_location("pair_audit", AUDIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", default=WINDOW)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sample", required=True, choices=["sample", "q4_sample"])
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = wdir(args.window) / f"pairs_{args.tag}" / f"{args.sample}.parquet"
    sample = pd.read_parquet(path)
    if len(sample) > args.limit:
        raise SystemExit(f"{len(sample)} pairs exceeds the --limit {args.limit}")
    shuffled = sample.sample(frac=1.0, random_state=3).reset_index(drop=True)
    if args.dry_run:
        print(f"{len(shuffled)} pairs; first prompt text:\n{shuffled.loc[0, 'text_a']!r}\n{shuffled.loc[0, 'text_b']!r}")
        return
    audit = load_audit()
    batch = shuffled.rename(columns={"text_a": "text_a", "text_b": "text_b"})
    labels = audit.judge_batch(batch[["text_a", "text_b"]])
    shuffled["label"] = labels
    shuffled.drop(columns=["text_a", "text_b"]).to_parquet(path.with_name(f"{args.sample}_read.parquet"))
    summary = {"pairs": len(shuffled), "unlabelled": int(shuffled["label"].isna().sum()),
               "labels": shuffled["label"].value_counts().to_dict()}
    if "stratum" in shuffled:
        summary["by_stratum"] = {
            str(k): v.to_dict()
            for k, v in shuffled.groupby("stratum", observed=True)["label"].value_counts().unstack(fill_value=0).iterrows()
        }
    print(json.dumps(summary, indent=2, default=str))
    path.with_name(f"{args.sample}_read.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

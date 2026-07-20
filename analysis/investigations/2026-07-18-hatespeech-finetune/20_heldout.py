# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
# ]
# ///
"""Create a fresh, model-blind human validation set for prompt v3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import pandas as pd

from _common import OUT, SEED

POOL_FRACTIONS = {
    "split_primary_soft": 0.30,
    "split_primary_hard": 0.10,
    "agree_hate": 0.15,
    "agree_offensive": 0.15,
    "random_remaining": 0.30,
}
DEFAULT_HALF_WIDTH = 0.10


def wilson_half_width(total: int, confidence: float = 0.95) -> float:
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    observed = 0.5
    denominator = 1 + z * z / total
    return (
        z
        * math.sqrt(
            observed * (1 - observed) / total + z * z / (4 * total * total)
        )
        / denominator
    )


def required_sample_size(
    half_width: float = DEFAULT_HALF_WIDTH,
    confidence: float = 0.95,
) -> int:
    """Smallest worst-case Wilson interval meeting the precision target."""
    if not 0 < half_width < 0.5:
        raise ValueError("half_width must be between 0 and 0.5")
    for total in range(1, 100_000):
        if wilson_half_width(total, confidence) <= half_width:
            return total
    raise RuntimeError("sample-size search did not converge")


def allocate_counts(total: int) -> dict[str, int]:
    raw = {name: total * fraction for name, fraction in POOL_FRACTIONS.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def sample_heldout(
    df: pd.DataFrame,
    *,
    n: int,
    excluded_ids: set[str],
    seed: int = SEED + 1,
) -> pd.DataFrame:
    available = df[
        ~df["post_id"].astype(str).isin({str(value) for value in excluded_ids})
    ].copy()
    available["post_id"] = available["post_id"].astype(str)
    gemini = available["label_gemini"]
    cursor = available["label_cursor"]
    masks = {
        "split_primary_soft": (gemini != "hate") & (cursor == "hate"),
        "split_primary_hard": (gemini == "hate") & (cursor != "hate"),
        "agree_hate": (gemini == cursor) & (gemini == "hate"),
        "agree_offensive": (gemini == cursor) & (gemini == "offensive"),
    }
    counts = allocate_counts(n)
    selected_ids: set[str] = set()
    picks = []
    for offset, name in enumerate(POOL_FRACTIONS):
        if name == "random_remaining":
            pool = available[~available["post_id"].isin(selected_ids)]
        else:
            pool = available[
                masks[name] & ~available["post_id"].isin(selected_ids)
            ]
        wanted = counts[name]
        if len(pool) < wanted:
            raise ValueError(f"{name}: need {wanted} rows, only {len(pool)} available")
        picked = pool.sample(wanted, random_state=seed + offset).assign(pool=name)
        selected_ids.update(picked["post_id"])
        picks.append(picked)
    result = pd.concat(picks, ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def read_jsonl_files(directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.glob("chunk_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if frame.empty or "post_id" not in frame:
        raise ValueError(f"no labelled rows found in {directory}")
    frame["post_id"] = frame["post_id"].astype(str)
    if frame["post_id"].duplicated().any():
        raise ValueError(f"duplicate post IDs in {directory}")
    return frame


def load_v2_source() -> pd.DataFrame:
    posts = read_jsonl_files(OUT / "chunks" / "full")[["post_id", "text"]]
    labellers = {
        "gemini": OUT / "labels" / "full" / "gemini-3.1-pro",
        "cursor": OUT / "labels" / "full" / "cursor-sonnet-4.5",
    }
    result = posts
    for name, directory in labellers.items():
        labels = read_jsonl_files(directory)[
            ["post_id", "label", "flags", "target_group", "confidence"]
        ].rename(
            columns={
                column: f"{column}_{name}"
                for column in ["label", "flags", "target_group", "confidence"]
            }
        )
        result = result.merge(labels, on="post_id", validate="one_to_one")
    return result


def load_excluded_ids() -> tuple[set[str], dict[str, int]]:
    calibration = pd.read_csv(
        OUT / "blind_check_coded_calibration.csv", dtype={"post_id": str}
    )
    split_meta = json.loads((OUT / "gold_2026_ids.json").read_text())
    calibration_ids = set(calibration["post_id"])
    gold_ids = {str(value) for value in split_meta["gold_ids"]}
    return calibration_ids | gold_ids, {
        "calibration": len(calibration_ids),
        "training_gold": len(gold_ids),
        "overlap": len(calibration_ids & gold_ids),
    }


def cmd_make(args: argparse.Namespace) -> None:
    sheet_path = OUT / "heldout_v3_human.csv"
    key_path = OUT / "heldout_v3_key.csv"
    parquet_path = OUT / "heldout_v3.parquet"
    report_path = OUT / "20_heldout_report.json"
    paths = [sheet_path, key_path, parquet_path, report_path]
    existing = [path for path in paths if path.exists()]
    if existing and not args.force:
        raise SystemExit(f"refusing to overwrite: {', '.join(map(str, existing))}")

    n = args.n or required_sample_size(args.half_width, args.confidence)
    source = load_v2_source()
    excluded_ids, exclusion_counts = load_excluded_ids()
    sample = sample_heldout(source, n=n, excluded_ids=excluded_ids, seed=args.seed)

    sample[["post_id", "text"]].assign(human_label="").to_csv(
        sheet_path, index=False
    )
    key_columns = [
        "post_id",
        "pool",
        "label_gemini",
        "flags_gemini",
        "label_cursor",
        "flags_cursor",
    ]
    sample[key_columns].to_csv(key_path, index=False)
    sample[["post_id", "text", "pool"]].rename(
        columns={"pool": "stratum"}
    ).to_parquet(parquet_path, index=False)

    report = {
        "n": n,
        "confidence": args.confidence,
        "target_half_width": args.half_width,
        "worst_case_wilson_half_width": round(
            wilson_half_width(n, args.confidence), 4
        ),
        "excluded": exclusion_counts,
        "pools": {
            str(name): int(count)
            for name, count in sample["pool"].value_counts().items()
        },
        "seed": args.seed,
        "human_labels_complete": False,
        "v3_labels_complete": False,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {sheet_path} ({n} rows)")
    print(f"hidden key: {key_path}")
    print(f"v3 labelling input: {parquet_path}")
    print(f"pool counts: {report['pools']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make")
    make.add_argument("--n", type=int)
    make.add_argument("--half-width", type=float, default=DEFAULT_HALF_WIDTH)
    make.add_argument("--confidence", type=float, default=0.95)
    make.add_argument("--seed", type=int, default=SEED + 1)
    make.add_argument("--force", action="store_true")
    make.set_defaults(func=cmd_make)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

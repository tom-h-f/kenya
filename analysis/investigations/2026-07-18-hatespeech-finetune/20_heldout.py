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
import ast
import json
import math
from pathlib import Path
from statistics import NormalDist

import pandas as pd

from _common import LABELS, OUT, SEED

POOL_FRACTIONS = {
    "split_primary_soft": 0.30,
    "split_primary_hard": 0.10,
    "agree_hate": 0.15,
    "agree_offensive": 0.15,
    "random_remaining": 0.30,
}
DEFAULT_HALF_WIDTH = 0.10
FLAGS = (
    "dehumanisation",
    "violence_call",
    "ethnic_targeting",
    "coded_language",
)
GATE_CLASS_AGREEMENT = 0.85
GATE_HATE_PRECISION = 0.70
GATE_HATE_RECALL = 0.70


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


def binary_metrics(human: pd.Series, predicted: pd.Series) -> dict:
    human = human.astype(bool)
    predicted = predicted.astype(bool)
    true_positive = int((human & predicted).sum())
    false_positive = int((~human & predicted).sum())
    false_negative = int((human & ~predicted).sum())
    support = int(human.sum())
    predicted_positive = int(predicted.sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = true_positive / support if support else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "support": support,
        "predicted": predicted_positive,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_agreement": round(float((human == predicted).mean()), 4),
    }


def class_metrics(human: pd.Series, predicted: pd.Series) -> dict:
    per_class = {}
    for label in LABELS:
        metrics = binary_metrics(human == label, predicted == label)
        per_class[label] = {
            key: value
            for key, value in metrics.items()
            if key in {"support", "predicted", "precision", "recall", "f1"}
        }
    confusion = pd.crosstab(predicted, human).reindex(
        index=LABELS, columns=LABELS, fill_value=0
    )
    return {
        "exact_agreement": round(float((human == predicted).mean()), 4),
        "macro_f1": round(
            sum(metrics["f1"] for metrics in per_class.values()) / len(LABELS), 4
        ),
        "hate": per_class["hate"],
        "per_class": per_class,
        "confusion": {
            str(model): {str(reference): int(value) for reference, value in row.items()}
            for model, row in confusion.to_dict(orient="index").items()
        },
    }


def flag_set(value) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return set()
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, list):
            return {str(item) for item in parsed}
    raise ValueError(f"invalid flags value: {value!r}")


def human_flag_series(df: pd.DataFrame, flag: str) -> pd.Series:
    return (
        df[f"human_{flag}"].astype(str).str.strip().str.lower().eq("true")
    )


def score_labeller(df: pd.DataFrame, suffix: str) -> dict:
    classes = class_metrics(df["human_label"], df[f"label_{suffix}"])
    predicted_flags = df[f"flags_{suffix}"].map(flag_set)
    flags = {
        flag: binary_metrics(
            human_flag_series(df, flag),
            predicted_flags.map(lambda values: flag in values),
        )
        for flag in FLAGS
    }
    by_pool = {
        str(pool): {
            "n": int(len(group)),
            "class_agreement": round(
                float((group["human_label"] == group[f"label_{suffix}"]).mean()), 4
            ),
        }
        for pool, group in df.groupby("pool")
    }
    return {
        "classes": classes,
        "flags": flags,
        "by_pool": by_pool,
        "gates": {
            "class_agreement": bool(
                classes["exact_agreement"] >= GATE_CLASS_AGREEMENT
            ),
            "hate_precision": bool(
                classes["hate"]["precision"] >= GATE_HATE_PRECISION
            ),
            "hate_recall": bool(classes["hate"]["recall"] >= GATE_HATE_RECALL),
        },
    }


def select_key_columns(key: pd.DataFrame) -> pd.DataFrame:
    return key[
        [
            "post_id",
            "pool",
            "label_gemini",
            "flags_gemini",
            "label_cursor",
            "flags_cursor",
        ]
    ].copy()


def reference_metadata(provenance: str) -> dict:
    if provenance == "independent-human":
        return {
            "kind": "human",
            "prelabel_source": None,
            "independent": True,
            "blind": True,
            "caveat": None,
        }
    if provenance == "opus-assisted-human":
        return {
            "kind": "human_validated",
            "prelabel_source": "claude-opus-code",
            "independent": False,
            "blind": False,
            "caveat": (
                "Human validation followed Opus prelabelling; agreement may be "
                "anchoring-biased."
            ),
        }
    raise ValueError(f"unknown reference provenance: {provenance}")


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


def cmd_score(args: argparse.Namespace) -> None:
    human = pd.read_csv(args.human, dtype={"post_id": str})
    if human["human_label"].isna().any() or not human["human_label"].isin(LABELS).all():
        raise SystemExit("human held-out labels are incomplete or invalid")
    key_path = OUT / "heldout_v3_key.csv"
    key = select_key_columns(pd.read_csv(key_path, dtype={"post_id": str}))
    key.to_csv(key_path, index=False)
    df = human.merge(key, on="post_id", validate="one_to_one")

    for short, directory in {
        "v3_gemini": OUT / "labels" / "heldout-v3" / "gemini-3.1-pro",
        "v3_cursor": OUT / "labels" / "heldout-v3" / "cursor-sonnet-4.5",
    }.items():
        labels = read_jsonl_files(directory)[["post_id", "label", "flags"]].rename(
            columns={"label": f"label_{short}", "flags": f"flags_{short}"}
        )
        df = df.merge(labels, on="post_id", validate="one_to_one")

    scores = {
        name: score_labeller(df, name)
        for name in ["gemini", "cursor", "v3_gemini", "v3_cursor"]
    }
    v3_agree = df["label_v3_gemini"] == df["label_v3_cursor"]
    agreed = df[v3_agree]
    interlabeller = {
        "class_agreement": round(float(v3_agree.mean()), 4),
        "agreed_n": int(v3_agree.sum()),
        "agreed_human_accuracy": round(
            float((agreed["label_v3_gemini"] == agreed["human_label"]).mean()), 4
        ) if len(agreed) else None,
    }

    flag_support_target = required_sample_size(half_width=0.20)
    flag_support = {
        flag: int(human_flag_series(df, flag).sum())
        for flag in FLAGS
    }
    flag_support_sufficient = {
        flag: support >= flag_support_target
        for flag, support in flag_support.items()
    }
    v3_gates_pass = {
        name: all(scores[name]["gates"].values())
        for name in ["v3_gemini", "v3_cursor"]
    }
    report_path = OUT / "20_heldout_report.json"
    report = json.loads(report_path.read_text())
    report.update({
        "human_labels_complete": True,
        "v3_labels_complete": True,
        "reference": reference_metadata(args.reference_provenance),
        "human_class_counts": {
            str(label): int(count)
            for label, count in df["human_label"].value_counts().items()
        },
        "human_flag_support": flag_support,
        "flag_support_gate": {
            "confidence": 0.95,
            "half_width": 0.20,
            "required_positive_support": flag_support_target,
            "pass": flag_support_sufficient,
        },
        "scores": scores,
        "v3_interlabeller": interlabeller,
        "decision": {
            "relabel_2440": (
                "approved" if all(v3_gates_pass.values()) else "not_approved"
            ),
            "flag_head_pilot": (
                "support_sufficient"
                if all(flag_support_sufficient.values())
                else "insufficient_human_positive_support"
            ),
            "v3_gate_pass": v3_gates_pass,
        },
    })
    report_path.write_text(json.dumps(report, indent=2))
    df.to_csv(OUT / "heldout_v3_scored.csv", index=False)

    print(f"=== held-out prompt comparison ({len(df)} rows) ===")
    for name, score in scores.items():
        classes = score["classes"]
        hate = classes["hate"]
        print(
            f"{name:10s} exact={classes['exact_agreement']:.3f} "
            f"macro-F1={classes['macro_f1']:.3f} "
            f"hate P/R/F1={hate['precision']:.3f}/{hate['recall']:.3f}/"
            f"{hate['f1']:.3f} gates={score['gates']}"
        )
    print(f"\nv3 interlabeller: {interlabeller}")
    print(f"human flag support: {flag_support}")
    print(f"DECISION: {report['decision']}")


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
    score = sub.add_parser("score")
    score.add_argument("--human", default=str(OUT / "heldout_v3_human.csv"))
    score.add_argument(
        "--reference-provenance",
        choices=["independent-human", "opus-assisted-human"],
        required=True,
    )
    score.set_defaults(func=cmd_score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

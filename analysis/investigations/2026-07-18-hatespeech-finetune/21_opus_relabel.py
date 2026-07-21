# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
#     "scikit-learn>=1.6",
# ]
# ///
"""Prepare and report Claude Opus 4.6 hate-speech relabelling runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from _common import OUT

CLASSES = ("neither", "offensive", "hate")
FLAGS = (
    "dehumanisation",
    "violence_call",
    "ethnic_targeting",
    "coded_language",
)
CONFIDENCE = ("high", "medium", "low")
LABEL_COLUMNS = ("post_id", "label", "flags", "confidence")


def require_columns(df: pd.DataFrame, columns: tuple[str, ...], context: str) -> None:
    missing = [column for column in columns if column not in df]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def validate_unique_ids(df: pd.DataFrame, context: str) -> None:
    require_columns(df, ("post_id",), context)
    if df["post_id"].isna().any():
        raise ValueError(f"{context}: missing post IDs")
    duplicates = sorted(
        df.loc[df["post_id"].duplicated(keep=False), "post_id"].astype(str).unique()
    )
    if duplicates:
        raise ValueError(f"{context}: duplicate post IDs: {duplicates}")


def validate_labels(series: pd.Series, context: str) -> None:
    unknown = sorted(set(series.dropna().astype(str)) - set(CLASSES))
    if series.isna().any() or unknown:
        values = unknown + (["<missing>"] if series.isna().any() else [])
        raise ValueError(f"{context}: unknown labels: {values}")


def _normalise_flags(value: Any, context: str) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context}: malformed flags: {value!r}; expected a list")
    if any(not isinstance(flag, str) for flag in value):
        raise ValueError(f"{context}: malformed flags: {value!r}")
    if len(value) != len(set(value)):
        raise ValueError(f"{context}: malformed flags: duplicate values in {value!r}")
    unknown = sorted(set(value) - set(FLAGS))
    if unknown:
        raise ValueError(f"{context}: malformed flags: unknown values {unknown}")
    return tuple(value)


def score_reference(
    df: pd.DataFrame, label_col: str, reference_col: str
) -> dict[str, Any]:
    """Score a label column against a reference with a fixed class universe."""
    require_columns(df, (label_col, reference_col), "reference scoring")
    validate_labels(df[label_col], label_col)
    validate_labels(df[reference_col], reference_col)
    reference = df[reference_col]
    predicted = df[label_col]
    precision, recall, hate_f1, _ = precision_recall_fscore_support(
        reference == "hate",
        predicted == "hate",
        average="binary",
        zero_division=0,
    )
    return {
        "exact_agreement": float((predicted == reference).mean()),
        "macro_f1": float(
            f1_score(
                reference,
                predicted,
                labels=list(CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "hate": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(hate_f1),
        },
    }


def label_movement(current: pd.Series, previous: pd.Series) -> dict[str, dict[str, int]]:
    """Return a complete previous-row/current-column 3x3 movement matrix."""
    if len(current) != len(previous):
        raise ValueError(
            f"label movement length mismatch: current={len(current)}, "
            f"previous={len(previous)}"
        )
    validate_labels(current, "current labels")
    validate_labels(previous, "previous labels")
    current = current.reset_index(drop=True)
    previous = previous.reset_index(drop=True)
    return {
        old: {
            new: int(((previous == old) & (current == new)).sum())
            for new in CLASSES
        }
        for old in CLASSES
    }


def flag_counts(flags: pd.Series) -> dict[str, int]:
    """Explode validated flag lists into stable, complete counts."""
    normalised = [
        _normalise_flags(value, f"flags row {index}")
        for index, value in flags.items()
    ]
    return {
        flag: int(sum(flag in values for values in normalised))
        for flag in FLAGS
    }


def _value_counts(series: pd.Series, order: tuple[str, ...] | None = None) -> dict:
    counts = series.value_counts()
    if order is None:
        return {str(key): int(counts[key]) for key in sorted(counts.index.astype(str))}
    return {value: int(counts.get(value, 0)) for value in order}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"cannot serialise {type(value).__name__}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )
        + "\n"
    )


def read_label_chunks(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("chunk_*.jsonl"))
    if not files:
        raise ValueError(f"no label chunks found in {directory}")
    rows: list[dict[str, Any]] = []
    for path in files:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: malformed JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: labelled row must be an object")
            rows.append(row)
    frame = pd.DataFrame(rows)
    require_columns(frame, LABEL_COLUMNS, str(directory))
    if frame.empty:
        raise ValueError(f"no labelled rows found in {directory}")
    if any(not isinstance(value, str) for value in frame["post_id"]):
        raise ValueError(f"{directory}: post IDs must be strings")
    validate_unique_ids(frame, str(directory))
    validate_labels(frame["label"], str(directory))
    frame["flags"] = [
        list(_normalise_flags(value, f"{directory} post_id={post_id}"))
        for post_id, value in zip(frame["post_id"], frame["flags"], strict=True)
    ]
    unknown_confidence = sorted(set(frame["confidence"]) - set(CONFIDENCE))
    if unknown_confidence:
        raise ValueError(
            f"{directory}: unknown confidence values: {unknown_confidence}"
        )
    return frame


def require_matching_ids(
    expected: pd.Series, actual: pd.Series, context: str
) -> None:
    expected_ids = set(expected.astype(str))
    actual_ids = set(actual.astype(str))
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ValueError(f"{context}: ID mismatch: missing={missing}, extra={extra}")


def load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        raise ValueError(f"run manifest missing: {path}")
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"run manifest must be an object: {path}")
    return manifest


def _read_reference(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source, dtype={"post_id": str})
    required = (
        "post_id",
        "text",
        "human_label",
        *(f"human_{flag}" for flag in FLAGS),
    )
    require_columns(frame, required, str(source))
    validate_unique_ids(frame, str(source))
    validate_labels(frame["human_label"], str(source))
    for column in (f"human_{flag}" for flag in FLAGS):
        frame[column] = frame[column].map(
            lambda value: _reference_bool(value, f"{source} column {column}")
        )
    return frame


def _reference_bool(value: Any, context: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{context}: expected true/false, got {value!r}")


def make_calibration(source: Path) -> Path:
    frame = _read_reference(source)
    human_columns = [column for column in frame if column.startswith("human_")]
    result = frame[["post_id", "text", *human_columns]].rename(
        columns={
            column: f"reference_{column.removeprefix('human_')}"
            for column in human_columns
        }
    )
    result.insert(2, "stratum", "calibration")
    destination = OUT / "opus_v4_calibration.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False)
    return destination


def _flag_metrics(
    reference_flags: pd.Series, predicted_flags: pd.Series
) -> dict[str, dict[str, Any]]:
    reference = [
        _normalise_flags(value, f"reference flags row {index}")
        for index, value in reference_flags.items()
    ]
    predicted = [
        _normalise_flags(value, f"predicted flags row {index}")
        for index, value in predicted_flags.items()
    ]
    result = {}
    for flag in FLAGS:
        expected = [flag in values for values in reference]
        actual = [flag in values for values in predicted]
        precision, recall, f1, _ = precision_recall_fscore_support(
            expected, actual, average="binary", zero_division=0
        )
        result[flag] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(sum(expected)),
        }
    return result


def score_calibration(tag: str, labeller: str, source: Path) -> tuple[Path, Path]:
    reference = _read_reference(source)
    reference["reference_flags"] = [
        [
            flag
            for flag in FLAGS
            if bool(row[f"human_{flag}"])
        ]
        for _, row in reference.iterrows()
    ]
    root = OUT / "labels" / tag
    labels = read_label_chunks(root / labeller)
    require_matching_ids(reference["post_id"], labels["post_id"], "calibration")
    joined = reference.merge(labels, on="post_id", validate="one_to_one")
    class_metrics = score_reference(joined, "label", "human_label")
    flag_metrics = _flag_metrics(joined["reference_flags"], joined["flags"])
    report = {
        "tag": tag,
        "labeller": labeller,
        "rows": len(joined),
        "classes": class_metrics,
        "class_counts": _value_counts(joined["label"], CLASSES),
        "flags": flag_metrics,
        "flag_counts": flag_counts(joined["flags"]),
        "confidence_counts": _value_counts(joined["confidence"], CONFIDENCE),
        "provenance": {
            "reference_source": str(source),
            "run_manifest": load_manifest(root),
        },
    }
    report_path = OUT / f"21_opus_calibration_{tag}.json"
    write_json(report_path, report)

    label_mismatch = joined["label"] != joined["human_label"]
    flag_mismatch = [
        set(reference_value) != set(predicted_value)
        for reference_value, predicted_value in zip(
            joined["reference_flags"], joined["flags"], strict=True
        )
    ]
    mismatches = joined[label_mismatch | pd.Series(flag_mismatch, index=joined.index)]
    mismatch_path = OUT / f"21_opus_calibration_{tag}_mismatches.csv"
    mismatches.to_csv(mismatch_path, index=False)
    return report_path, mismatch_path


def _baseline_provenance(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    names = (
        "prompt_version",
        "label_source",
        "source",
        "model",
        "labeller",
        "tag",
    )
    columns = {}
    for column in frame:
        if column in names or any(column.startswith(f"{name}_") for name in names):
            values = sorted(str(value) for value in frame[column].dropna().unique())
            columns[column] = values
    return {"path": str(path), "columns": columns}


def _confidence_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    columns = [
        column
        for column in frame
        if column == "confidence"
        or column.startswith("confidence_")
        or column.startswith("conf_")
    ]
    return {
        column: _value_counts(frame[column].dropna())
        for column in sorted(columns)
    }


def compare_full(tag: str, labeller: str, baseline_path: Path) -> tuple[Path, Path]:
    baseline = pd.read_parquet(baseline_path)
    require_columns(baseline, ("post_id", "label", "flags"), str(baseline_path))
    baseline["post_id"] = baseline["post_id"].astype(str)
    validate_unique_ids(baseline, str(baseline_path))
    validate_labels(baseline["label"], str(baseline_path))
    baseline["flags"] = [
        list(_normalise_flags(value, f"{baseline_path} post_id={post_id}"))
        for post_id, value in zip(
            baseline["post_id"], baseline["flags"], strict=True
        )
    ]

    root = OUT / "labels" / tag
    labels = read_label_chunks(root / labeller)
    require_matching_ids(baseline["post_id"], labels["post_id"], "full comparison")
    joined = baseline.merge(
        labels,
        on="post_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_previous", "_new"),
        sort=False,
    )
    if "stratum" not in joined:
        batch_path = root / "batch.parquet"
        if not batch_path.exists():
            raise ValueError(
                "full comparison: stratum missing from baseline and run batch"
            )
        batch = pd.read_parquet(batch_path)
        require_columns(batch, ("post_id", "stratum"), str(batch_path))
        batch["post_id"] = batch["post_id"].astype(str)
        validate_unique_ids(batch, str(batch_path))
        require_matching_ids(joined["post_id"], batch["post_id"], "run batch")
        joined = joined.merge(
            batch[["post_id", "stratum"]],
            on="post_id",
            validate="one_to_one",
        )

    joined["label_changed"] = joined["label_previous"] != joined["label_new"]
    joined["flags_changed"] = [
        set(previous) != set(current)
        for previous, current in zip(
            joined["flags_previous"], joined["flags_new"], strict=True
        )
    ]
    joined["changed"] = joined["label_changed"] | joined["flags_changed"]
    changed_rows = int(joined["changed"].sum())
    by_stratum = {
        str(stratum): {
            "n": int(len(group)),
            "rows": int(group["changed"].sum()),
            "rate": float(group["changed"].mean()),
            "label_rows": int(group["label_changed"].sum()),
            "flag_rows": int(group["flags_changed"].sum()),
        }
        for stratum, group in joined.groupby("stratum", sort=True, dropna=False)
    }
    report = {
        "tag": tag,
        "labeller": labeller,
        "rows": len(joined),
        "class_counts": {
            "previous": _value_counts(joined["label_previous"], CLASSES),
            "new": _value_counts(joined["label_new"], CLASSES),
        },
        "flag_counts": {
            "previous": flag_counts(joined["flags_previous"]),
            "new": flag_counts(joined["flags_new"]),
        },
        "confidence_counts": {
            "source": _confidence_counts(baseline),
            "new": _value_counts(joined["confidence"], CONFIDENCE),
        },
        "label_movement": label_movement(
            joined["label_new"], joined["label_previous"]
        ),
        "changed": {
            "rows": changed_rows,
            "rate": float(joined["changed"].mean()),
            "label_rows": int(joined["label_changed"].sum()),
            "label_rate": float(joined["label_changed"].mean()),
            "flag_rows": int(joined["flags_changed"].sum()),
            "flag_rate": float(joined["flags_changed"].mean()),
            "by_stratum": by_stratum,
        },
        "provenance": {
            "source": _baseline_provenance(baseline, baseline_path),
            "new": {"run_manifest": load_manifest(root)},
        },
    }
    report_path = OUT / f"21_opus_full_{tag}.json"
    write_json(report_path, report)

    changed = joined.loc[joined["changed"]].copy()
    changed = changed.rename(
        columns={
            "label_previous": "previous_label",
            "label_new": "new_label",
            "flags_previous": "previous_flags",
            "flags_new": "new_flags",
        }
    )
    changed_path = OUT / f"21_opus_full_{tag}_changed.csv"
    changed.to_csv(changed_path, index=False)
    return report_path, changed_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    make = commands.add_parser("make-calibration")
    make.add_argument(
        "--source",
        type=Path,
        default=OUT / "blind_check_coded_calibration.csv",
    )

    score = commands.add_parser("score-calibration")
    score.add_argument("--tag", required=True)
    score.add_argument("--labeller", default="claude-opus-4.6")
    score.add_argument(
        "--source",
        type=Path,
        default=OUT / "blind_check_coded_calibration.csv",
    )

    compare = commands.add_parser("compare-full")
    compare.add_argument("--tag", required=True)
    compare.add_argument("--labeller", default="claude-opus-4.6")
    compare.add_argument("--baseline", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "make-calibration":
        make_calibration(args.source)
    elif args.command == "score-calibration":
        score_calibration(args.tag, args.labeller, args.source)
    elif args.command == "compare-full":
        compare_full(args.tag, args.labeller, args.baseline)
    else:  # pragma: no cover - argparse enforces this
        raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()

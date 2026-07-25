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
import hashlib
import importlib.util
import json
import os
import re
import tempfile
from contextlib import contextmanager
from functools import lru_cache
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
SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@lru_cache(maxsize=1)
def producer_contract():
    """Load the numbered producer module as the validation source of truth."""
    path = Path(__file__).with_name("13_label_drive.py")
    spec = importlib.util.spec_from_file_location("_opus_label_drive_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load label producer contract from {path}")
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    return contract


def validate_path_component(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or value in {".", ".."}
        or Path(value).is_absolute()
        or SAFE_COMPONENT_RE.fullmatch(value) is None
    ):
        raise ValueError(
            f"unsafe {name} {value!r}: expected one alphanumeric path component "
            "containing only letters, digits, '.', '_' or '-'"
        )
    return value


def ensure_outputs_available(paths: tuple[Path, ...], force: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"output files already exist: {existing}; pass --force to replace "
            "the complete output set"
        )


@contextmanager
def atomic_target(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, content: str) -> None:
    with atomic_target(path) as temporary:
        temporary.write_text(content)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    with atomic_target(path) as temporary:
        frame.to_parquet(temporary, index=False)


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
    atomic_write_text(
        path,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )
        + "\n",
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_label_chunks(
    directory: Path,
    expected_ids: list[str] | None = None,
    *,
    chunk_size: int | None = None,
) -> pd.DataFrame:
    files = sorted(directory.glob("chunk_*.jsonl"))
    if not files:
        raise ValueError(f"no label chunks found in {directory}")
    if expected_ids is not None and chunk_size is not None:
        expected_names = [
            f"chunk_{offset // chunk_size:03d}.jsonl"
            for offset in range(0, len(expected_ids), chunk_size)
        ]
        actual_names = [path.name for path in files]
        if actual_names != expected_names:
            raise ValueError(
                f"{directory}: chunk file mismatch: "
                f"got={actual_names}, expected={expected_names}"
            )
    rows: list[dict[str, Any]] = []
    validated_chunks: list[dict[str, Any]] = []
    for file_index, path in enumerate(files):
        file_rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: malformed JSON: {exc.msg}"
                ) from exc
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: labelled row must be an object")
            post_id = row.get("post_id")
            confidence = row.get("confidence")
            if (
                "confidence" in row
                and (
                    not isinstance(confidence, str)
                    or confidence not in CONFIDENCE
                )
            ):
                raise ValueError(
                    f"{path}:{line_number} post_id={post_id!r}: "
                    f"bad confidence {confidence!r}"
                )
            rows.append(row)
            file_rows.append(row)
        if expected_ids is not None and chunk_size is not None:
            start = file_index * chunk_size
            wanted = expected_ids[start : start + chunk_size]
            try:
                validated_chunks.extend(
                    producer_contract().parse_response(
                        "\n".join(
                            json.dumps(row, ensure_ascii=False) for row in file_rows
                        ),
                        wanted,
                        allow_warnings=True,
                    )
                )
            except ValueError as exc:
                raise ValueError(f"{path}: invalid labelled output: {exc}") from exc
    if not rows:
        raise ValueError(f"no labelled rows found in {directory}")
    if expected_ids is not None and chunk_size is not None:
        validated = validated_chunks
    else:
        got_ids = [row.get("post_id") for row in rows]
        wanted_ids = expected_ids if expected_ids is not None else got_ids
        try:
            validated = producer_contract().parse_response(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                wanted_ids,
                allow_warnings=True,
            )
        except ValueError as exc:
            raise ValueError(f"{directory}: invalid labelled output: {exc}") from exc
    frame = pd.DataFrame(validated)
    validate_unique_ids(frame, str(directory))
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


def file_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def load_validated_run(
    tag: str, labeller: str
) -> tuple[Path, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    tag = validate_path_component(tag, "tag")
    labeller = validate_path_component(labeller, "labeller")
    root = OUT / "labels" / tag
    manifest = load_manifest(root)
    required_manifest = {
        "tag",
        "created_at",
        "input_path",
        "input_name",
        "rows",
        "input_sha256",
        "prompt_filename",
        "prompt_version",
        "prompt_sha256",
        "labellers",
        "chunk_size",
        "concurrency",
    }
    missing = sorted(required_manifest - manifest.keys())
    if missing:
        raise ValueError(f"run manifest missing required fields: {missing}")
    if manifest["tag"] != tag:
        raise ValueError(
            f"manifest tag mismatch: got {manifest['tag']!r}, expected {tag!r}"
        )
    if Path(str(manifest["input_path"])).name != manifest["input_name"]:
        raise ValueError(
            "manifest input identity mismatch: input_path basename does not "
            "match input_name"
        )
    labellers = manifest["labellers"]
    if not isinstance(labellers, dict) or labeller not in labellers:
        raise ValueError(f"requested labeller missing from manifest: {labeller!r}")
    configured = producer_contract().LABELLERS.get(labeller)
    if configured is not None:
        expected_mapping = {"cli": configured[0], "model": configured[1]}
        if labellers[labeller] != expected_mapping:
            raise ValueError(
                f"labeller mapping mismatch for {labeller!r}: "
                f"got={labellers[labeller]!r}, expected={expected_mapping!r}"
            )

    batch_path = root / "batch.parquet"
    if not batch_path.exists():
        raise ValueError(f"run batch missing: {batch_path}")
    batch = pd.read_parquet(batch_path)
    require_columns(batch, ("post_id", "text", "stratum"), str(batch_path))
    if any(not isinstance(value, str) for value in batch["post_id"]):
        raise ValueError(f"{batch_path}: post IDs must be strings")
    if any(not isinstance(value, str) for value in batch["text"]):
        raise ValueError(f"{batch_path}: text values must be strings")
    validate_unique_ids(batch, str(batch_path))
    if (
        not isinstance(manifest["rows"], int)
        or isinstance(manifest["rows"], bool)
        or manifest["rows"] != len(batch)
    ):
        raise ValueError(
            f"manifest rows mismatch: got {manifest['rows']!r}, "
            f"batch has {len(batch)}"
        )
    actual_hash = producer_contract().batch_sha256(batch)
    if manifest["input_sha256"] != actual_hash:
        raise ValueError(
            "manifest batch hash mismatch: "
            f"got {manifest['input_sha256']!r}, expected {actual_hash!r}"
        )
    chunk_size = manifest["chunk_size"]
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError(f"manifest has invalid chunk_size: {chunk_size!r}")
    if chunk_size != producer_contract().CHUNK_SIZE:
        raise ValueError(
            f"manifest chunk_size mismatch: got {chunk_size}, "
            f"expected {producer_contract().CHUNK_SIZE}"
        )
    labels = read_label_chunks(
        root / labeller,
        batch["post_id"].tolist(),
        chunk_size=chunk_size,
    )
    return root, manifest, batch, labels


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


def validate_reference_matches_batch(
    reference: pd.DataFrame, batch: pd.DataFrame
) -> None:
    """Ensure the score-time CSV still describes the same posts as the run."""
    require_matching_ids(
        batch["post_id"], reference["post_id"], "calibration source"
    )
    comparison = batch[["post_id", "text"]].merge(
        reference[["post_id", "text"]],
        on="post_id",
        validate="one_to_one",
        suffixes=("_batch", "_source"),
    )
    changed = comparison.loc[
        comparison["text_batch"] != comparison["text_source"], "post_id"
    ].tolist()
    if changed:
        raise ValueError(f"calibration source text mismatch for post IDs: {changed}")


def validate_baseline_matches_batch(
    baseline: pd.DataFrame, batch: pd.DataFrame
) -> None:
    """Ensure the prior labels describe the same posts as the Opus run batch."""
    require_columns(baseline, ("post_id", "text", "stratum"), "baseline")
    require_matching_ids(batch["post_id"], baseline["post_id"], "full comparison")
    comparison = batch[["post_id", "text", "stratum"]].merge(
        baseline[["post_id", "text", "stratum"]],
        on="post_id",
        validate="one_to_one",
        suffixes=("_batch", "_baseline"),
    )
    text_changed = comparison.loc[
        comparison["text_batch"] != comparison["text_baseline"], "post_id"
    ].tolist()
    if text_changed:
        raise ValueError(f"baseline text mismatch for post IDs: {text_changed}")
    stratum_changed = comparison.loc[
        ~comparison["stratum_batch"].eq(comparison["stratum_baseline"])
        & ~(
            comparison["stratum_batch"].isna()
            & comparison["stratum_baseline"].isna()
        ),
        "post_id",
    ].tolist()
    if stratum_changed:
        raise ValueError(f"baseline stratum mismatch for post IDs: {stratum_changed}")


def _reference_bool(value: Any, context: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{context}: expected true/false, got {value!r}")


def dataframe_csv(frame: pd.DataFrame) -> str:
    serialised = frame.copy()
    for column in serialised:
        if serialised[column].map(
            lambda value: isinstance(value, (list, tuple, np.ndarray, dict))
        ).any():
            serialised[column] = serialised[column].map(
                lambda value: json.dumps(
                    value.tolist() if isinstance(value, np.ndarray) else value,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if isinstance(value, (list, tuple, np.ndarray, dict))
                else value
            )
    return serialised.to_csv(index=False)


def make_calibration(source: Path, *, force: bool = False) -> Path:
    """Write a driver-compatible parquet with post text only.

    Human labels stay in the calibration CSV and are used only at score time,
    so correcting the reference after a blind model run remains legitimate.
    """
    destination = OUT / "opus_v4_calibration.parquet"
    ensure_outputs_available((destination,), force)
    frame = _read_reference(source)
    result = frame[["post_id", "text"]].copy()
    result.insert(2, "stratum", "calibration")
    atomic_write_parquet(result, destination)
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


def score_calibration(
    tag: str,
    labeller: str,
    source: Path,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    tag = validate_path_component(tag, "tag")
    labeller = validate_path_component(labeller, "labeller")
    report_path = OUT / f"21_opus_calibration_{tag}.json"
    mismatch_path = OUT / f"21_opus_calibration_{tag}_mismatches.csv"
    ensure_outputs_available((report_path, mismatch_path), force)
    reference = _read_reference(source)
    reference["reference_flags"] = [
        [
            flag
            for flag in FLAGS
            if bool(row[f"human_{flag}"])
        ]
        for _, row in reference.iterrows()
    ]
    root, manifest, batch, labels = load_validated_run(tag, labeller)
    validate_reference_matches_batch(reference, batch)
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
            "reference_source": file_fingerprint(source),
            "run_batch": file_fingerprint(root / "batch.parquet"),
            "run_manifest": manifest,
        },
    }
    label_mismatch = joined["label"] != joined["human_label"]
    flag_mismatch = [
        set(reference_value) != set(predicted_value)
        for reference_value, predicted_value in zip(
            joined["reference_flags"], joined["flags"], strict=True
        )
    ]
    mismatches = joined[label_mismatch | pd.Series(flag_mismatch, index=joined.index)]
    mismatch_csv = dataframe_csv(mismatches)
    write_json(report_path, report)
    atomic_write_text(mismatch_path, mismatch_csv)
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
    return {**file_fingerprint(path), "columns": columns}


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


def _stratum_key(value: Any) -> str:
    if value is None or (
        not isinstance(value, (list, tuple, dict, np.ndarray))
        and bool(pd.isna(value))
    ):
        return "<null>"
    if isinstance(value, str):
        return (
            f"<string:{json.dumps(value, ensure_ascii=False)}>"
            if value.startswith("<")
            else value
        )
    return f"<{type(value).__name__}:{value}>"


def compare_full(
    tag: str,
    labeller: str,
    baseline_path: Path,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    tag = validate_path_component(tag, "tag")
    labeller = validate_path_component(labeller, "labeller")
    report_path = OUT / f"21_opus_full_{tag}.json"
    changed_path = OUT / f"21_opus_full_{tag}_changed.csv"
    ensure_outputs_available((report_path, changed_path), force)
    baseline = pd.read_parquet(baseline_path)
    require_columns(
        baseline, ("post_id", "text", "stratum", "label", "flags"), str(baseline_path)
    )
    baseline["post_id"] = baseline["post_id"].astype(str)
    validate_unique_ids(baseline, str(baseline_path))
    validate_labels(baseline["label"], str(baseline_path))
    baseline["flags"] = [
        list(_normalise_flags(value, f"{baseline_path} post_id={post_id}"))
        for post_id, value in zip(
            baseline["post_id"], baseline["flags"], strict=True
        )
    ]

    root, manifest, batch, labels = load_validated_run(tag, labeller)
    validate_baseline_matches_batch(baseline, batch)
    require_matching_ids(baseline["post_id"], labels["post_id"], "full comparison")
    joined = baseline.merge(
        labels,
        on="post_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_previous", "_new"),
        sort=False,
    )
    # Prefer the validated run-batch stratum so reporting is not aliased to a
    # baseline column rename after the merge.
    joined = joined.drop(columns=["stratum"], errors="ignore").merge(
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
    joined["_stratum_key"] = joined["stratum"].map(_stratum_key)
    new_confidence_column = (
        "confidence_new" if "confidence_new" in joined else "confidence"
    )
    by_stratum = {
        stratum: {
            "n": int(len(group)),
            "rows": int(group["changed"].sum()),
            "rate": float(group["changed"].mean()),
            "label_rows": int(group["label_changed"].sum()),
            "flag_rows": int(group["flags_changed"].sum()),
        }
        for stratum, group in joined.groupby("_stratum_key", sort=True)
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
            "new": _value_counts(joined[new_confidence_column], CONFIDENCE),
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
            "new": {
                "run_batch": file_fingerprint(root / "batch.parquet"),
                "run_manifest": manifest,
            },
        },
    }
    changed = joined.loc[joined["changed"]].drop(columns="_stratum_key").copy()
    changed = changed.rename(
        columns={
            "label_previous": "previous_label",
            "label_new": "new_label",
            "flags_previous": "previous_flags",
            "flags_new": "new_flags",
        }
    )
    changed_csv = dataframe_csv(changed)
    write_json(report_path, report)
    atomic_write_text(changed_path, changed_csv)
    return report_path, changed_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    make = commands.add_parser(
        "make-calibration",
        help=(
            "Write post_id/text/stratum parquet for labelling. Human labels "
            "remain in the source CSV and are used only by score-calibration."
        ),
    )
    make.add_argument(
        "--source",
        type=Path,
        default=OUT / "blind_check_coded_calibration.csv",
    )
    make.add_argument("--force", action="store_true")

    score = commands.add_parser("score-calibration")
    score.add_argument("--tag", required=True)
    score.add_argument("--labeller", default="claude-opus-4.6")
    score.add_argument(
        "--source",
        type=Path,
        default=OUT / "blind_check_coded_calibration.csv",
    )
    score.add_argument("--force", action="store_true")

    compare = commands.add_parser("compare-full")
    compare.add_argument("--tag", required=True)
    compare.add_argument("--labeller", default="claude-opus-4.6")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if hasattr(args, "tag"):
        validate_path_component(args.tag, "tag")
    if hasattr(args, "labeller"):
        validate_path_component(args.labeller, "labeller")
    if args.command == "make-calibration":
        make_calibration(args.source, force=args.force)
    elif args.command == "score-calibration":
        score_calibration(
            args.tag,
            args.labeller,
            args.source,
            force=args.force,
        )
    elif args.command == "compare-full":
        compare_full(
            args.tag,
            args.labeller,
            args.baseline,
            force=args.force,
        )
    else:  # pragma: no cover - argparse enforces this
        raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()

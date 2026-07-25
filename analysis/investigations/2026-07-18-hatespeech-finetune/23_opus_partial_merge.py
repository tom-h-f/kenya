# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
# ]
# ///
"""Merge automated + manual Opus v4 labels into a partial training parquet.

Does not require the full 2,440-row batch or complete chunk files.

    uv run 23_opus_partial_merge.py
    uv run 23_opus_partial_merge.py --force
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from _common import OUT

TAG = "opus-v4-full"
LABELLER = "claude-opus-code"
CLASSES = ("neither", "offensive", "hate")
FLAGS = (
    "dehumanisation",
    "violence_call",
    "ethnic_targeting",
    "coded_language",
)
DEFAULT_BASELINE = Path(
    "/Users/tom/Code/Misc/kenya-monitor-2027"
    "/analysis/investigations/2026-07-18-hatespeech-finetune"
    "/out/labels_2026_full_final.parquet"
)


def _load_jsonl_dir(directory: Path, pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob(pattern)):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_labels(automated_dir: Path, manual_dir: Path) -> pd.DataFrame:
    """Union automated chunk labels and manual outbox labels."""
    automated = _load_jsonl_dir(automated_dir, "chunk_*.jsonl")
    manual = _load_jsonl_dir(manual_dir, "batch_*.jsonl")
    by_id: dict[str, dict[str, Any]] = {}
    origins: dict[str, str] = {}

    for origin, rows in (("automated", automated), ("manual", manual)):
        for row in rows:
            post_id = str(row["post_id"])
            label = row.get("label")
            flags = list(row.get("flags") or [])
            if post_id in by_id:
                previous = by_id[post_id]
                if previous.get("label") != label or list(previous.get("flags") or []) != flags:
                    raise ValueError(
                        f"conflicting labels for post_id={post_id}: "
                        f"{origins[post_id]}={previous.get('label')}/{previous.get('flags')} "
                        f"vs {origin}={label}/{flags}"
                    )
                continue
            cleaned = {
                "post_id": post_id,
                "label": label,
                "flags": flags,
                "target_group": row.get("target_group"),
                "confidence": row.get("confidence"),
                "rationale": row.get("rationale"),
                "warnings": list(row.get("warnings") or []),
                "label_origin": origin,
            }
            by_id[post_id] = cleaned
            origins[post_id] = origin

    if not by_id:
        raise ValueError("no Opus labels found in automated or manual directories")
    return pd.DataFrame(by_id.values())


def _value_counts(series: pd.Series, order: tuple[str, ...] | None = None) -> dict[str, int]:
    counts = series.value_counts()
    if order is None:
        return {str(key): int(counts[key]) for key in counts.index}
    return {value: int(counts.get(value, 0)) for value in order}


def _flag_counts(flags: pd.Series) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in flags:
        counter.update(list(value or []))
    return {flag: int(counter.get(flag, 0)) for flag in FLAGS}


def _label_movement(current: pd.Series, previous: pd.Series) -> dict[str, dict[str, int]]:
    current = current.reset_index(drop=True)
    previous = previous.reset_index(drop=True)
    return {
        old: {
            new: int(((previous == old) & (current == new)).sum())
            for new in CLASSES
        }
        for old in CLASSES
    }


def merge_partial(
    *,
    batch: pd.DataFrame,
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    prompt_version: str,
    prompt_sha256: str,
    labeller: str,
    model: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    batch = batch.copy()
    batch["post_id"] = batch["post_id"].astype(str)
    labels = labels.copy()
    labels["post_id"] = labels["post_id"].astype(str)
    baseline = baseline.copy()
    baseline["post_id"] = baseline["post_id"].astype(str)

    unknown = sorted(set(labels["post_id"]) - set(batch["post_id"]))
    if unknown:
        raise ValueError(f"labels reference unknown batch IDs: {unknown[:5]}")

    keep_batch = [
        column
        for column in (
            "post_id",
            "author_handle",
            "created_at",
            "text",
            "stratum",
            "p_hate",
            "p_offensive",
            "is_lexicon",
            "is_nli_tail",
        )
        if column in batch.columns
    ]
    merged = batch[keep_batch].merge(labels, on="post_id", how="inner", validate="one_to_one")
    if "split" in baseline.columns:
        merged = merged.merge(
            baseline[["post_id", "split"]],
            on="post_id",
            how="left",
            validate="one_to_one",
        )
    merged["label_claude"] = merged["label"]
    merged["conf_claude"] = merged["confidence"]
    merged["rationale_claude"] = merged["rationale"]
    merged["flags_claude"] = merged["flags"].map(list)
    merged["target_claude"] = merged["target_group"]
    merged["agreed"] = True
    merged["label_source"] = "single_labeller_claude"
    merged["prompt_version"] = prompt_version

    overlap = merged.merge(
        baseline[["post_id", "label"]].rename(columns={"label": "label_previous"}),
        on="post_id",
        how="inner",
        validate="one_to_one",
    )
    changed = int((overlap["label"] != overlap["label_previous"]).sum())
    complete = len(merged) >= 2440
    report = {
        "dataset": (
            f"opus-v4-full-{len(merged)}" if complete else f"opus-v4-partial-{len(merged)}"
        ),
        "decision": (
            "use_full_2440" if complete else "use_partial_not_full_2440"
        ),
        "rows": len(merged),
        "labeller": labeller,
        "model": model,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "label_origin_counts": _value_counts(merged["label_origin"]),
        "class_counts": _value_counts(merged["label"], CLASSES),
        "stratum_counts": _value_counts(merged["stratum"]) if "stratum" in merged else {},
        "confidence_counts": _value_counts(merged["confidence"]),
        "flag_counts": _flag_counts(merged["flags"]),
        "prior_split_counts": (
            _value_counts(merged["split"].dropna()) if "split" in merged else {}
        ),
        "vs_baseline_overlap": {
            "n": len(overlap),
            "label_changed": changed,
            "label_change_rate": round(changed / len(overlap), 4) if len(overlap) else None,
            "movement": _label_movement(overlap["label"], overlap["label_previous"]),
        },
        "reliability_caveat": (
            "Single labeller (Claude Opus via Claude Code). Operator treated "
            "Opus v4 labels as human-validated for training."
        ),
        "artifacts": {
            "labels": (
                "out/labels_2026_opus-v4-full.parquet"
                if complete
                else "out/labels_2026_opus-v4-partial.parquet"
            ),
            "report": f"out/21_opus_partial_{len(merged)}_report.json",
        },
    }
    return merged, report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--labeller", default=LABELLER)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    root = OUT / "labels" / args.tag
    automated_dir = root / args.labeller
    manual_dir = OUT / "opus_v4_full_manual" / "outbox"
    batch_path = root / "batch.parquet"
    manifest_path = root / "manifest.json"
    if not batch_path.exists():
        raise SystemExit(f"missing batch: {batch_path}")
    if not args.baseline.exists():
        raise SystemExit(f"missing baseline: {args.baseline}")

    labels = collect_labels(automated_dir, manual_dir)
    batch = pd.read_parquet(batch_path)
    baseline = pd.read_parquet(args.baseline)
    prompt_version = "v4"
    prompt_sha256 = ""
    model = "opus"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        prompt_version = str(manifest.get("prompt_version") or prompt_version)
        prompt_sha256 = str(manifest.get("prompt_sha256") or "")
        labellers = manifest.get("labellers") or {}
        if args.labeller in labellers:
            model = str(labellers[args.labeller].get("model") or model)

    merged, report = merge_partial(
        batch=batch,
        labels=labels,
        baseline=baseline,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
        labeller=args.labeller,
        model=model,
    )

    labels_path = OUT / "labels_2026_opus-v4-partial.parquet"
    alias_path = OUT / "labels_2026_opus-v4-full.parquet"
    report_path = OUT / f"21_opus_partial_{len(merged)}_report.json"
    for path in (labels_path, alias_path, report_path):
        if path.exists() and not args.force:
            raise SystemExit(f"{path} exists; pass --force to replace")

    merged.to_parquet(labels_path, index=False)
    merged.to_parquet(alias_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

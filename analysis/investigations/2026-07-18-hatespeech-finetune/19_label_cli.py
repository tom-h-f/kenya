# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2",
#     "deep-translator",
# ]
# ///

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import pandas as pd

from _common import LABELS, OUT

SHEET_PATH = OUT / "blind_check_coded.csv"
CALIBRATION_V1_PATH = OUT / "blind_check_coded_calibration_v1.csv"
CALIBRATION_PATH = OUT / "blind_check_coded_calibration.csv"
FLAGS = (
    "dehumanisation",
    "violence_call",
    "ethnic_targeting",
    "coded_language",
)
STRUCTURED_COLUMNS = (
    *(f"human_{flag}" for flag in FLAGS),
    "human_confidence",
    "human_rationale",
    "translation_used",
)
RUBRIC = """\
PROTECTED-GROUP RUBRIC
  hate       attacks a protected group, or a person because of protected-group
             membership. The target must be identifiable from the post/context.
  offensive  abuse, threats, profanity, or degradation without such a target.
  neither    criticism, reporting, electoral arithmetic, counterspeech, or banter.

Boundary checks:
  "Ruto is a thief" -> offensive (individual abuse)
  "Kalenjins are thieves" -> hate (collective protected-group contempt)
  Generic or coded violence with no identifiable protected target is not hate;
  label offensive and set violence_call/coded_language as applicable.
  Quoted hate that the author condemns is neither; judge the author's stance.

Flags are independent of class:
  dehumanisation  people framed as vermin, disease, filth, demons, or non-human
  violence_call   threat, call, celebration, or approval of physical violence
  ethnic_targeting protected ethnic/tribal/religious/regional group is targeted
  coded_language  harmful meaning depends on euphemism, metaphor, or local code
"""


def prepare_recode_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add structured fields while preserving an existing first-pass label."""
    result = df.copy()
    if "human_label" not in result:
        result["human_label"] = ""
    result["human_label"] = result["human_label"].fillna("").astype(str).str.strip()

    if "human_label_v1" not in result:
        result["human_label_v1"] = result["human_label"]
        if result["human_label"].isin(LABELS).any():
            result["human_label"] = ""
    else:
        result["human_label_v1"] = (
            result["human_label_v1"].fillna("").astype(str).str.strip()
        )

    for column in STRUCTURED_COLUMNS:
        if column not in result:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str).str.strip()
    return result


def requires_recode_preparation(df: pd.DataFrame) -> bool:
    return (
        "human_label_v1" not in df
        and "human_label" in df
        and df["human_label"].fillna("").astype(str).str.strip().isin(LABELS).any()
    )


def write_recode_files(source: Path, archive: Path, output: Path) -> None:
    if archive.exists() or output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing recode files: {archive}, {output}"
        )
    original = pd.read_csv(source, dtype={"post_id": str})
    original.to_csv(archive, index=False)
    prepare_recode_frame(original).to_csv(output, index=False)


def record_annotation(
    df: pd.DataFrame,
    index: int,
    *,
    label: str,
    flags: set[str],
    confidence: str,
    rationale: str,
    translation_used: bool,
) -> None:
    if label not in LABELS:
        raise ValueError(f"invalid label: {label}")
    if not flags <= set(FLAGS):
        raise ValueError(f"invalid flags: {flags - set(FLAGS)}")
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(f"invalid confidence: {confidence}")
    if not rationale.strip():
        raise ValueError("rationale is required")

    df.at[index, "human_label"] = label
    for flag in FLAGS:
        df.at[index, f"human_{flag}"] = str(flag in flags).lower()
    df.at[index, "human_confidence"] = confidence
    df.at[index, "human_rationale"] = rationale.strip()
    df.at[index, "translation_used"] = str(translation_used).lower()


def completed_mask(df: pd.DataFrame) -> pd.Series:
    flags_complete = pd.Series(True, index=df.index)
    for flag in FLAGS:
        flags_complete &= df[f"human_{flag}"].isin(["true", "false"])
    return (
        df["human_label"].isin(LABELS)
        & df["human_confidence"].isin(["high", "medium", "low"])
        & df["human_rationale"].ne("")
        & flags_complete
    )


@functools.lru_cache(maxsize=1000)
def get_translation(text: str) -> str:
    try:
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        return f"[Translation failed: {e}]"


def parse_flags(value: str) -> set[str]:
    aliases = {"d": "dehumanisation", "v": "violence_call",
               "e": "ethnic_targeting", "c": "coded_language"}
    tokens = [token.strip().lower() for token in value.split(",") if token.strip()]
    flags = {aliases.get(token, token) for token in tokens}
    unknown = flags - set(FLAGS)
    if unknown:
        raise ValueError(f"unknown flags: {', '.join(sorted(unknown))}")
    return flags


def prompt_annotation(text: str) -> tuple[str, set[str], str, str, bool] | None:
    translation_used = False
    while True:
        choice = input(
            "Class: [1/n] neither | [2/o] offensive | [3/h] hate | "
            "[t] translate | [r] rubric | [b] back | [q] quit: "
        ).strip().lower()
        choices = {
            "1": "neither", "n": "neither", "neither": "neither",
            "2": "offensive", "o": "offensive", "offensive": "offensive",
            "3": "hate", "h": "hate", "hate": "hate",
        }
        if choice in choices:
            label = choices[choice]
            break
        if choice in {"b", "back"}:
            return None
        if choice in {"q", "quit", "exit"}:
            raise EOFError
        if choice == "r":
            print("\n" + RUBRIC)
            continue
        if choice == "t":
            print("\nMachine translation (use cautiously):")
            print(get_translation(text))
            translation_used = True
            continue
        print("Invalid class choice.")

    while True:
        try:
            flags = parse_flags(input(
                "Flags comma-separated [d]ehumanisation, [v]iolence, "
                "[e]thnic targeting, [c]oded; blank for none: "
            ))
            break
        except ValueError as error:
            print(error)

    confidence_choices = {
        "h": "high", "high": "high",
        "m": "medium", "medium": "medium",
        "l": "low", "low": "low",
    }
    while True:
        confidence = confidence_choices.get(
            input("Confidence [h]igh/[m]edium/[l]ow: ").strip().lower()
        )
        if confidence:
            break
        print("Invalid confidence.")

    while True:
        rationale = input("One-sentence rationale (quote the operative phrase): ").strip()
        if rationale:
            break
        print("Rationale is required.")
    return label, flags, confidence, rationale, translation_used


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrated human labelling CLI")
    parser.add_argument("--sheet", default=str(SHEET_PATH), help="Path to the CSV sheet")
    parser.add_argument(
        "--prepare-recode",
        action="store_true",
        help="preserve the completed sheet and create a blank calibrated copy",
    )
    parser.add_argument("--archive", default=str(CALIBRATION_V1_PATH))
    parser.add_argument("--output", default=str(CALIBRATION_PATH))
    args = parser.parse_args()

    sheet_path = Path(args.sheet)
    if not sheet_path.exists():
        sys.exit(f"Error: {sheet_path} does not exist. Run 18_blind_check.py make first.")
    if args.prepare_recode:
        try:
            write_recode_files(sheet_path, Path(args.archive), Path(args.output))
        except FileExistsError as error:
            sys.exit(str(error))
        print(f"preserved original labels: {args.archive}")
        print(f"calibrated working sheet: {args.output}")
        print(f"label with: uv run 19_label_cli.py --sheet {args.output}")
        return

    original = pd.read_csv(sheet_path, dtype={"post_id": str})
    if requires_recode_preparation(original):
        sys.exit(
            "completed legacy sheet detected; preserve it first with "
            "--prepare-recode, then label the generated calibration sheet"
        )
    df = prepare_recode_frame(original)
    if not df.equals(original):
        df.to_csv(sheet_path, index=False)

    total = len(df)
    labeled_mask = completed_mask(df)
    labeled_count = int(labeled_mask.sum())

    first_unlabeled = 0
    for idx, complete in enumerate(labeled_mask):
        if not complete:
            first_unlabeled = idx
            break
    else:
        first_unlabeled = total - 1 if total > 0 else 0

    print("=" * 60)
    print("Kenya Hate-Speech Labelling CLI")
    print(f"Total posts: {total}")
    print(f"Labeled so far: {labeled_count}/{total} ({labeled_count/total:.1%})")
    print("=" * 60)
    print(RUBRIC)

    try:
        start_input = input(f"Start index (0-{total-1}, default {first_unlabeled}): ").strip()
        if start_input:
            idx = int(start_input)
            if not (0 <= idx < total):
                raise ValueError
        else:
            idx = first_unlabeled
    except ValueError:
        print(f"Invalid input. Starting at default {first_unlabeled}.")
        idx = first_unlabeled

    while 0 <= idx < total:
        row = df.iloc[idx]
        post_id = row["post_id"]
        text = row["text"]
        current_label = row["human_label"]

        print("\n" + "=" * 60)
        print(f"Post {idx + 1} of {total}  (Index: {idx})  [ID: {post_id}]")
        print(f"Current Label: {current_label or '[none]'}")
        print("-" * 60)
        print("Original Text:")
        print(text)
        print("=" * 60)

        try:
            annotation = prompt_annotation(text)
        except EOFError:
            print(f"\nSaved progress. {int(completed_mask(df).sum())}/{total} complete.")
            return
        if annotation is None:
            idx = max(0, idx - 1)
            continue
        label, flags, confidence, rationale, translation_used = annotation
        record_annotation(
            df,
            idx,
            label=label,
            flags=flags,
            confidence=confidence,
            rationale=rationale,
            translation_used=translation_used,
        )
        df.to_csv(sheet_path, index=False)
        idx += 1

    print(f"\nFinished! All {total} rows labeled.")
    print("You can now run: uv run 18_blind_check.py score")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSession interrupted. Progress saved.")

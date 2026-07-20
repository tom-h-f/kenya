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
from deep_translator import GoogleTranslator
from _common import LABELS, OUT

SHEET_PATH = OUT / "blind_check_coded.csv"


@functools.lru_cache(maxsize=1000)
def get_translation(text: str) -> str:
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        return f"[Translation failed: {e}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Label CLI with translation")
    parser.add_argument("--sheet", default=str(SHEET_PATH), help="Path to the CSV sheet")
    args = parser.parse_args()

    sheet_path = Path(args.sheet)
    if not sheet_path.exists():
        sys.exit(f"Error: {sheet_path} does not exist. Run 18_blind_check.py make first.")

    df = pd.read_csv(sheet_path)
    
    if "human_label" not in df.columns:
        df["human_label"] = ""
    df["human_label"] = df["human_label"].fillna("").astype(str).str.strip()

    total = len(df)
    labeled_mask = df["human_label"].isin(LABELS)
    labeled_count = labeled_mask.sum()
    
    first_unlabeled = 0
    for idx, label in enumerate(df["human_label"]):
        if label not in LABELS:
            first_unlabeled = idx
            break
    else:
        first_unlabeled = total - 1 if total > 0 else 0

    print("=" * 60)
    print("Kenya Hate-Speech Labelling CLI")
    print(f"Total posts: {total}")
    print(f"Labeled so far: {labeled_count}/{total} ({labeled_count/total:.1%})")
    print("=" * 60)
    
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
        print("-" * 60)
        print("Translating Swahili/Sheng to English...")
        translation = get_translation(text)
        print("Translation:")
        print(translation)
        print("=" * 60)
        
        while True:
            prompt = "Rate: [1/n] neither | [2/o] offensive | [3/h] hate | [?] explain | [b] back | [q] quit: "
            choice = input(prompt).strip().lower()

            if choice in ("1", "n", "neither"):
                df.at[idx, "human_label"] = "neither"
                df.to_csv(sheet_path, index=False)
                idx += 1
                break
            elif choice in ("2", "o", "offensive"):
                df.at[idx, "human_label"] = "offensive"
                df.to_csv(sheet_path, index=False)
                idx += 1
                break
            elif choice in ("3", "h", "hate"):
                df.at[idx, "human_label"] = "hate"
                df.to_csv(sheet_path, index=False)
                idx += 1
                break
            elif choice == "?":
                print("\nQuerying agy for explanation...")
                explanation_prompt = (
                    "Explain what this Kenyan social media post means. "
                    "Translate and explain any Swahili, Sheng, or Kenyan political/ethnic terms, slang, or references:\n\n"
                    f"{text}"
                )
                try:
                    import subprocess
                    proc = subprocess.run(
                        ["agy", "-p", explanation_prompt],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    print("\nExplanation:")
                    print(proc.stdout.strip())
                    print("=" * 60)
                except Exception as e:
                    print(f"\nError running agy: {e}")
            elif choice in ("b", "back"):
                if idx > 0:
                    idx -= 1
                else:
                    print("Already at the first post.")
                break
            elif choice in ("q", "quit", "exit"):
                print(f"\nSaved progress. {df['human_label'].isin(LABELS).sum()}/{total} labeled.")
                return
            else:
                print("Invalid input. Please choose 1/n, 2/o, 3/h, ?, b, or q.")

    print(f"\nFinished! All {total} rows labeled.")
    print("You can now run: uv run 18_blind_check.py score")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSession interrupted. Progress saved.")

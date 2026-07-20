# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
# ]
# ///
"""LLM adjudication of class disagreements by blind third vote (Plan A4).

    uv run 15_adjudicate.py --tag full

The handoff design sends disagreements to Tom. He opted for model judgement
instead, so a third independent model (Claude Opus 4.6) labels the disputed
posts **blind** - it never sees the other two labellers' verdicts or
rationales. Anchoring it on their arguments would make it a referee of
rhetoric rather than an independent vote, and majority-of-three only means
anything if the third vote is genuinely independent.

Resolution: the label holding 2 of 3 votes wins. If all three differ, the
median severity (neither < offensive < hate) wins - always `offensive`.

Reported alongside: how often the adjudicator sides with each original
labeller. Opus shares a family with one of them, so a systematic tilt would
be a real bias in the resulting labels rather than a neutral tie-break, and
it must be visible rather than assumed away.
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path

import pandas as pd

from _common import OUT

drive = import_module("13_label_drive")

ADJUDICATOR = "claude-opus-4.6"
ADJUDICATOR_MODEL = "Claude Opus 4.6 (Thinking)"
SEVERITY = {"neither": 0, "offensive": 1, "hate": 2}


def resolve(votes: list[str]) -> tuple[str, str]:
    counts = pd.Series(votes).value_counts()
    if counts.iloc[0] >= 2:
        return counts.index[0], "majority"
    return sorted(votes, key=SEVERITY.get)[1], "median_severity"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompt-version", default="v2")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--print-timeout", default="10m")
    args = ap.parse_args()

    merged = pd.read_parquet(OUT / f"labels_2026_{args.tag}.parquet")
    disputed = merged[~merged["agreed"]].reset_index(drop=True)
    print(f"{len(disputed)} disputed rows of {len(merged)}")
    if disputed.empty:
        return

    root = OUT / "labels" / f"{args.tag}-adjudication"
    chunk_dir = OUT / "chunks" / f"{args.tag}-adjudication"
    prompt_path = Path(drive.PROMPT_DIR) / f"label_{args.prompt_version}.md"
    chunks = drive.write_chunks(disputed, chunk_dir)
    print(f"{len(chunks)} chunks -> {ADJUDICATOR_MODEL}")

    out_dir, failed_dir = root / ADJUDICATOR, root / "failed"
    todo = [c for c in chunks if not (out_dir / f"{c.stem}.jsonl").exists()]
    if todo:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(
                lambda c: drive.label_chunk(ADJUDICATOR, ADJUDICATOR_MODEL, c,
                                            out_dir, failed_dir,
                                            args.print_timeout, prompt_path),
                todo,
            ))

    third = drive_load(out_dir)
    have = disputed["post_id"].isin(third.index)
    print(f"adjudicated {int(have.sum())} of {len(disputed)}")

    short_a, short_b = "gemini", "claude"
    resolved = disputed[have].copy()
    resolved["label_opus"] = third.loc[resolved["post_id"], "label"].values
    resolved["rationale_opus"] = third.loc[resolved["post_id"], "rationale"].values
    outcome = [
        resolve([a, b, c])
        for a, b, c in zip(resolved[f"label_{short_a}"], resolved[f"label_{short_b}"],
                           resolved["label_opus"])
    ]
    resolved["label"] = [o[0] for o in outcome]
    resolved["resolution"] = [o[1] for o in outcome]
    resolved["label_source"] = "llm_adjudicated"

    report = {
        "tag": args.tag,
        "adjudicator": ADJUDICATOR_MODEL,
        "prompt_version": args.prompt_version,
        "disputed": len(disputed),
        "adjudicated": len(resolved),
        "resolution_kind": resolved["resolution"].value_counts().to_dict(),
        "resolved_labels": resolved["label"].value_counts().to_dict(),
        "sided_with_gemini": int(
            (resolved["label_opus"] == resolved[f"label_{short_a}"]).sum()
        ),
        "sided_with_claude": int(
            (resolved["label_opus"] == resolved[f"label_{short_b}"]).sum()
        ),
        "sided_with_neither": int(
            ((resolved["label_opus"] != resolved[f"label_{short_a}"]) &
             (resolved["label_opus"] != resolved[f"label_{short_b}"])).sum()
        ),
    }

    final = merged.copy()
    final = final.merge(
        resolved[["post_id", "label", "label_opus", "rationale_opus", "resolution",
                  "label_source"]],
        on="post_id", how="left", suffixes=("", "_adj"),
    )
    final["label"] = final["label"].fillna(final["label_adj"])
    final["label_source"] = final["label_source"].mask(
        final["label_adj"].notna(), "llm_adjudicated"
    )
    final = final.drop(columns=["label_adj"])

    report["final_labelled"] = int(final["label"].notna().sum())
    report["final_counts"] = final["label"].value_counts().to_dict()
    report["final_positives"] = int(final["label"].isin({"hate", "offensive"}).sum())

    final.to_parquet(OUT / f"labels_2026_{args.tag}_adjudicated.parquet", index=False)
    (OUT / f"15_adjudication_report_{args.tag}.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


def drive_load(out_dir: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for f in sorted(out_dir.glob("chunk_*.jsonl"))
        for line in f.read_text().splitlines()
        if line
    ]
    df = pd.DataFrame(rows)
    df["post_id"] = df["post_id"].astype(str)
    return df.drop_duplicates("post_id").set_index("post_id")


if __name__ == "__main__":
    main()

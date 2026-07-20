# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
#     "scikit-learn>=1.4",
# ]
# ///
"""Merge the two labellers, measure agreement, queue disagreements (Plan A4).

    uv run 14_label_merge.py --tag pilot
    uv run 14_label_merge.py --tag full --blind-check 100

Agreement on class -> accepted. Class disagreement -> adjudication_queue.csv
for Tom; disagreements are never resolved by rule (a tie-break heuristic
would silently manufacture the labels this dataset exists to establish).
Flags are OR'd across labellers, confidences and rationales both kept.

Positive rate is reported per stratum on accepted rows only, and the random
control stratum is the sole honest prevalence estimate - mined strata measure
the miner, not the corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from _common import OUT

DEFAULT_LABELLERS = ["gemini-3.1-pro", "claude-sonnet-4.6"]
POSITIVE = {"hate", "offensive"}


def load_labels(root: Path, labeller: str) -> pd.DataFrame:
    files = sorted((root / labeller).glob("chunk_*.jsonl"))
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines() if line]
    df = pd.DataFrame(rows)
    df["post_id"] = df["post_id"].astype(str)
    return df.drop_duplicates("post_id").set_index("post_id")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompt-version", default="v2")
    ap.add_argument("--labellers", default=",".join(DEFAULT_LABELLERS),
                    help="one name = single-labeller mode: no agreement metric")
    ap.add_argument("--blind-check", type=int, default=0,
                    help="write N randomly sampled agreed rows for human review")
    args = ap.parse_args()

    root = OUT / "labels" / args.tag
    batch = pd.read_parquet(root / "batch.parquet")
    batch["post_id"] = batch["post_id"].astype(str)

    names = args.labellers.split(",")
    sources = [load_labels(root, name) for name in names]
    labelled = batch
    for src in sources:
        labelled = labelled[labelled["post_id"].isin(src.index)]
    report: dict[str, object] = {
        "tag": args.tag,
        "prompt_version": args.prompt_version,
        "labellers": names,
        "single_labeller": len(names) == 1,
        "batch_rows": len(batch),
        "labelled_by_both": len(labelled),
        "missing": int(len(batch) - len(labelled)),
    }

    idx = labelled["post_id"]
    merged = labelled.copy()
    for name, src in zip(names, sources):
        short = name.split("-")[0]
        merged[f"label_{short}"] = src.loc[idx, "label"].values
        merged[f"conf_{short}"] = src.loc[idx, "confidence"].values
        merged[f"rationale_{short}"] = src.loc[idx, "rationale"].values
        merged[f"flags_{short}"] = [list(f) for f in src.loc[idx, "flags"].values]
        merged[f"target_{short}"] = src.loc[idx, "target_group"].values

    shorts = [n.split("-")[0] for n in names]
    if len(names) == 1:
        only = shorts[0]
        merged["agreed"] = True
        merged["label"] = merged[f"label_{only}"]
        merged["flags"] = merged[f"flags_{only}"]
        merged["label_source"] = f"single_labeller_{only}"
        merged["prompt_version"] = args.prompt_version
        report["agreement_rate"] = None
        report["kappa_overall"] = None
        report["reliability_caveat"] = (
            "Single labeller: no agreement or kappa can be computed and no "
            "row was adjudicated. Every label rests on one model's judgement. "
            "The pilot's kappa 0.665 was measured on a two-labeller design "
            "that was not used here and does not validate these labels."
        )
        report["label_counts"] = {only: merged["label"].value_counts().to_dict()}
    else:
        short_a, short_b = shorts
        merged["agreed"] = merged[f"label_{short_a}"] == merged[f"label_{short_b}"]
        merged["label"] = merged[f"label_{short_a}"].where(merged["agreed"])
        merged["flags"] = [
            sorted(set(x) | set(y))
            for x, y in zip(merged[f"flags_{short_a}"], merged[f"flags_{short_b}"])
        ]
        merged["label_source"] = merged["agreed"].map(
            {True: "both_agree", False: "pending_adjudication"}
        )
        merged["prompt_version"] = args.prompt_version

        y1, y2 = merged[f"label_{short_a}"], merged[f"label_{short_b}"]
        report["agreement_rate"] = round(float(merged["agreed"].mean()), 4)
        report["kappa_overall"] = round(float(cohen_kappa_score(y1, y2)), 4)
        report["kappa_per_class"] = {
            cls: round(float(cohen_kappa_score(y1 == cls, y2 == cls)), 4)
            for cls in ("hate", "offensive", "neither")
        }
        report["label_counts"] = {
            short_a: y1.value_counts().to_dict(),
            short_b: y2.value_counts().to_dict(),
        }
        report["confusion"] = (
            pd.crosstab(y1, y2).rename_axis(short_a).rename_axis(short_b, axis=1)
            .to_dict()
        )

    accepted = merged[merged["agreed"]]
    report["accepted"] = len(accepted)
    report["confirmed_positives"] = int(accepted["label"].isin(POSITIVE).sum())
    report["confirmed_hate"] = int((accepted["label"] == "hate").sum())
    report["per_stratum"] = {
        stratum: {
            "n": len(g),
            "accepted": int(g["agreed"].sum()),
            "positive_rate": round(
                float(g.loc[g["agreed"], "label"].isin(POSITIVE).mean()), 4
            ) if g["agreed"].any() else None,
            "hate_rate": round(
                float((g.loc[g["agreed"], "label"] == "hate").mean()), 4
            ) if g["agreed"].any() else None,
        }
        for stratum, g in merged.groupby("stratum", sort=False)
    }
    report["warnings_flagged"] = int(
        sum(bool(w) for src in sources for w in src.loc[idx, "warnings"])
    )

    if len(names) > 1:
        queue = merged[~merged["agreed"]].copy()
        queue["hate_involved"] = (
            (queue[f"label_{shorts[0]}"] == "hate")
            | (queue[f"label_{shorts[1]}"] == "hate")
        )
        queue = queue.sort_values("hate_involved", ascending=False)
        queue_cols = ["post_id", "stratum", "text", f"label_{shorts[0]}",
                      f"label_{shorts[1]}", f"rationale_{shorts[0]}",
                      f"rationale_{shorts[1]}", "hate_involved"]
        queue[queue_cols].to_csv(OUT / f"adjudication_queue_{args.tag}.csv", index=False)
        report["adjudication_queue"] = len(queue)
        report["adjudication_hate_involved"] = int(queue["hate_involved"].sum())

    merged.to_parquet(OUT / f"labels_2026_{args.tag}.parquet", index=False)

    if args.blind_check:
        sample = accepted.sample(min(args.blind_check, len(accepted)), random_state=7)
        sample[["post_id", "text"]].assign(human_label="").to_csv(
            OUT / f"blind_check_{args.tag}.csv", index=False
        )
        (OUT / f"blind_check_{args.tag}_key.csv").write_text(
            sample[["post_id", "label"]].to_csv(index=False)
        )
        report["blind_check_rows"] = len(sample)

    (OUT / f"14_merge_report_{args.tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

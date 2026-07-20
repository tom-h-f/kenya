# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
#     "scikit-learn>=1.4",
# ]
# ///
"""Coded-weighted blind check: the human gate on the 2026 labels.

`14_label_merge.py --blind-check` samples accepted rows uniformly, which
spends ~94% of the reviewer's time on benign posts. The axis that actually
failed is coded hate, so this samples that axis hard and keeps a random
slice for calibration.

    uv run 18_blind_check.py make  --labels out/labels_2026_full_final.parquet
    # ... Tom fills in the human_label column ...
    uv run 18_blind_check.py score --sheet out/blind_check_coded.csv

`make` writes two files: the sheet (post_id, text, human_label - NO model
columns, order shuffled) and a key held separately. `score` reports overall
and per-stratum agreement, the hate-axis confusion, and the two gates.
"""

from __future__ import annotations

import argparse
import json
import math
from statistics import NormalDist

import pandas as pd

from _common import ID2LABEL, LABELS, OUT, SEED

# Fractions of the sheet drawn from each pool. Coded pools first: these are
# the rows where the labeller's judgement is least trustworthy and where
# round 2 regressed. `random` is the calibration slice - without it the check
# cannot see over-flagging, only under-flagging.
POOLS = {
    "lexicon": 0.20,
    "nli_tail": 0.20,
    "hate_labelled": 0.20,
    "p_hate_high": 0.10,
    "random": 0.30,
}

# Used instead when the labels file carries two labellers. The 2026-07-20
# merge measured a 4.56x asymmetry - 146 rows Gemini called not-hate that
# Sonnet called hate, versus 32 the other way - so the sharpest question a
# human can settle is who is right where they split, not whether the
# consensus is sane. `agree_*` pools keep that second question in view.
DUAL_POOLS = {
    "split_gem_soft": 0.25,   # Gemini neither/offensive, Sonnet hate
    "split_gem_hard": 0.10,   # the reverse - guards against assuming a winner
    "agree_hate": 0.15,
    "agree_coded": 0.20,      # lexicon/nli_tail rows both labellers agreed on
    "random_agreed": 0.30,
}
GATE_OVERALL = 0.85
GATE_HATE = 0.70


def label_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    """The two per-labeller label columns, training labeller first.

    Column names follow the labeller short name, so the second opinion may be
    label_claude (agy Sonnet 4.6) or label_cursor (Cursor Sonnet 4.5) - do not
    hardcode either. Gemini goes first because it produced the training labels;
    the whole point is to test its calls.
    """
    meta = {"label_source"}  # merge metadata, not a labeller
    cols = [c for c in df.columns if c.startswith("label_") and c not in meta]
    if len(cols) != 2:
        return None
    primary = "label_gemini" if "label_gemini" in cols else cols[0]
    return primary, next(c for c in cols if c != primary)


def build_dual_pools(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pcol, scol = label_columns(df)
    g, s = df[pcol], df[scol]
    coded = df["stratum"].isin(["lexicon", "nli_tail"])
    used: set = set()
    pools = {}

    def take(name: str, mask: pd.Series) -> None:
        sub = df[mask & ~df["post_id"].isin(used)]
        pools[name] = sub
        used.update(sub["post_id"])

    take("split_gem_soft", (g != "hate") & (s == "hate"))
    take("split_gem_hard", (g == "hate") & (s != "hate"))
    take("agree_hate", (g == s) & (g == "hate"))
    take("agree_coded", (g == s) & coded)
    take("random_agreed", g == s)
    return pools


def build_pools(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    used: set = set()
    pools = {}

    def take(name: str, mask: pd.Series) -> None:
        sub = df[mask & ~df["post_id"].isin(used)]
        pools[name] = sub
        used.update(sub["post_id"])

    take("lexicon", df["stratum"] == "lexicon")
    take("nli_tail", df["stratum"] == "nli_tail")
    take("hate_labelled", df["label"] == "hate")
    take("p_hate_high", df["p_hate"] >= 0.5)
    take("random", pd.Series(True, index=df.index))
    return pools


def hate_axis_errors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Directional hate errors on rows carrying a consensus label."""
    scorable = df[df["label"].notna() & (df["label"] != "")]
    missed = scorable[
        (scorable["human_label"] == "hate") & (scorable["label"] != "hate")
    ]
    over = scorable[
        (scorable["label"] == "hate") & (scorable["human_label"] != "hate")
    ]
    return missed, over


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    observed = successes / total
    denominator = 1 + z * z / total
    centre = (observed + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(observed * (1 - observed) / total + z * z / (4 * total * total))
        / denominator
    )
    return centre - margin, centre + margin


def consensus_pool_agreement(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Agreement by pool where a consensus label actually exists."""
    scorable = df[df["label"].notna() & (df["label"] != "")].copy()
    scorable["agree"] = scorable["human_label"] == scorable["label"]
    return {
        str(name): {
            "n": int(len(group)),
            "agreement": round(float(group["agree"].mean()), 4),
        }
        for name, group in scorable.groupby("pool")
    }


def labeller_metrics(df: pd.DataFrame, column: str) -> dict:
    """Exact and hate-axis agreement for one labeller against the human."""
    human_hate = df["human_label"] == "hate"
    model_hate = df[column] == "hate"
    true_positive = int((human_hate & model_hate).sum())
    false_negative = int((human_hate & ~model_hate).sum())
    false_positive = int((~human_hate & model_hate).sum())
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    confusion = pd.crosstab(df[column], df["human_label"]).reindex(
        index=LABELS, columns=LABELS, fill_value=0
    )
    return {
        "exact_agreement": round(float((df[column] == df["human_label"]).mean()), 4),
        "hate": {
            "true_positive": true_positive,
            "false_negative": false_negative,
            "false_positive": false_positive,
            "precision": round(
                true_positive / precision_denominator if precision_denominator else 0.0, 4
            ),
            "recall": round(
                true_positive / recall_denominator if recall_denominator else 0.0, 4
            ),
        },
        "confusion": {
            str(model): {str(human): int(value) for human, value in row.items()}
            for model, row in confusion.to_dict(orient="index").items()
        },
    }


def interval_report(successes: int, total: int) -> dict:
    low, high = wilson_interval(successes, total)
    return {
        "successes": int(successes),
        "n": int(total),
        "rate": round(successes / total, 4) if total else None,
        "ci95": [round(low, 4), round(high, 4)] if total else None,
    }


def cmd_make(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.labels)
    if "split" in df and not args.include_gold:
        # gold is the future test set; leave it untouched so a human-verified
        # gold can be built separately without reusing rows seen here
        df = df[df["split"] != "gold"]
    cols = label_columns(df)
    dual = cols is not None
    if dual:
        print(f"dual-labeller file: {len(df)} rows, arbitrating disagreements\n")
        pools, spec = build_dual_pools(df), DUAL_POOLS
    else:
        print(f"single-labeller file: {len(df)} rows, coded-weighted sample\n")
        pools, spec = build_pools(df), POOLS

    picks = []
    for name, frac in spec.items():
        want = round(args.n * frac)
        pool = pools[name]
        got = pool.sample(min(want, len(pool)), random_state=SEED)
        picks.append(got.assign(pool=name))
        print(f"{name:14s} want {want:3d}  available {len(pool):5d}  took {len(got):3d}")

    sheet = pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sheet_path = OUT / "blind_check_coded.csv"
    key_path = OUT / "blind_check_coded_key.csv"

    key_cols = ["post_id", "pool", "stratum", "label"]
    if dual:
        key_cols += list(cols)
    sheet[["post_id", "text"]].assign(human_label="").to_csv(sheet_path, index=False)
    sheet[key_cols].to_csv(key_path, index=False)

    print(f"\nwrote {sheet_path} ({len(sheet)} rows) and {key_path}")
    print(f"labels to use: {', '.join(LABELS)}")
    print("fill in human_label for every row, then run: 18_blind_check.py score")


def cmd_score(args: argparse.Namespace) -> None:
    sheet = pd.read_csv(args.sheet)
    key = pd.read_csv(OUT / "blind_check_coded_key.csv")
    df = sheet.merge(key, on="post_id", validate="one_to_one")

    df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()
    blank = df["human_label"].isin(["", "nan"])
    if blank.any():
        raise SystemExit(f"{blank.sum()} rows have no human_label - fill them in first")
    bad = set(df["human_label"]) - set(LABELS)
    if bad:
        raise SystemExit(f"unrecognised labels: {bad}. Use: {LABELS}")

    arbitration = None
    per_labeller = {}
    cols = label_columns(df)
    if cols:
        pcol, scol = cols
        per_labeller = {
            pcol: labeller_metrics(df, pcol),
            scol: labeller_metrics(df, scol),
        }
        split = df[df[pcol] != df[scol]]
        if len(split):
            gem = int((split["human_label"] == split[pcol]).sum())
            son = int((split["human_label"] == split[scol]).sum())
            neither_side = len(split) - gem - son
            print(f"=== disagreement arbitration ({len(split)} rows) ===")
            print(f"  human sided with Gemini (the training labeller): {gem}")
            print(f"  human sided with the second labeller:            {son}")
            print(f"  human agreed with neither:                       {neither_side}")
            soft = split[(split[pcol] != "hate") & (split[scol] == "hate")]
            soft_report = None
            if len(soft):
                human_hate = int((soft["human_label"] == "hate").sum())
                soft_report = interval_report(human_hate, len(soft))
                low, high = soft_report["ci95"]
                print(f"\n  On the {len(soft)} rows Gemini called not-hate and Sonnet "
                      f"called hate,\n  the human called {human_hate} of them hate "
                      f"({human_hate / len(soft):.0%}, 95% CI {low:.0%}-{high:.0%}).")
                print("  Predeclared relabel gate (>50%): "
                      f"{'PASS' if human_hate / len(soft) > 0.5 else 'FAIL'}")
            arbitration = {
                "n": int(len(split)),
                "human_sided_primary": gem,
                "human_sided_secondary": son,
                "human_sided_neither": int(neither_side),
                "primary_soft_secondary_hate": soft_report,
                "relabel_gate_pass": bool(
                    soft_report is not None and soft_report["rate"] > 0.5
                ),
            }
            print()

    scorable = df[df["label"].notna() & (df["label"] != "")]
    df["agree"] = df["label"].notna() & (df["human_label"] == df["label"])
    overall = scorable["human_label"].eq(scorable["label"]).mean()
    hate_rows = scorable[scorable["label"] == "hate"]
    hate_agree = (
        hate_rows["human_label"].eq(hate_rows["label"]).mean()
        if len(hate_rows) else float("nan")
    )
    print(f"=== consensus gates (on the {len(scorable)} rows both labellers agreed) ===")

    overall_report = interval_report(
        int(scorable["human_label"].eq(scorable["label"]).sum()), len(scorable)
    )
    hate_report = interval_report(
        int(hate_rows["human_label"].eq(hate_rows["label"]).sum()), len(hate_rows)
    )
    print(f"overall agreement: {overall:.3f}  (gate {GATE_OVERALL}) "
          f"{'PASS' if overall >= GATE_OVERALL else 'FAIL'}")
    print(f"hate-row agreement: {hate_agree:.3f}  (gate {GATE_HATE}) "
          f"{'PASS' if hate_agree >= GATE_HATE else 'FAIL'}  n={len(hate_rows)}")

    by_pool = consensus_pool_agreement(df)
    print("\nby consensus pool:")
    for name, metrics in by_pool.items():
        print(f"  {name:14s} n={metrics['n']:3d}  agreement {metrics['agreement']:.3f}")

    print("\nconsensus confusion (rows = consensus, cols = human):")
    print(pd.crosstab(df["label"], df["human_label"]).to_string())

    missed, over = hate_axis_errors(df)
    print(f"\nconsensus MISSED hate (human=hate, consensus=not): {len(missed)}")
    print(f"consensus OVER-called hate (consensus=hate, human=not): {len(over)}")

    if per_labeller:
        print("\nper labeller:")
        for column, metrics in per_labeller.items():
            hate = metrics["hate"]
            print(
                f"  {column}: exact={metrics['exact_agreement']:.3f} "
                f"hate P/R={hate['precision']:.3f}/{hate['recall']:.3f} "
                f"FN={hate['false_negative']} FP={hate['false_positive']}"
            )

    gate_results = {
        "primary_split_relabel": arbitration["relabel_gate_pass"] if arbitration else None,
        "consensus_overall": bool(overall >= GATE_OVERALL),
        "consensus_hate": bool(hate_agree >= GATE_HATE),
    }
    print("\nPREDECLARED GATES:")
    for name, passed in gate_results.items():
        status = "N/A" if passed is None else "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    print("DECISION: calibration required; no automatic relabel or flag-head promotion")

    report = {
        "n": len(df),
        "arbitration": arbitration,
        "consensus": {
            "n": int(len(scorable)),
            "overall": overall_report,
            "hate_rows": hate_report,
            "missed_hate": int(len(missed)),
            "over_called_hate": int(len(over)),
            "by_pool": by_pool,
        },
        "per_labeller": per_labeller,
        "gates": gate_results,
        "decision": "calibration_required",
    }
    (OUT / "18_blind_check_report.json").write_text(json.dumps(report, indent=2))
    df.to_csv(OUT / "blind_check_coded_scored.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make")
    mk.add_argument("--labels", default=str(OUT / "labels_2026_full_final.parquet"))
    mk.add_argument("--n", type=int, default=120)
    mk.add_argument("--include-gold", action="store_true")
    mk.set_defaults(func=cmd_make)

    sc = sub.add_parser("score")
    sc.add_argument("--sheet", default=str(OUT / "blind_check_coded.csv"))
    sc.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

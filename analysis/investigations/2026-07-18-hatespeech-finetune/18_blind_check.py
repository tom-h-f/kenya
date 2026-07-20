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

    cols = label_columns(df)
    if cols:
        pcol, scol = cols
        split = df[df[pcol] != df[scol]]
        if len(split):
            gem = (split["human_label"] == split[pcol]).sum()
            son = (split["human_label"] == split[scol]).sum()
            neither_side = len(split) - gem - son
            print(f"=== disagreement arbitration ({len(split)} rows) ===")
            print(f"  human sided with Gemini (the training labeller): {gem}")
            print(f"  human sided with the second labeller:            {son}")
            print(f"  human agreed with neither:                       {neither_side}")
            soft = split[(split[pcol] != "hate") & (split[scol] == "hate")]
            if len(soft):
                human_hate = (soft["human_label"] == "hate").sum()
                print(f"\n  On the {len(soft)} rows Gemini called not-hate and Sonnet "
                      f"called hate,\n  the human called {human_hate} of them hate "
                      f"({human_hate / len(soft):.0%}).")
                print("  >50% means the TRAINING LABELS ARE TOO CONSERVATIVE - "
                      "prompt v3 + relabel.")
            print()

    scorable = df[df["label"].notna() & (df["label"] != "")]
    df["agree"] = df["human_label"] == df["label"]
    overall = scorable["human_label"].eq(scorable["label"]).mean()
    hate_rows = scorable[scorable["label"] == "hate"]
    hate_agree = (
        hate_rows["human_label"].eq(hate_rows["label"]).mean()
        if len(hate_rows) else float("nan")
    )
    print(f"=== consensus gates (on the {len(scorable)} rows both labellers agreed) ===")

    print(f"overall agreement: {overall:.3f}  (gate {GATE_OVERALL}) "
          f"{'PASS' if overall >= GATE_OVERALL else 'FAIL'}")
    print(f"hate-row agreement: {hate_agree:.3f}  (gate {GATE_HATE}) "
          f"{'PASS' if hate_agree >= GATE_HATE else 'FAIL'}  n={len(hate_rows)}")

    print("\nby pool:")
    for name, g in df.groupby("pool"):
        print(f"  {name:14s} n={len(g):3d}  agreement {g['agree'].mean():.3f}")

    print("\nconfusion (rows = labeller, cols = human):")
    print(pd.crosstab(df["label"], df["human_label"]).to_string())

    # The directional question: is the labeller too soft on coded hate?
    missed, over = hate_axis_errors(df)
    print(f"\nlabeller MISSED hate (human=hate, labeller=not): {len(missed)}")
    print(f"labeller OVER-called hate (labeller=hate, human=not): {len(over)}")
    verdict = (
        "labels are too conservative on hate -> widen the prompt (v3), relabel"
        if len(missed) > len(over) + 2
        else "labels are too aggressive on hate -> tighten the prompt"
        if len(over) > len(missed) + 2
        else "no systematic hate-axis bias detected"
    )
    print(f"VERDICT: {verdict}")

    if len(missed):
        print("\nmissed-hate examples (these define prompt v3):")
        for _, r in missed.head(8).iterrows():
            print(f"  [{r['label']}] {r['text'][:110]}")

    report = {
        "n": len(df),
        "overall_agreement": round(float(overall), 4),
        "hate_agreement": round(float(hate_agree), 4),
        "gate_overall_pass": bool(overall >= GATE_OVERALL),
        "gate_hate_pass": bool(hate_agree >= GATE_HATE),
        "missed_hate": len(missed),
        "over_called_hate": len(over),
        "by_pool": {k: round(float(g["agree"].mean()), 4) for k, g in df.groupby("pool")},
        "verdict": verdict,
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

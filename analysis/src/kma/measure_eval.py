"""A9: measure the relevance gate instead of trusting it.

    uv run kma-measure-eval sample --snapshot 2026-09-05-promotion-off --n 300
    # fill in the `label` column by hand, then:
    uv run kma-measure-eval score out/measure_sample.csv

    # B2: a blind human pass over 100 of the same posts, scored beside the model's labels
    uv run kma-measure-eval human-sheet out/measure_sample.csv --n 100
    uv run kma-measure-eval agree out/measure_human.csv out/measure_sample.csv

`kma.measure.domain_bucket` is a regex lexicon that decides whether a post is
about Kenya. It gates cluster promotion in the collector, it is how the v2 top
500 was reported as 37% Kenya-referencing, and its error rate has never been
measured. Every number that rests on it currently rests on an assumption.

## Why the sample is stratified

Kenya-relevant posts are a minority of the corpus, and the gate's failure modes
are asymmetric: a lexicon of names and places catches explicit mentions and
misses everything said obliquely - which is precisely how coded political speech
works. A uniform sample would be mostly `offdomain` and would measure precision
well and recall barely at all. Stratifying by predicted bucket spends the
labelling budget where each error type actually lives.

The consequence: raw counts from this sample are NOT corpus rates. Precision is
estimated within each stratum and recall needs the stratum weights, which
`score` applies.

## What a label means

`kenya` if the post is about Kenyan politics, society or the election, including
posts that never say a Kenyan word but are unmistakably about it in context.
`offdomain` if it is about somewhere else. `unclear` if you cannot tell without
more context - these are excluded from the estimates and reported separately,
because forcing them into a class would hide the gate's real ambiguity.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from kma import bench, coord2, measure
from kma.db import connect

log = logging.getLogger(__name__)

BUCKETS = ("kenya", "offdomain", "ambiguous")
LABELS = ("kenya", "offdomain", "unclear")
DEFAULT_OUT = Path("out/measure_sample.csv")


def draw_sample(
    con: duckdb.DuckDBPyConnection,
    snapshot: str,
    *,
    n: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """An equal-sized sample per predicted bucket, plus the stratum weights.

    Weights are the bucket's share of the corpus, carried on every row so
    `score` can turn per-stratum rates back into corpus-level ones without
    re-reading the snapshot.
    """
    manifest = bench.load(snapshot, con=con)
    view = coord2.posts_view(bench.pinned_source(manifest, "posts"))

    posts = con.sql(
        f"""SELECT post_id, user_id, text FROM ({view})
            WHERE text IS NOT NULL AND length(trim(text)) > 0
            ORDER BY post_id"""
    ).df()
    posts["bucket"] = [measure.domain_bucket(t) for t in posts["text"]]

    shares = posts["bucket"].value_counts(normalize=True)
    per_bucket = max(1, n // len(BUCKETS))
    rng = np.random.default_rng(seed)

    parts = []
    for bucket in BUCKETS:
        rows = posts[posts["bucket"] == bucket]
        if rows.empty:
            continue
        take = rows.sample(min(per_bucket, len(rows)), random_state=rng.integers(1 << 31))
        parts.append(take.assign(stratum_share=float(shares.get(bucket, 0.0))))

    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # Blind: the labeller must not see what the gate guessed.
    return sample.assign(label="")[
        ["post_id", "user_id", "text", "label", "bucket", "stratum_share"]
    ]


def score(labelled: pd.DataFrame) -> dict:
    """Precision, recall and F1 for the `kenya` class, corpus-weighted.

    Wilson intervals rather than normal ones: at these sample sizes a normal
    interval on a proportion near 0 or 1 runs outside [0, 1] and reads as
    precision above 100%.
    """
    df = labelled.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    unclear = int((df["label"] == "unclear").sum())
    unlabelled = int((df["label"] == "").sum())
    df = df[df["label"].isin(("kenya", "offdomain"))]
    if df.empty:
        raise ValueError("no usable labels: fill in the `label` column first")

    df["predicted_kenya"] = df["bucket"] == "kenya"
    df["actual_kenya"] = df["label"] == "kenya"

    # Each labelled row stands for its stratum's share of the corpus.
    counts = df.groupby("bucket").size()
    df["weight"] = df.apply(
        lambda r: r["stratum_share"] / max(int(counts[r["bucket"]]), 1), axis=1
    )

    tp = float(df.loc[df.predicted_kenya & df.actual_kenya, "weight"].sum())
    fp = float(df.loc[df.predicted_kenya & ~df.actual_kenya, "weight"].sum())
    fn = float(df.loc[~df.predicted_kenya & df.actual_kenya, "weight"].sum())

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else float("nan")

    n_pred = int((df["bucket"] == "kenya").sum())
    n_correct = int((df.predicted_kenya & df.actual_kenya).sum())
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "precision_ci": _wilson(n_correct, n_pred),
        "labelled": len(df),
        "unclear": unclear,
        "unlabelled": unlabelled,
        "by_bucket": df.groupby("bucket")["actual_kenya"].mean().round(3).to_dict(),
    }


def _wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def human_subset(sample: pd.DataFrame, n: int = 100, seed: int = 0) -> pd.DataFrame:
    """A blind, stratified subset of an already-labelled sample, for a human pass.

    The first labelled sample carries model labels, so every Kenya-share figure
    corrected by its recall rests on how far a model agrees with the regex. A
    human pass over part of the SAME posts turns that into something defensible
    without relabelling all of it.

    Allocation matches `draw_sample` - equal per bucket - so the subset scores
    through `score` unchanged. Blind twice over: the sheet shows neither the
    gate's bucket nor the model's label.
    """
    k = len(BUCKETS)
    sizes = {bucket: n // k + (1 if i < n % k else 0) for i, bucket in enumerate(BUCKETS)}
    rng = np.random.default_rng(seed)

    parts = []
    for bucket in BUCKETS:
        rows = sample[sample["bucket"] == bucket]
        if rows.empty:
            continue
        take = min(sizes[bucket], len(rows))
        parts.append(rows.sample(take, random_state=rng.integers(1 << 31)))

    subset = pd.concat(parts, ignore_index=True)
    subset = subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return subset.assign(label="")[["post_id", "text", "label"]]


def agreement(human: pd.DataFrame, sample: pd.DataFrame) -> dict:
    """The gate scored against the human and against the model, on the SAME posts.

    The model's full-sample figures are not comparable to a subset; only the
    same posts are. Agreement between the two labellers is reported raw and as
    Cohen's kappa over all three labels, `unclear` included - disagreeing about
    what is unclear is still disagreeing.
    """
    h = human.assign(
        post_id=human["post_id"].astype(str),
        label=human["label"].fillna("").astype(str).str.strip().str.lower(),
    )
    unknown = sorted(set(h["label"]) - set(LABELS) - {""})
    if unknown:
        raise ValueError(f"labels must be one of {LABELS}; found {unknown}")

    key = sample.assign(
        post_id=sample["post_id"].astype(str),
        label=sample["label"].fillna("").astype(str).str.strip().str.lower(),
    )
    merged = h[["post_id", "label"]].merge(
        key[["post_id", "bucket", "stratum_share", "label"]],
        on="post_id",
        suffixes=("_human", "_model"),
        validate="one_to_one",
    )
    labelled = merged[merged["label_human"] != ""]
    if labelled.empty:
        raise ValueError("no usable labels: fill in the `label` column first")

    a, b = labelled["label_human"], labelled["label_model"]
    observed = float((a == b).mean())
    expected = float(sum((a == lab).mean() * (b == lab).mean() for lab in LABELS))
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")

    return {
        "posts": len(labelled),
        "agreement": round(observed, 3),
        "kappa": round(kappa, 3),
        "confusion_human_by_model": pd.crosstab(a, b).to_dict(),
        "gate_vs_human": score(labelled.rename(columns={"label_human": "label"})),
        "gate_vs_model": score(labelled.rename(columns={"label_model": "label"})),
    }


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Measure the relevance gate.")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="draw a blind stratified sample to label")
    s.add_argument("--snapshot", required=True)
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=DEFAULT_OUT)

    c = sub.add_parser("score", help="score a filled-in sample")
    c.add_argument("path", type=Path)

    h = sub.add_parser("human-sheet", help="blind subset of a labelled sample for a human pass")
    h.add_argument("sample", type=Path)
    h.add_argument("--n", type=int, default=100)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--out", type=Path, default=Path("out/measure_human.csv"))

    g = sub.add_parser("agree", help="score a human sheet against the gate and the model")
    g.add_argument("sheet", type=Path)
    g.add_argument("sample", type=Path)

    args = ap.parse_args()

    if args.command == "sample":
        con = connect()
        sample = draw_sample(con, args.snapshot, n=args.n, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(args.out, index=False)
        print(f"wrote {args.out} ({len(sample)} rows)")
        print("Label each row `kenya`, `offdomain` or `unclear`, then:")
        print(f"    uv run kma-measure-eval score {args.out}")
        return

    if args.command == "human-sheet":
        sheet = human_subset(pd.read_csv(args.sample, dtype={"post_id": str}), n=args.n, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(args.out, index=False)
        print(f"wrote {args.out} ({len(sheet)} rows)")
        print("Label each row `kenya`, `offdomain` or `unclear`, then:")
        print(f"    uv run kma-measure-eval agree {args.out} {args.sample}")
        return

    if args.command == "agree":
        result = agreement(
            pd.read_csv(args.sheet, dtype={"post_id": str}),
            pd.read_csv(args.sample, dtype={"post_id": str}),
        )
        for key, value in result.items():
            print(f"{key}: {value}")
        return

    result = score(pd.read_csv(args.path))
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

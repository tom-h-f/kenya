# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
#     "scikit-learn>=1.4",
#     "lingua-language-detector>=2.0",
#     "datasketch>=1.6",
# ]
# ///
"""Clean HateSpeech_Kenya.csv and write stratified train/val/test splits.

Class mapping (verified from annotator-count means in the source CSV):
0 = neither, 1 = offensive, 2 = hate.

v2 (Plan B): annotator-agreement columns, language-ID drop of clearly
non-en/sw rows, near-duplicate clustering (MinHash on char 5-grams) with all
multi-row clusters routed to train so variants never straddle splits, plus a
unanimous-only test file as the canonical eval set. Split v2 is NOT comparable
with round-1 splits - retrain before quoting numbers.
"""

from __future__ import annotations

import argparse
import ast
import json
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from _common import LABELS, OUT, SEED, default_csv

USERNAME_RE = re.compile(r"USERNAME_\d+")
WS_RE = re.compile(r"\s+")
NORM_RE = re.compile(r"[^a-z0-9 ]+")

COUNT_COLS = ["neither", "offensive_language", "hate_speech"]


def unwrap(raw: str) -> str:
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return " ".join(str(p) for p in parsed)
    except (ValueError, SyntaxError):
        pass
    return raw.strip().strip("[]").strip("'\"")


def clean(raw: str) -> str:
    text = unwrap(raw)
    text = USERNAME_RE.sub("@user", text)
    return WS_RE.sub(" ", text).strip()


KENYA_MARKERS = re.compile(
    r"kikuyu|kalenjin|luo|luhya|kamba|kisii|masai|maasai|kenya|ruto|uhuru"
    r"|odinga|raila|nairobi|mombasa|tribal|tribe|ukabila",
    re.IGNORECASE,
)


def language_drop_mask(texts: pd.Series) -> pd.Series:
    """True = drop: confidently detected as neither English nor Swahili.

    Conservative: high-accuracy mode, en/sw must be absent from the top-2
    candidates, and Kenya-topic rows are always kept - tweets are short and
    code-mixed, so false drops cost more than leftover noise.
    """
    from lingua import Language, LanguageDetectorBuilder

    keep = {Language.ENGLISH, Language.SWAHILI}
    candidates = [
        Language.ENGLISH, Language.SWAHILI, Language.MALAY,
        Language.INDONESIAN, Language.FRENCH, Language.PORTUGUESE,
        Language.SPANISH, Language.GERMAN, Language.DUTCH, Language.SOMALI,
        Language.TAGALOG,
    ]
    detector = LanguageDetectorBuilder.from_languages(*candidates).build()
    drop = []
    for t in texts:
        if len(t) < 40 or KENYA_MARKERS.search(t):
            drop.append(False)
            continue
        values = detector.compute_language_confidence_values(t)
        top2 = values[:2]
        drop.append(
            top2[0].value >= 0.95
            and all(v.language not in keep for v in top2)
        )
    return pd.Series(drop, index=texts.index)


def near_dup_clusters(texts: pd.Series) -> pd.Series:
    """Cluster id per row; rows sharing MinHash-LSH near-dups share an id."""
    from datasketch import MinHash, MinHashLSH

    def norm(t: str) -> str:
        return NORM_RE.sub("", t.lower().replace("@user", " "))

    def shingles(t: str) -> set[bytes]:
        return {t[i : i + 5].encode() for i in range(max(1, len(t) - 4))}

    lsh = MinHashLSH(threshold=0.85, num_perm=64)
    hashes = {}
    for idx, t in texts.items():
        m = MinHash(num_perm=64)
        for s in shingles(norm(t)):
            m.update(s)
        hashes[idx] = m
        lsh.insert(idx, m)

    parent = {idx: idx for idx in texts.index}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for idx, m in hashes.items():
        for other in lsh.query(m):
            ra, rb = find(idx), find(other)
            if ra != rb:
                parent[rb] = ra
    return pd.Series({idx: find(idx) for idx in texts.index})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(default_csv()))
    ap.add_argument("--skip-langid", action="store_true")
    args = ap.parse_args()
    report: dict = {"split_version": 2}

    df = pd.read_csv(args.csv)
    df["text"] = df["Tweet"].map(clean)
    df["label"] = df["Class"].astype(int)

    df["n_votes"] = df[COUNT_COLS].sum(axis=1)
    df["agreement"] = df[COUNT_COLS].max(axis=1) / df["n_votes"].clip(lower=1)
    df["is_unanimous"] = df["agreement"] >= 1.0

    report["rows_raw"] = len(df)
    df = df[df["text"].str.len() > 0].drop_duplicates("text")
    report["rows_after_exact_dedup"] = len(df)

    agree_dist = (
        df["agreement"].round(2).value_counts().sort_index().to_dict()
    )
    report["agreement_distribution"] = {str(k): v for k, v in agree_dist.items()}
    report["vote_totals"] = df["n_votes"].value_counts().sort_index().to_dict()
    print(f"agreement: {agree_dist}")
    print(f"unanimous: {df['is_unanimous'].sum()} / {len(df)}")

    if not args.skip_langid:
        drop = language_drop_mask(df["text"])
        report["langid_dropped"] = int(drop.sum())
        (OUT / "00_langid_dropped.csv").parent.mkdir(exist_ok=True)
        df[drop][["text", "label"]].to_csv(
            OUT / "00_langid_dropped.csv", index=False
        )
        print(f"langid drop: {drop.sum()} rows (sample in out/00_langid_dropped.csv)")
        df = df[~drop]

    clusters = near_dup_clusters(df["text"])
    df["cluster"] = clusters
    sizes = df.groupby("cluster").size()
    multi = sizes[sizes > 1]
    report["near_dup_clusters"] = int(len(multi))
    report["near_dup_rows"] = int(multi.sum())
    print(f"near-dup: {len(multi)} clusters covering {multi.sum()} rows -> train")

    df = df[["text", "label", "n_votes", "agreement", "is_unanimous", "cluster"]]
    in_multi = df["cluster"].isin(multi.index)
    singles = df[~in_multi]

    train, rest = train_test_split(
        singles, test_size=0.2, stratify=singles["label"], random_state=SEED
    )
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest["label"], random_state=SEED
    )
    train = pd.concat([train, df[in_multi]])

    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / "clean.parquet", index=False)
    splits = [("train", train), ("val", val), ("test", test),
              ("test_unanimous", test[test["is_unanimous"]])]
    for name, part in splits:
        part.reset_index(drop=True).to_parquet(OUT / f"{name}.parquet", index=False)
        counts = part["label"].value_counts().sort_index()
        dist = ", ".join(
            f"{LABELS[i]}={counts.get(i, 0)}" for i in range(len(LABELS))
        )
        print(f"{name}: {len(part)} rows ({dist})")
        report[f"split_{name}"] = {LABELS[i]: int(counts.get(i, 0)) for i in range(3)}

    (OUT / "00_prep_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

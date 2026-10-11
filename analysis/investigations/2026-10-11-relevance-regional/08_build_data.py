"""Assemble the data directory that goes to mike-pc: train set and every eval set.

    cd analysis && uv run python investigations/2026-10-11-relevance-regional/08_build_data.py

train.parquet: the original 3,000 opus labels plus the new regional train labels,
minus any row whose author is in the holdout. Eval sets (never trained on):

    human88    Tom's 88 placeable posts from the 2026-09-13 check
    b2human    Tom's 100 from B2 (2026-09-11)
    b2model    the model-labelled 300 of B2
    tz200      the 200 Tanzania / community 8 posts headless-labelled 2026-10-11
    regional   the fresh author-disjoint regional holdout, opus-labelled 2026-10-11
"""

import shutil
from pathlib import Path

import pandas as pd

from kma import measure

HERE = Path(__file__).parent
OUT = HERE / "out"
OLD = OUT / "old"
DATA = OUT / "data"


def gate_shares() -> dict[str, float]:
    ms = pd.read_csv(OLD / "measure_sample.csv")
    return ms.groupby("bucket")["stratum_share"].first().to_dict()


def main() -> None:
    DATA.mkdir(exist_ok=True)
    shares = gate_shares()

    old = pd.read_parquet(OLD / "labelled.parquet")[["post_id", "user_id", "text", "label"]].astype({"post_id": str})
    old["source"] = "opus3000"
    old["country"] = None
    new = pd.read_parquet(OUT / "labelled_train.parquet").rename(columns={"author_id": "user_id"})
    new = new[["post_id", "user_id", "text", "label", "country", "stratum"]].assign(source="regional")
    hold = pd.read_parquet(OUT / "labelled_holdout.parquet")
    hold["post_id"] = hold["post_id"].astype(str)
    held_authors = set(hold["author_id"].astype(str))
    train = pd.concat([old, new], ignore_index=True)
    before = len(train)
    train = train[~train["user_id"].astype(str).isin(held_authors)]
    train = train.drop_duplicates("post_id")
    train.drop(columns=["user_id"]).to_parquet(DATA / "train.parquet")
    print(f"train {len(train)} (dropped {before - len(train)} sharing an author with the holdout); "
          f"{train['label'].value_counts().to_dict()}")

    key = pd.read_csv(OLD / "human_check_key.csv").astype({"post_id": str})
    filled = pd.read_csv(OLD / "human_check_filled.csv").astype({"post_id": str})
    h88 = filled.merge(key[["post_id", "text"]], on="post_id")
    h88["bucket"] = h88["text"].map(measure.domain_bucket)
    h88.to_parquet(DATA / "eval_human88.parquet")

    for name, path in (("b2human", "measure_human.csv"), ("b2model", "measure_sample.csv")):
        frame = pd.read_csv(OLD / path).astype({"post_id": str})
        if "bucket" not in frame:
            frame = frame.merge(pd.read_csv(OLD / "measure_sample.csv")[["post_id", "bucket"]].astype({"post_id": str}), on="post_id")
        frame.to_parquet(DATA / f"eval_{name}.parquet")

    tz = pd.read_parquet(OLD / "tz_labelled.parquet")
    tz.astype({"post_id": str}).to_parquet(DATA / "eval_tz200.parquet")

    hold.to_parquet(DATA / "eval_regional.parquet")
    for path in sorted(DATA.glob("*.parquet")):
        print(f"  {path.name}: {len(pd.read_parquet(path))}")
    (DATA / "gate_shares.json").write_text(__import__("json").dumps(shares))
    shutil.copy(HERE.parents[1] / "src" / "kma" / "relevance_prep.py", DATA.parent / "relevance_prep.py")


if __name__ == "__main__":
    main()

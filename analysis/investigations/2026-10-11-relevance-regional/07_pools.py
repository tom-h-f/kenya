"""Draw the regional hard-negative and hard-positive pools to label.

    cd analysis && uv run python investigations/2026-10-11-relevance-regional/07_pools.py

Universe: non-retweet posts, one per distinct normalised text, with words of their
own. Every post falls in exactly one stratum (first match wins), so a stratum's
size in the universe is its weight: a labelled post stands for N_stratum / n_stratum
posts, which gives corpus-weighted figures on the holdout.

Train and holdout are split by AUTHOR (hash of author_id), so the holdout never
shares an account with the training pool. Excluded everywhere: the original 3,000
training labels, the B2 sets, Tom's 88-post human check, and the 200 Tanzania/c8
posts labelled on 2026-10-11 (kept as a separate eval set).
"""

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from regional import REGION_RE

HERE = Path(__file__).parent
OUT = HERE / "out"
OLD = OUT / "old"
HOLDOUT_MOD = 5

REGION_GROUP = {"tanzania": "tanzania", "uganda": "uganda", "rwanda": "great_lakes", "burundi": "great_lakes",
                "drc": "great_lakes", "ethiopia": "horn", "somalia": "horn", "sudan": "horn", "south_sudan": "horn",
                "nigeria": "west_africa", "ghana": "west_africa", "south_africa": "south_africa",
                "other_africa": "other_africa"}
GROUP_ORDER = ["tanzania", "uganda", "great_lakes", "horn", "west_africa", "south_africa", "other_africa"]

SWAHILI = re.compile(r"\b(?:ni|na|kwa|wa|ya|sana|hii|hiyo|yako|wewe|mimi|hapa|kuna|nini|bado|tafadhali|asante|"
                     r"serikali|wananchi|rais|sasa|lakini|kama|ndio|hapana|pia|yetu|wetu|watu|kila|mimi|"
                     r"tuko|uko|niko|kweli|sawa|poa|mbaya|nzuri|leo|kesho|jana)\b", re.I)
MENTION = re.compile(r"@(\w+)")

# (train, holdout) posts per stratum
QUOTA = {
    **{f"region_{g}_hi": q for g, q in zip(GROUP_ORDER, [(100, 50), (60, 40), (50, 30), (30, 20), (50, 40), (30, 20), (30, 20)])},
    **{f"region_{g}_lo": q for g, q in zip(GROUP_ORDER, [(20, 15), (20, 15), (30, 20), (40, 30), (30, 20), (20, 20), (20, 20)])},
    "c8": (40, 25), "anchor_region": (70, 25), "diaspora": (30, 15), "swahili_kenyan": (60, 30),
    "swahili_other": (30, 15), "handle_short": (60, 30), "handle_short_other": (30, 20), "mined_fp": (50, 25),
    "rest": (0, 100),
}


def in_holdout(author) -> bool:
    return int(hashlib.md5(str(author).encode()).hexdigest(), 16) % HOLDOUT_MOD == 0


def main() -> None:
    feats = pd.read_parquet(OUT / "features.parquet")
    text = pd.read_parquet(OUT / "corpus.parquet", columns=["platform_post_id", "text"])
    text["platform_post_id"] = text["platform_post_id"].astype(str)
    posts = feats.merge(text, on="platform_post_id")
    posts = posts[~posts["is_rt"] & (posts["content_words"] >= 1)].drop_duplicates("norm")

    excluded: set[str] = set(pd.read_parquet(OLD / "labelled.parquet")["post_id"].astype(str))
    for name in ("measure_human.csv", "measure_sample.csv", "human_check_key.csv"):
        excluded |= set(pd.read_csv(OLD / name)["post_id"].astype(str))
    tz = pd.read_parquet(OLD / "tz_labelled.parquet")
    excluded |= set(tz["post_id"].astype(str))
    c8_ids = set(pd.read_parquet(OLD / "c8_member_posts.parquet")["post_id"].astype(str)) - excluded
    posts = posts[~posts["platform_post_id"].isin(excluded)].copy()

    known = frozenset(Path(OUT / "kenyan_handles.txt").read_text().split())
    mentioned = posts["text"].map(lambda t: [m.lower() for m in MENTION.findall(t)])
    posts["has_known_handle"] = mentioned.map(lambda hs: any(h in known for h in hs))
    posts["has_handle"] = mentioned.map(bool)
    posts["sw_words"] = posts["text"].map(lambda t: len(SWAHILI.findall(t)))
    # first group in GROUP_ORDER among those hit, so a post naming Kenya's neighbours lands in the same place every time
    posts["group"] = posts["regions"].map(
        lambda r: next((g for g in GROUP_ORDER if any(REGION_GROUP[k] == g for k in r.split(",") if k)), None))
    hi = posts["p_old"] >= 0.5
    short = posts["content_words"] <= 4

    stratum = pd.Series("rest", index=posts.index)
    def assign(mask: pd.Series, name) -> None:
        free = stratum == "rest"
        stratum[mask & free] = name

    for g in GROUP_ORDER:
        base = (posts["group"] == g) & ~posts["anchor"]
        assign(base & hi, f"region_{g}_hi")
        assign(base & ~hi, f"region_{g}_lo")
    assign(posts["platform_post_id"].isin(c8_ids), "c8")
    assign(posts["anchor"] & posts["group"].notna(), "anchor_region")
    assign(posts["diaspora"], "diaspora")
    swahili = posts["sw_words"] >= 2
    assign(swahili & (posts["anchor"] | hi), "swahili_kenyan")
    assign(swahili & ~hi & ~posts["anchor"], "swahili_other")
    assign(short & posts["has_known_handle"], "handle_short")
    assign(short & posts["has_handle"] & ~posts["has_known_handle"] & hi, "handle_short_other")
    assign((posts["p_old"] >= 0.9) & ~posts["anchor"], "mined_fp")
    posts["stratum"] = stratum
    posts["holdout"] = posts["author_id"].map(in_holdout)

    sizes = posts["stratum"].value_counts()
    print("universe", len(posts), "excluded", len(excluded))
    rng = np.random.default_rng(7)
    parts = {"train": [], "holdout": []}
    report = []
    for name, (n_train, n_hold) in QUOTA.items():
        for split, n in (("train", n_train), ("holdout", n_hold)):
            pool = posts[(posts["stratum"] == name) & (posts["holdout"] == (split == "holdout"))]
            take = pool.sample(min(n, len(pool)), random_state=int(rng.integers(1 << 31)))
            parts[split].append(take)
            report.append({"stratum": name, "split": split, "want": n, "got": len(take), "available": len(pool),
                           "universe": int(sizes.get(name, 0))})
    print(pd.DataFrame(report).to_string(index=False))
    cols = ["platform_post_id", "author_id", "text", "stratum", "p_old", "regions", "anchor", "content_words",
            "sw_words", "has_known_handle", "lang"]
    for split, frames in parts.items():
        pool = pd.concat(frames)[cols].rename(columns={"platform_post_id": "post_id"})
        pool["stratum_size"] = pool["stratum"].map(sizes).astype(int)
        pool["stratum_n"] = pool.groupby("stratum")["stratum"].transform("size")
        pool["weight"] = pool["stratum_size"] / pool["stratum_n"]
        pool = pool.sample(frac=1.0, random_state=3).reset_index(drop=True)
        pool.to_parquet(OUT / f"pool_{split}.parquet")
        print(split, len(pool))
    posts[["platform_post_id", "stratum", "holdout"]].to_parquet(OUT / "universe_strata.parquet")


if __name__ == "__main__":
    main()

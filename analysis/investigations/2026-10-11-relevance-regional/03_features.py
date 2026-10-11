"""Per-post features for stratifying the regional pools. Local only.

    cd analysis && uv run python investigations/2026-10-11-relevance-regional/03_features.py

Reads out/corpus.parquet and the old model's persisted scores, writes
out/features.parquet: Kenya anchor, regions hit, diaspora, short-content flag,
a normalised text key for de-duplication, retweet flag, and the old p_kenya.
"""

import re
from pathlib import Path

import pandas as pd

from kma import measure, relevance
from regional import DIASPORA, REGION_RE

HERE = Path(__file__).parent
OUT = HERE / "out"
URL = re.compile(r"https?://\S+")
MENTION = re.compile(r"@\w+")


def main() -> None:
    posts = pd.read_parquet(OUT / "corpus.parquet", columns=["platform_post_id", "author_id", "author_handle",
                                                              "created_at", "text", "lang", "type", "mentions",
                                                              "is_repost"])
    posts["platform_post_id"] = posts["platform_post_id"].astype(str)
    old = pd.read_parquet(OUT / f"scores_{relevance.MODEL}.parquet")
    old["platform_post_id"] = old["platform_post_id"].astype(str)
    posts = posts.merge(old.rename(columns={"p_kenya": "p_old"}), on="platform_post_id", how="left")

    text = posts["text"].astype(str)
    posts["is_rt"] = posts["is_repost"].fillna(False).astype(bool) | text.str.startswith("RT @")
    posts["anchor"] = text.str.contains(measure._KENYA_RE)
    posts["diaspora"] = text.str.contains(DIASPORA.pattern, flags=re.I, regex=True)
    regions = {k: text.str.contains(rx.pattern, flags=re.I, regex=True) for k, rx in REGION_RE.items()}
    frame = pd.DataFrame(regions)
    posts["regions"] = frame.apply(lambda r: ",".join(frame.columns[r.to_numpy()]), axis=1)
    stripped = text.str.replace(URL, " ", regex=True).str.replace(MENTION, " ", regex=True)
    posts["content_words"] = stripped.str.findall(r"\w+").map(len)
    posts["norm"] = stripped.str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    posts.drop(columns=["text", "is_repost"]).to_parquet(OUT / "features.parquet")
    print(posts.shape, "rt", int(posts["is_rt"].sum()), "anchor", int(posts["anchor"].sum()),
          "scored", int(posts["p_old"].notna().sum()))
    print(posts["regions"].str.get_dummies(",").sum().to_dict())


if __name__ == "__main__":
    main()

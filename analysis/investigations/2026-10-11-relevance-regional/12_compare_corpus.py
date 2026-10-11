"""Old versus new relevance scores over the whole corpus.

    cd analysis && uv run python investigations/2026-10-11-relevance-regional/12_compare_corpus.py out/scores_new

What moved: the share of the corpus each model calls Kenyan (p >= 0.5), the flip
matrix, and the same by region keyword group, by gate bucket and by month.
"""

import sys
from pathlib import Path

import pandas as pd

from kma import measure, relevance

HERE = Path(__file__).parent
OUT = HERE / "out"

new = pd.concat([pd.read_parquet(p) for p in sorted(Path(sys.argv[1]).glob("part_*.parquet"))])
new["platform_post_id"] = new["platform_post_id"].astype(str)
old = pd.read_parquet(OUT / f"scores_{relevance.PREVIOUS_MODEL}.parquet").rename(columns={"p_kenya": "p_old"})
old["platform_post_id"] = old["platform_post_id"].astype(str)
feats = pd.read_parquet(OUT / "features.parquet", columns=["platform_post_id", "created_at", "anchor", "regions", "is_rt"])
df = feats.merge(new.rename(columns={"p_kenya": "p_new"}), on="platform_post_id").merge(old, on="platform_post_id", how="left")
df["new"] = df["p_new"] >= 0.5
df["old"] = df["p_old"] >= 0.5
both = df.dropna(subset=["p_old"])
print(f"{len(df):,} posts scored by the new model, {len(both):,} also scored by the old")
print(f"called Kenyan: old {both['old'].mean():.3f}, new {both['new'].mean():.3f}")
print(pd.crosstab(both["old"].map({True: "old kenya", False: "old other"}),
                  both["new"].map({True: "new kenya", False: "new other"}), margins=True).to_string())

groups = {k: both["regions"].str.contains(k) & ~both["anchor"] for k in
          ("tanzania", "uganda", "rwanda", "burundi", "drc", "ethiopia", "somalia", "south_sudan", "sudan",
           "nigeria", "ghana", "south_africa", "other_africa")}
rows = [{"group": k, "posts": int(m.sum()), "old_kenya": round(float(both.loc[m, "old"].mean()), 3),
         "new_kenya": round(float(both.loc[m, "new"].mean()), 3)} for k, m in groups.items()]
print("\nregion keyword, no Kenyan anchor (all posts, retweets included):")
print(pd.DataFrame(rows).to_string(index=False))

both = both.assign(month=both["created_at"].dt.strftime("%Y-%m"))
print("\nby month:")
print(both.groupby("month")[["old", "new"]].mean().round(3).join(both.groupby("month").size().rename("posts")).to_string())
anchor = both["anchor"]
print(f"\nposts with a Kenyan anchor word: old {both.loc[anchor, 'old'].mean():.3f} new {both.loc[anchor, 'new'].mean():.3f}")
print(f"posts without: old {both.loc[~anchor, 'old'].mean():.3f} new {both.loc[~anchor, 'new'].mean():.3f}")

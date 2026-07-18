"""Language targeting: EN vs SW framing of the same claim, and language mix of
flagged account groups.

- story language split: stories with both EN and SW member posts; divergent
  sentiment between the two languages = audience-differentiated framing
- cluster language mix: coordination clusters posting predominantly in one
  language (audience targeting fingerprint)

The platform `lang` field mislabels Sheng/code-switched text - a sample of
"en"-labelled posts containing Swahili function words is printed for eyeball
calibration before trusting the splits.
"""

from __future__ import annotations

import pandas as pd

import _lib
from kma import db
from kma import stories as st

MIN_PER_LANG = 4
SW_MARKERS = [" ni ", " na ", " ya ", " wa ", " kwa ", " sisi ", " hii ", " sana "]


def main() -> None:
    args = _lib.parse_args("Language targeting", default_sample=200)
    con = _lib.connect()

    posts = con.sql(
        f"""
        WITH lp AS (
            SELECT * FROM {db.posts_source('x')}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY collected_at DESC
            ) = 1
        ),
        ll AS (
            SELECT * FROM {db.labels_source('x')}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY labeled_at DESC
            ) = 1
        )
        SELECT lp.platform_post_id, lp.author_id, lp.author_handle, lp.lang,
               lp.text, ll.sentiment
        FROM lp LEFT JOIN ll USING (platform_post_id)
        WHERE lp.created_at IS NOT NULL
        """
    ).df()

    en = posts[posts["lang"] == "en"]
    marked = en[
        en["text"].str.lower().str.contains("|".join(SW_MARKERS), regex=True)
    ]
    print(
        f"lang calibration: {len(marked)}/{len(en)} 'en'-labelled posts carry "
        f"Swahili function words ({len(marked)/max(len(en),1):.1%}) - sample:"
    )
    for t in marked["text"].head(5):
        print("  |", t[:100].replace("\n", " "))

    members = st.candidate_stories(con, days=14, include_thin=True)
    m = members.merge(
        posts[["platform_post_id", "lang", "sentiment"]], on="platform_post_id"
    )
    m["lang2"] = m["lang"].where(m["lang"].isin(["en", "sw"]), "other")
    rows = []
    for sid, grp in m.groupby("story_id"):
        counts = grp["lang2"].value_counts()
        if counts.get("en", 0) < MIN_PER_LANG or counts.get("sw", 0) < MIN_PER_LANG:
            continue
        neg = grp.groupby("lang2")["sentiment"].apply(
            lambda s: float((s.dropna() == "negative").mean())
        )
        rows.append(
            {
                "story_id": sid,
                "n_en": int(counts["en"]),
                "n_sw": int(counts["sw"]),
                "neg_share_en": round(neg.get("en", float("nan")), 3),
                "neg_share_sw": round(neg.get("sw", float("nan")), 3),
                "neg_gap": round(
                    abs(neg.get("en", 0) - neg.get("sw", 0)), 3
                ),
                "text": grp["text"].iloc[0][:100],
            }
        )
    split = pd.DataFrame(
        rows, columns=["story_id", "n_en", "n_sw", "neg_share_en",
                       "neg_share_sw", "neg_gap", "text"],
    ).sort_values("neg_gap", ascending=False)
    _lib.show(split, "bilingual stories by EN/SW negative-sentiment gap")
    _lib.save(split, "07_bilingual_stories.csv")

    clusters = _lib.coordination_clusters()
    cm = clusters.merge(posts, left_on="author_id", right_on="author_id")
    mix = cm.groupby("cluster_id").agg(
        cluster_size=("size", "first"),
        n_posts=("platform_post_id", "size"),
        sw_share=("lang", lambda s: float((s == "sw").mean())),
        en_share=("lang", lambda s: float((s == "en").mean())),
        handles=("author_handle", lambda s: ", ".join(s.unique()[:5])),
    )
    mix = mix[mix["n_posts"] >= 10].sort_values("sw_share", ascending=False)
    _lib.show(mix.reset_index(), "coordination clusters by language mix")
    _lib.save(mix.reset_index(), "07_cluster_lang_mix.csv")

    _lib.print_caveats()


if __name__ == "__main__":
    main()

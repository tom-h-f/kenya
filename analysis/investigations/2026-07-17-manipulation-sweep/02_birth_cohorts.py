"""Birth cohorts: account-creation clustering inside coordination clusters and
hashtag communities, plus handle pattern families.

Tells looked for:
- coordination clusters whose members were created in a tight window
  (max share of members inside any 90-day creation window, vs random-draw null)
- hashtag author cohorts skewed toward recently created accounts
- handle families: same skeleton with different digit runs (name123/name456),
  long digit tails, created close together
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import _lib
from kma import db

WINDOW_DAYS = 90
RECENT_DAYS = 180
MIN_CLUSTER_DATED = 4
MIN_TAG_AUTHORS = 20
TOP_TAGS = 30


def max_window_share(days: np.ndarray, window: int = WINDOW_DAYS) -> float:
    d = np.sort(days)
    best, j = 0, 0
    for i in range(len(d)):
        while d[i] - d[j] > window:
            j += 1
        best = max(best, i - j + 1)
    return best / len(d)


def window_null_pvalue(
    universe_days: np.ndarray, n: int, observed: float, draws: int, rng
) -> float:
    stats = np.array(
        [
            max_window_share(rng.choice(universe_days, size=n, replace=False))
            for _ in range(draws)
        ]
    )
    return float((1 + (stats >= observed).sum()) / (draws + 1))


def handle_skeleton(handle: str) -> str:
    return re.sub(r"\d+", "#", handle.lower())


def main() -> None:
    args = _lib.parse_args("Birth cohorts: creation-date + handle forensics")
    draws = 200 if args.sample else 1000
    rng = np.random.default_rng(7)
    con = db.connect()

    authors = (
        db.latest_authors(con)
        .df()[["platform_user_id", "handle", "created_at", "followers_count"]]
        .dropna(subset=["created_at"])
    )
    if args.sample:
        authors = authors.sample(min(args.sample, len(authors)), random_state=7)
    authors["created_day"] = _lib.to_days(authors["created_at"])
    universe_days = authors["created_day"].to_numpy()
    now_day = universe_days.max()
    universe_recent_share = float((now_day - universe_days <= RECENT_DAYS).mean())
    print(
        f"{len(authors)} dated authors; corpus share created in last "
        f"{RECENT_DAYS}d: {universe_recent_share:.3f}"
    )

    clusters = _lib.coordination_clusters()
    dated = clusters.merge(
        authors, left_on="author_id", right_on="platform_user_id"
    )
    rows = []
    for cid, grp in dated.groupby("cluster_id"):
        if len(grp) < MIN_CLUSTER_DATED:
            continue
        days = grp["created_day"].to_numpy()
        observed = max_window_share(days)
        rows.append(
            {
                "cluster_id": cid,
                "members_dated": len(grp),
                "cluster_size": int(grp["size"].iloc[0]),
                "channels": grp["channels"].iloc[0],
                "max_90d_creation_share": round(observed, 3),
                "p_vs_random_draw": window_null_pvalue(
                    universe_days, len(grp), observed, draws, rng
                ),
                "recent_share": round(
                    float((now_day - days <= RECENT_DAYS).mean()), 3
                ),
                "handles": ", ".join(grp["handle"].head(8)),
            }
        )
    cluster_birth = pd.DataFrame(rows).sort_values(
        "p_vs_random_draw"
    ) if rows else pd.DataFrame()
    _lib.show(cluster_birth, "coordination clusters by creation compactness")
    if not cluster_birth.empty:
        _lib.save(cluster_birth, "02_cluster_birth.csv")

    limit = f"USING SAMPLE {args.sample} ROWS (reservoir, 7)" if args.sample else ""
    pairs = con.sql(
        f"""
        WITH lp AS (
            SELECT author_id, hashtags
            FROM {db.posts_source('x')} {limit}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY collected_at DESC
            ) = 1
        )
        SELECT DISTINCT lower(tag) AS tag, author_id
        FROM lp, unnest(hashtags) AS t(tag)
        """
    ).df()
    tag_counts = pairs.groupby("tag")["author_id"].nunique()
    top_tags = tag_counts[tag_counts >= MIN_TAG_AUTHORS].nlargest(TOP_TAGS)
    tag_rows = []
    for tag in top_tags.index:
        grp = pairs[pairs["tag"] == tag].merge(
            authors, left_on="author_id", right_on="platform_user_id"
        )
        if len(grp) < MIN_TAG_AUTHORS:
            continue
        days = grp["created_day"].to_numpy()
        recent = float((now_day - days <= RECENT_DAYS).mean())
        se = np.sqrt(
            universe_recent_share * (1 - universe_recent_share) / len(grp)
        )
        digit_tail = float(
            grp["handle"].str.contains(r"\d{4,}$", regex=True).mean()
        )
        tag_rows.append(
            {
                "tag": tag,
                "authors_dated": len(grp),
                "recent_share": round(recent, 3),
                "recent_z": round((recent - universe_recent_share) / se, 2),
                "max_90d_creation_share": round(max_window_share(days), 3),
                "digit_tail_share": round(digit_tail, 3),
            }
        )
    tag_cohorts = pd.DataFrame(tag_rows).sort_values("recent_z", ascending=False)
    _lib.show(tag_cohorts, "hashtag author cohorts by recent-account skew")
    _lib.save(tag_cohorts, "02_hashtag_cohorts.csv")

    fam = authors.assign(skeleton=authors["handle"].map(handle_skeleton))
    fam = fam[fam["skeleton"].str.contains("#") & (fam["skeleton"].str.len() > 3)]
    fam_rows = []
    for skel, grp in fam.groupby("skeleton"):
        if len(grp) < 3:
            continue
        days = grp["created_day"].to_numpy()
        fam_rows.append(
            {
                "skeleton": skel,
                "accounts": len(grp),
                "creation_span_days": int(days.max() - days.min()),
                "handles": ", ".join(grp["handle"].head(8)),
            }
        )
    families = pd.DataFrame(fam_rows).sort_values(
        ["accounts", "creation_span_days"], ascending=[False, True]
    )
    _lib.show(families, "handle skeleton families (>=3 accounts)")
    if not families.empty:
        _lib.save(families, "02_handle_families.csv")

    _lib.print_caveats()


if __name__ == "__main__":
    main()

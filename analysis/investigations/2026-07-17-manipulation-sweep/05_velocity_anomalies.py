"""Engagement velocity anomalies.

- step jumps: posts whose like/view curve gains most of its total in one
  snapshot interval after a flat start (bought-engagement shape); only the
  ~1k metrics-tracked posts have curves
- ratio outliers (all posts, latest counts): likes far above what the
  author's follower count predicts (quantile-regression-free version: log-log
  residual vs follower-bucket median), and reply/like inversions

Tracked posts are the hot subset - selection is already biased toward
virality, so step shapes are flagged for reading, not scored against a null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _lib
from kma import db

MIN_SNAPSHOTS = 8
MIN_FINAL_LIKES = 50
STEP_SHARE = 0.6
MIN_LIKES_RATIO = 20


def main() -> None:
    args = _lib.parse_args("Velocity anomalies", default_sample=200)
    con = db.connect()

    limit = f"LIMIT {args.sample}" if args.sample else ""
    tracked = con.sql(
        f"""
        SELECT platform_post_id FROM {db.metrics_source('x')}
        GROUP BY 1 HAVING count(*) >= {MIN_SNAPSHOTS}
        AND max(like_count) >= {MIN_FINAL_LIKES} {limit}
        """
    ).df()["platform_post_id"]
    curves = con.sql(
        f"""
        SELECT platform_post_id, like_count, view_count, collected_at
        FROM {db.metrics_source('x')}
        WHERE platform_post_id IN (SELECT unnest(?))
        ORDER BY platform_post_id, collected_at
        """,
        params=[list(tracked)],
    ).df()
    print(f"{tracked.size} tracked posts with >={MIN_SNAPSHOTS} snapshots")

    rows = []
    for pid, grp in curves.groupby("platform_post_id"):
        likes = grp["like_count"].to_numpy(dtype=float)
        total = likes[-1] - likes[0]
        if total < MIN_FINAL_LIKES:
            continue
        deltas = np.diff(likes)
        step = deltas.max() / total if total > 0 else 0
        hours = (
            grp["collected_at"].iloc[-1] - grp["collected_at"].iloc[0]
        ).total_seconds() / 3600
        rows.append(
            {
                "platform_post_id": pid,
                "snapshots": len(grp),
                "span_h": round(hours, 1),
                "likes_gained": int(total),
                "max_step_share": round(float(step), 3),
                "step_at_snapshot": int(deltas.argmax()),
            }
        )
    steps = pd.DataFrame(rows).sort_values("max_step_share", ascending=False)
    jumps = steps[steps["max_step_share"] >= STEP_SHARE]
    meta = con.sql(
        f"""
        SELECT platform_post_id, author_handle, left(text, 100) AS text
        FROM {db.posts_source('x')}
        QUALIFY row_number() OVER (
            PARTITION BY platform_post_id ORDER BY collected_at DESC
        ) = 1
        """
    ).df()
    jumps = jumps.merge(meta, on="platform_post_id", how="left")
    _lib.show(jumps, f"step-jump like curves (one interval >= {STEP_SHARE:.0%})")
    _lib.save(steps, "05_like_curves.csv")
    _lib.save(jumps, "05_step_jumps.csv")

    posts = con.sql(
        f"""
        WITH lp AS (
            SELECT * FROM {db.posts_source('x')}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY collected_at DESC
            ) = 1
        ),
        la AS (
            SELECT * FROM {db.authors_source('x')}
            QUALIFY row_number() OVER (
                PARTITION BY platform_user_id ORDER BY collected_at DESC
            ) = 1
        )
        SELECT lp.platform_post_id, lp.author_handle, lp.like_count,
               lp.reply_count, lp.view_count, la.followers_count,
               left(lp.text, 90) AS text
        FROM lp JOIN la ON lp.author_id = la.platform_user_id
        WHERE NOT lp.is_repost AND lp.like_count >= {MIN_LIKES_RATIO}
          AND la.followers_count IS NOT NULL
        """
    ).df()
    posts["log_likes"] = np.log10(posts["like_count"].astype(float))
    posts["fbucket"] = pd.cut(
        np.log10(posts["followers_count"].astype(float).clip(lower=1)),
        bins=np.arange(0, 9, 0.5),
    )
    med = posts.groupby("fbucket", observed=True)["log_likes"].median()
    posts["excess_likes_dex"] = posts["log_likes"] - posts["fbucket"].map(med).astype(float)
    outliers = posts.sort_values("excess_likes_dex", ascending=False).head(200)
    _lib.show(
        outliers[["author_handle", "like_count", "followers_count",
                  "excess_likes_dex", "text"]],
        "likes far above follower-bucket median (dex = decades)",
    )
    _lib.save(
        outliers.drop(columns=["fbucket"]), "05_ratio_outliers.csv"
    )

    _lib.print_caveats()


if __name__ == "__main__":
    main()

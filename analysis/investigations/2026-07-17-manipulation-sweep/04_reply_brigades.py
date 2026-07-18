"""Reply brigading: who dogpiles whom, how fast, and with what tone.

- target concentration: reply volume per target account, share from
  low-follower repliers
- fast repliers: accounts habitually among the first to reply to captured
  parents; rank-percentile-vs-uniform null per account (a real audience
  replies at random positions in a thread; a watcher/bot is early always)
- dogpiles: single parent posts hit by many low-follower repliers in a tight
  window, with negative-sentiment share from persisted labels

Capture caveat: reply snowball prioritises busy conversations, so volumes are
biased toward hot threads; latency needs the parent captured (coverage
printed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _lib
from kma import db

LOW_FOLLOWERS = 100
MIN_REPLIES_FOR_RATE = 10
DOGPILE_MIN_REPLIERS = 10
DOGPILE_WINDOW_MIN = 120


def low_follower_share(s: pd.Series) -> float:
    s = s.dropna()
    return float((s < LOW_FOLLOWERS).mean()) if len(s) else float("nan")


def negative_share(s: pd.Series) -> float:
    s = s.dropna()
    return float((s == "negative").mean()) if len(s) else float("nan")


def main() -> None:
    args = _lib.parse_args("Reply brigading", default_sample=20000)
    con = db.connect()
    limit = f"LIMIT {args.sample}" if args.sample else ""

    replies = con.sql(
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
        ),
        ll AS (
            SELECT * FROM {db.labels_source('x')}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY labeled_at DESC
            ) = 1
        )
        SELECT r.platform_post_id, r.author_id, r.author_handle, r.created_at,
               r.in_reply_to_id, r.in_reply_to_user_id,
               ta.handle AS target_handle, ta.followers_count AS target_followers,
               ra.followers_count AS replier_followers,
               p.created_at AS parent_created_at, p.author_handle AS parent_handle,
               ll.sentiment
        FROM lp r
        LEFT JOIN la ta ON r.in_reply_to_user_id = ta.platform_user_id
        LEFT JOIN la ra ON r.author_id = ra.platform_user_id
        LEFT JOIN lp p ON r.in_reply_to_id = p.platform_post_id
        LEFT JOIN ll ON r.platform_post_id = ll.platform_post_id
        WHERE r.in_reply_to_id IS NOT NULL AND r.created_at IS NOT NULL
        {limit}
        """
    ).df()
    with_parent = replies["parent_created_at"].notna()
    print(
        f"{len(replies)} replies; parent captured for {with_parent.mean():.1%}"
    )

    t = replies.dropna(subset=["target_handle"])
    targets = t.groupby("target_handle").agg(
        n_replies=("platform_post_id", "size"),
        n_repliers=("author_id", "nunique"),
        low_follower_share=("replier_followers", low_follower_share),
        neg_share=("sentiment", negative_share),
        target_followers=("target_followers", "first"),
    ).sort_values("n_replies", ascending=False)
    _lib.show(targets.reset_index(), "reply volume by target account")
    _lib.save(targets.reset_index(), "04_targets.csv")

    lat = replies[with_parent].copy()
    lat["lag_min"] = (
        lat["created_at"] - lat["parent_created_at"]
    ).dt.total_seconds() / 60
    lat = lat[lat["lag_min"] >= 0]
    lat["rank_pct"] = lat.groupby("in_reply_to_id")["lag_min"].rank(pct=True)
    sizes = lat.groupby("in_reply_to_id")["platform_post_id"].transform("size")
    lat_multi = lat[sizes >= 3]
    fast = lat_multi.groupby("author_handle").agg(
        n=("rank_pct", "size"),
        mean_rank_pct=("rank_pct", "mean"),
        median_lag_min=("lag_min", "median"),
        targets=("target_handle", "nunique"),
    )
    fast = fast[fast["n"] >= MIN_REPLIES_FOR_RATE].copy()
    fast["z_vs_uniform"] = (fast["mean_rank_pct"] - 0.5) / (
        1 / np.sqrt(12 * fast["n"])
    )
    fast = fast.sort_values("z_vs_uniform")
    clusters_cache = _lib.coordination_clusters()
    id_by_handle = replies.drop_duplicates("author_handle").set_index(
        "author_handle"
    )["author_id"]
    fast["in_coord_cluster"] = fast.index.map(id_by_handle).isin(
        set(clusters_cache["author_id"])
    )
    _lib.show(
        fast.reset_index(),
        "habitual first repliers (mean thread-rank percentile vs uniform null)",
    )
    _lib.save(fast.reset_index(), "04_fast_repliers.csv")

    d = lat.groupby("in_reply_to_id").agg(
        parent_handle=("parent_handle", "first"),
        n_repliers=("author_id", "nunique"),
        low_follower_share=("replier_followers", low_follower_share),
        neg_share=("sentiment", negative_share),
        p90_lag_min=("lag_min", lambda s: float(s.quantile(0.9))),
    )
    dogpiles = d[
        (d["n_repliers"] >= DOGPILE_MIN_REPLIERS)
        & (d["p90_lag_min"] <= DOGPILE_WINDOW_MIN)
    ].sort_values("n_repliers", ascending=False)
    _lib.show(
        dogpiles.reset_index(),
        f"dogpiles (>= {DOGPILE_MIN_REPLIERS} repliers, p90 lag <= "
        f"{DOGPILE_WINDOW_MIN} min)",
    )
    _lib.save(dogpiles.reset_index(), "04_dogpiles.csv")

    _lib.print_caveats()


if __name__ == "__main__":
    main()

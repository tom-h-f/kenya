"""Account authenticity / bot triage scoring.

No ground-truth labels exist, so this ranks accounts by how bot-like their
profile and behaviour look; it is triage, not a trained classifier. Two lenses:
a transparent weighted heuristic (`heuristic_score`) and an unsupervised
isolation-forest anomaly rank (`authenticity_score`).

    from kma.db import connect
    from kma.authenticity import authenticity_score
    df = authenticity_score(connect()).sort_values("suspicion", ascending=False)
"""

from __future__ import annotations

import duckdb

from kma.db import authors_source, posts_source

# Heuristic weights: account age and follower/following ratio dominate, as they
# are hardest for a cheap account farm to fake. To be calibrated against data.
WEIGHTS = {
    "ratio": 0.30,   # more following than followers
    "age": 0.25,     # young account
    "rate": 0.15,    # high posting rate
    "dup": 0.15,     # repeats identical text
    "bio": 0.05,     # empty bio
    "img": 0.05,     # default profile image
    "handle": 0.05,  # digit-heavy handle
}

# Columns fed to the isolation forest. Skewed counts are log1p-scaled first.
ANOMALY_FEATURES = [
    "account_age_days", "followers_count", "following_count", "tweet_count",
    "listed_count", "followers_following_ratio", "tweet_rate", "listed_ratio",
    "duplicate_text_ratio", "reply_ratio", "repost_ratio", "quote_ratio",
    "handle_digit_ratio",
]
LOG_FEATURES = {
    "followers_count", "following_count", "tweet_count", "listed_count",
    "tweet_rate", "followers_following_ratio",
}


def _post_columns(con: duckdb.DuckDBPyConnection, platform: str) -> set[str]:
    return set(con.sql(f"SELECT * FROM {posts_source(platform)} LIMIT 0").columns)


def _features_sql(platform: str, has_quote: bool = True) -> str:
    authors, posts = authors_source(platform), posts_source(platform)
    quote_ratio = (
        "avg(COALESCE(is_quote, FALSE)::INT)" if has_quote else "0.0"
    )
    return f"""
    WITH la AS (
        SELECT * FROM {authors}
        QUALIFY row_number() OVER (
            PARTITION BY platform, platform_user_id ORDER BY collected_at DESC
        ) = 1
    ),
    lp AS (
        SELECT * FROM {posts}
        QUALIFY row_number() OVER (
            PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
        ) = 1
    ),
    beh AS (
        SELECT author_id,
            count(*) AS n_posts,
            avg((in_reply_to_id IS NOT NULL)::INT) AS reply_ratio,
            avg(is_repost::INT) AS repost_ratio,
            {quote_ratio} AS quote_ratio,
            1.0 - count(DISTINCT lower(trim(text))) * 1.0 / count(*) AS duplicate_text_ratio
        FROM lp GROUP BY author_id
    )
    SELECT
        la.platform_user_id, la.handle, la.display_name, la.bio, la.location,
        la.followers_count, la.following_count, la.tweet_count, la.listed_count,
        la.verified, la.blue, la.created_at, la.profile_image_url,
        greatest(date_diff('day', la.created_at, now()), 0) AS account_age_days,
        la.followers_count * 1.0 / greatest(la.following_count, 1) AS followers_following_ratio,
        la.tweet_count * 1.0 / greatest(date_diff('day', la.created_at, now()), 1) AS tweet_rate,
        la.listed_count * 1.0 / greatest(la.followers_count, 1) AS listed_ratio,
        (la.bio IS NULL OR la.bio = '') AS empty_bio,
        (la.profile_image_url ILIKE '%default_profile%') AS default_profile_image,
        len(regexp_replace(la.handle, '[^0-9]', '', 'g')) * 1.0
            / greatest(len(la.handle), 1) AS handle_digit_ratio,
        COALESCE(beh.n_posts, 0) AS n_posts,
        COALESCE(beh.reply_ratio, 0.0) AS reply_ratio,
        COALESCE(beh.repost_ratio, 0.0) AS repost_ratio,
        COALESCE(beh.quote_ratio, 0.0) AS quote_ratio,
        COALESCE(beh.duplicate_text_ratio, 0.0) AS duplicate_text_ratio
    FROM la LEFT JOIN beh ON la.platform_user_id = beh.author_id
    """


def author_features(con: duckdb.DuckDBPyConnection, platform: str = "x"):
    """One row per author with profile + behaviour features. Returns a relation."""
    return con.sql(_features_sql(platform, "is_quote" in _post_columns(con, platform)))


def _score_sql(platform: str, has_quote: bool = True) -> str:
    w = WEIGHTS
    return f"""
    WITH f AS ({_features_sql(platform, has_quote)})
    SELECT *,
        1.0 / (1.0 + followers_following_ratio) AS s_ratio,
        greatest(0.0, 1.0 - account_age_days / 365.0) AS s_age,
        least(1.0, tweet_rate / 50.0) AS s_rate,
        duplicate_text_ratio AS s_dup,
        empty_bio::INT AS s_bio,
        default_profile_image::INT AS s_img,
        handle_digit_ratio AS s_handle,
        {w['ratio']} * (1.0 / (1.0 + followers_following_ratio))
        + {w['age']} * greatest(0.0, 1.0 - account_age_days / 365.0)
        + {w['rate']} * least(1.0, tweet_rate / 50.0)
        + {w['dup']} * duplicate_text_ratio
        + {w['bio']} * empty_bio::INT
        + {w['img']} * default_profile_image::INT
        + {w['handle']} * handle_digit_ratio AS suspicion
    FROM f
    """


def heuristic_score(con: duckdb.DuckDBPyConnection, platform: str = "x"):
    """Features + per-signal sub-scores (`s_*`) + weighted `suspicion`. Relation."""
    return con.sql(_score_sql(platform, "is_quote" in _post_columns(con, platform)))


def authenticity_score(
    con: duckdb.DuckDBPyConnection,
    platform: str = "x",
    with_anomaly: bool = True,
    contamination: float = 0.1,
    seed: int = 0,
):
    """Heuristic score as a pandas DataFrame, plus an isolation-forest
    `anomaly_score` (higher = more anomalous) and its percentile `anomaly_rank`."""
    df = con.sql(_score_sql(platform, "is_quote" in _post_columns(con, platform))).df()
    if with_anomaly and len(df) > 50:
        import numpy as np
        from sklearn.ensemble import IsolationForest

        x = df[ANOMALY_FEATURES].astype("float64").copy()
        for col in LOG_FEATURES:
            x[col] = np.log1p(x[col].clip(lower=0))
        x = x.fillna(0.0)
        clf = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=seed
        )
        clf.fit(x)
        df["anomaly_score"] = -clf.score_samples(x)
        df["anomaly_rank"] = df["anomaly_score"].rank(pct=True)
    return df

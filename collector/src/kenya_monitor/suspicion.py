"""Heuristic suspicion ranking for collector-side target selection.

Mirrors the SQL heuristic in analysis `kma.authenticity` (no sklearn dep).
"""

from __future__ import annotations

import duckdb

WEIGHTS = {
    "ratio": 0.30,
    "age": 0.25,
    "rate": 0.15,
    "dup": 0.15,
    "bio": 0.05,
    "img": 0.05,
    "handle": 0.05,
}


def _post_columns(con: duckdb.DuckDBPyConnection, posts_view: str) -> set[str]:
    return set(con.sql(f"SELECT * FROM {posts_view} LIMIT 0").columns)


def _score_sql(authors_view: str, posts_view: str, has_quote: bool) -> str:
    w = WEIGHTS
    quote_ratio = "avg(COALESCE(is_quote, FALSE)::INT)" if has_quote else "0.0"
    return f"""
    WITH la AS (
        SELECT * FROM {authors_view}
        QUALIFY row_number() OVER (
            PARTITION BY platform, platform_user_id ORDER BY collected_at DESC
        ) = 1
    ),
    lp AS (
        SELECT * FROM {posts_view}
        QUALIFY row_number() OVER (
            PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
        ) = 1
    ),
    beh AS (
        SELECT author_id,
            count(*) AS n_posts,
            1.0 - count(DISTINCT lower(trim(text))) * 1.0 / count(*) AS duplicate_text_ratio
        FROM lp
        WHERE text IS NOT NULL
        GROUP BY author_id
    )
    SELECT
        la.handle,
        greatest(date_diff('day', la.created_at, now()), 0) AS account_age_days,
        la.followers_count * 1.0 / greatest(la.following_count, 1) AS followers_following_ratio,
        la.tweet_count * 1.0 / greatest(date_diff('day', la.created_at, now()), 1) AS tweet_rate,
        (la.bio IS NULL OR la.bio = '') AS empty_bio,
        (la.profile_image_url ILIKE '%default_profile%') AS default_profile_image,
        len(regexp_replace(la.handle, '[^0-9]', '', 'g')) * 1.0
            / greatest(len(la.handle), 1) AS handle_digit_ratio,
        COALESCE(beh.duplicate_text_ratio, 0.0) AS duplicate_text_ratio,
        {w['ratio']} * (1.0 / (1.0 + la.followers_count * 1.0 / greatest(la.following_count, 1)))
        + {w['age']} * greatest(0.0, 1.0 - greatest(date_diff('day', la.created_at, now()), 0) / 365.0)
        + {w['rate']} * least(1.0, la.tweet_count * 1.0
            / greatest(date_diff('day', la.created_at, now()), 1) / 50.0)
        + {w['dup']} * COALESCE(beh.duplicate_text_ratio, 0.0)
        + {w['bio']} * (la.bio IS NULL OR la.bio = '')::INT
        + {w['img']} * (la.profile_image_url ILIKE '%default_profile%')::INT
        + {w['handle']} * (len(regexp_replace(la.handle, '[^0-9]', '', 'g')) * 1.0
            / greatest(len(la.handle), 1)) AS suspicion
    FROM la
    LEFT JOIN beh ON la.platform_user_id = beh.author_id
    WHERE la.handle IS NOT NULL AND trim(la.handle) != ''
    """


def top_suspicious_handles(
    con: duckdb.DuckDBPyConnection,
    authors_view: str,
    posts_view: str,
    n: int = 1000,
) -> list[str]:
    """Return handles of the top `n` accounts by heuristic suspicion."""
    has_quote = "is_quote" in _post_columns(con, posts_view)
    rows = con.sql(
        f"""
        SELECT handle FROM ({_score_sql(authors_view, posts_view, has_quote)})
        ORDER BY suspicion DESC NULLS LAST
        LIMIT {int(n)}
        """
    ).fetchall()
    return [r[0] for r in rows]

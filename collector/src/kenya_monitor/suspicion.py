"""Heuristic suspicion ranking for collector-side target selection.

Mirrors the SQL heuristic in analysis `kma.authenticity` (no sklearn dep).

Every query here is bounded three ways, because both inputs grow without bound
and the collector runs under a 600MB DuckDB cap (`COLLECTOR_MEMORY_LIMIT`):

1. **Windowed.** Posts are pruned on `dt` before anything reads them.
2. **Projected.** Only the columns the score needs are carried; never `SELECT *`,
   which drags `text` and every author field through the operator.
3. **Spillable.** Hash aggregates only - no window functions over the corpus, no
   `count(DISTINCT ...)`. DuckDB spills those to `temp_directory`; a window over
   2M wide rows and a DISTINCT hash set are what it cannot spill.

Measured 2026-09-01, when the unbounded form died: 2,165,609 post rows /
898,170 distinct ids / 282,675 distinct authors, and 5,005,226 author rows /
1,195,614 distinct users. Both seed paths went through here - `hate_signal`
embeds this as its `susp` CTE - so one unbounded query took out hate seeding and
its fallback together.
"""

from __future__ import annotations

import duckdb

from kenya_monitor.config import SUSPICION_LOOKBACK_DAYS

WEIGHTS = {
    "ratio": 0.30,
    "age": 0.25,
    "rate": 0.15,
    "dup": 0.15,
    "bio": 0.05,
    "img": 0.05,
    "handle": 0.05,
}

# Carried through the post dedup. `text` is here only for the duplicate ratio.
_POST_COLS = ("author_id", "text")

# Carried through the author dedup - the inputs to the score, nothing else.
_AUTHOR_COLS = (
    "handle",
    "created_at",
    "followers_count",
    "following_count",
    "tweet_count",
    "bio",
    "profile_image_url",
)


def _post_columns(con: duckdb.DuckDBPyConnection, posts_view: str) -> set[str]:
    return set(con.sql(f"SELECT * FROM {posts_view} LIMIT 0").columns)


def _struct(cols: tuple[str, ...]) -> str:
    return "struct_pack(" + ", ".join(f"{c} := {c}" for c in cols) + ")"


def _score_sql(
    authors_view: str,
    posts_view: str,
    has_quote: bool,
    lookback_days: int = SUSPICION_LOOKBACK_DAYS,
    has_dt: bool = True,
) -> str:
    w = WEIGHTS
    # `has_quote` probes for the column but the quote ratio it guarded was
    # computed into a local and never interpolated into the SQL below - a signal
    # that was designed, plumbed and then never wired to anything. Removed
    # rather than connected: WEIGHTS already sums to 1.00, so adding it would
    # silently reweight every other component and change collector-side seed
    # ranking with no measurement behind it. Give it a weight deliberately if
    # you want it back. The probe stays so the signature is unchanged for
    # callers and the column check is still available.

    # Callers register bare tables in tests, and `dt` only exists on the hive
    # partitioning of the real R2 views. Probed rather than assumed so an
    # unpartitioned view still ranks instead of failing to bind.
    window = (
        f"WHERE dt >= current_date - INTERVAL {int(lookback_days)} DAY"
        if has_dt
        else ""
    )
    # `arg_max(struct, collected_at)` replaces `QUALIFY row_number() OVER
    # (PARTITION BY ... ORDER BY collected_at DESC)`. Same row wins, but it is a
    # hash aggregate rather than a sort over the whole corpus. The struct keeps
    # whole-row semantics: unpacking column by column would let a NULL field on
    # the newest row fall back to an older row's value, which for `bio` would
    # flip the `empty_bio` component.
    return f"""
    WITH lp AS (
        SELECT
            platform, platform_post_id,
            arg_max({_struct(_POST_COLS)}, collected_at) AS r
        FROM {posts_view}
        {window}
        GROUP BY platform, platform_post_id
    ),
    p AS (
        SELECT r.author_id AS author_id, r.text AS text FROM lp
    ),
    -- Two stages instead of `count(DISTINCT lower(trim(text)))`: the inner
    -- group collapses repeats, the outer counts them. Identical result, no
    -- per-author hash set of full post text held in memory.
    txt AS (
        SELECT author_id, lower(trim(text)) AS t, count(*) AS n
        FROM p WHERE text IS NOT NULL
        GROUP BY author_id, t
    ),
    beh AS (
        SELECT author_id,
            sum(n) AS n_posts,
            1.0 - count(*) * 1.0 / sum(n) AS duplicate_text_ratio
        FROM txt GROUP BY author_id
    ),
    -- Only authors seen posting in the window can be ranked. The authors prefix
    -- carries every account ever collected (1.2M distinct against 283k that
    -- appear in posts); ranking one that has posted nothing recent would spend
    -- a crawl seed on a dormant account anyway.
    seen AS (
        SELECT DISTINCT author_id FROM p WHERE author_id IS NOT NULL
    ),
    la AS (
        SELECT
            platform, platform_user_id,
            arg_max({_struct(_AUTHOR_COLS)}, collected_at) AS r
        FROM {authors_view}
        WHERE platform_user_id IN (SELECT author_id FROM seen)
        GROUP BY platform, platform_user_id
    ),
    a AS (
        SELECT
            platform_user_id,
            r.handle AS handle,
            r.created_at AS created_at,
            r.followers_count AS followers_count,
            r.following_count AS following_count,
            r.tweet_count AS tweet_count,
            r.bio AS bio,
            r.profile_image_url AS profile_image_url
        FROM la
    )
    SELECT
        a.handle,
        greatest(date_diff('day', a.created_at, now()), 0) AS account_age_days,
        a.followers_count * 1.0 / greatest(a.following_count, 1) AS followers_following_ratio,
        a.tweet_count * 1.0 / greatest(date_diff('day', a.created_at, now()), 1) AS tweet_rate,
        (a.bio IS NULL OR a.bio = '') AS empty_bio,
        (a.profile_image_url ILIKE '%default_profile%') AS default_profile_image,
        len(regexp_replace(a.handle, '[^0-9]', '', 'g')) * 1.0
            / greatest(len(a.handle), 1) AS handle_digit_ratio,
        COALESCE(beh.duplicate_text_ratio, 0.0) AS duplicate_text_ratio,
        {w['ratio']} * (1.0 / (1.0 + a.followers_count * 1.0 / greatest(a.following_count, 1)))
        + {w['age']} * greatest(0.0, 1.0 - greatest(date_diff('day', a.created_at, now()), 0) / 365.0)
        + {w['rate']} * least(1.0, a.tweet_count * 1.0
            / greatest(date_diff('day', a.created_at, now()), 1) / 50.0)
        + {w['dup']} * COALESCE(beh.duplicate_text_ratio, 0.0)
        + {w['bio']} * (a.bio IS NULL OR a.bio = '')::INT
        + {w['img']} * (a.profile_image_url ILIKE '%default_profile%')::INT
        + {w['handle']} * (len(regexp_replace(a.handle, '[^0-9]', '', 'g')) * 1.0
            / greatest(len(a.handle), 1)) AS suspicion
    FROM a
    LEFT JOIN beh ON a.platform_user_id = beh.author_id
    WHERE a.handle IS NOT NULL AND trim(a.handle) != ''
    """


def top_suspicious_handles(
    con: duckdb.DuckDBPyConnection,
    authors_view: str,
    posts_view: str,
    n: int = 1000,
    lookback_days: int = SUSPICION_LOOKBACK_DAYS,
) -> list[str]:
    """Return handles of the top `n` accounts by heuristic suspicion.

    Ranks only accounts that posted within `lookback_days`; see the module
    docstring for why the corpus-wide form cannot run under the collector's
    memory cap."""
    cols = _post_columns(con, posts_view)
    has_quote = "is_quote" in cols
    rows = con.sql(
        f"""
        SELECT handle FROM (
            {_score_sql(authors_view, posts_view, has_quote, lookback_days, "dt" in cols)}
        )
        ORDER BY suspicion DESC NULLS LAST
        LIMIT {int(n)}
        """
    ).fetchall()
    return [r[0] for r in rows]

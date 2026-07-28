"""Turn per-post hate scores into network seeds, in pure SQL.

Mirrors the shape of `suspicion.py`: view expressions are passed as strings so
tests can substitute local temp tables, and there is no sklearn / transformers
dependency. The collector runs no models - it reads what `kma.hatespeech`
already scored into the R2 `hatespeech/` prefix.

**One toxic post is not a network.** A single account ranting is a loud
individual; a hate network is accounts amplifying each other's toxic material
and swarming the same targets. So the score is deliberately dominated by
coordination shape rather than by toxicity level:

    co_amplify + brigade  = 0.45   how shared the behaviour is
    toxic_rate_lb         = 0.25   how toxic, Wilson-corrected
    persistence           = 0.15   how sustained
    suspicion             = 0.15   ordinary bot-likeness (reused verbatim)

Three hard floors run before any scoring, so thin evidence never ranks at all.
"""

from __future__ import annotations

import logging

import duckdb

from kenya_monitor.suspicion import _post_columns, _score_sql

log = logging.getLogger("kenya_monitor")

HATE_SEED_WEIGHTS = {
    "co_amplify": 0.25,
    "toxic_rate_lb": 0.25,
    "brigade": 0.20,
    "persistence": 0.15,
    "suspicion": 0.15,
}

HATE_LOOKBACK_DAYS = 14
HATE_MIN_POSTS = 10  # too few posts to estimate a rate from
HATE_MIN_TOXIC_POSTS = 3  # one bad day is not a pattern
HATE_SEED_MIN_PEERS = 2  # nobody else amplified it -> not a network
BRIGADE_MIN_AUTHORS = 3  # distinct toxic repliers before a thread is a pile-on
SCORES_STALE_HOURS = 24


def _toxic_expr(cols: set[str]) -> str:
    """Toxicity predicate over the hatespeech row.

    `hate_flag` is the deploy rule (`p_hate >= 0.28`), deliberately an explicit
    threshold rather than argmax, because the taxonomy pushes coded menace
    without a protected-group target into `offensive`. Both are read together.
    `in_kenya_scope` / `coded_suspect` appear once the measure columns are
    persisted; until then this degrades to unscoped toxicity."""
    toxic = "(h.label <> 'neither' OR COALESCE(h.hate_flag, FALSE))"
    if "coded_suspect" in cols:
        toxic = f"({toxic} OR COALESCE(h.coded_suspect, FALSE))"
    if "in_kenya_scope" in cols:
        toxic = f"({toxic} AND COALESCE(h.in_kenya_scope, TRUE))"
    return toxic


def scores_are_stale(
    con: duckdb.DuckDBPyConnection, hatespeech_view: str, max_age_hours: int = SCORES_STALE_HOURS
) -> bool:
    """True when the enrich worker has stopped writing. Selecting targets from
    stale scores silently freezes the seed set, so callers fall back."""
    try:
        row = con.sql(
            f"""
            SELECT max(scored_at) IS NULL
                OR max(scored_at) < now() - INTERVAL {int(max_age_hours)} HOUR
            FROM {hatespeech_view}
            """
        ).fetchone()
    except duckdb.Error:
        return True
    return bool(row[0])


def _hate_account_sql(
    hatespeech_view: str,
    posts_view: str,
    authors_view: str,
    engagements_view: str | None,
    toxic: str,
    lookback_days: int,
    min_posts: int,
    min_toxic: int,
    min_peers: int,
    has_quote: bool,
) -> str:
    # Retweet incidence unions collected retweets with the snowballed retweeter
    # census, so co-amplification does not depend on us happening to have
    # collected each retweet as a post.
    census = (
        f"""
        UNION
        SELECT platform_user_id AS author_id, platform_post_id AS obj
        FROM {engagements_view} WHERE kind = 'retweet'
        """
        if engagements_view
        else ""
    )
    return f"""
    WITH lp AS (
        SELECT * FROM {posts_view}
        QUALIFY row_number() OVER (
            PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
        ) = 1
    ), lh AS (
        SELECT * FROM {hatespeech_view}
        QUALIFY row_number() OVER (
            PARTITION BY platform_post_id ORDER BY scored_at DESC
        ) = 1
    ), flag AS (
        SELECT h.platform_post_id, {toxic} AS toxic FROM lh h
    ), recent AS (
        SELECT lp.*, COALESCE(f.toxic, FALSE) AS toxic
        FROM lp LEFT JOIN flag f USING (platform_post_id)
        WHERE lp.created_at > now() - INTERVAL {int(lookback_days)} DAY
    ), acct AS (
        SELECT author_id,
               count(*) AS n_posts,
               count(*) FILTER (WHERE toxic) AS n_toxic,
               count(DISTINCT CASE WHEN toxic THEN date_trunc('day', created_at) END)
                   AS toxic_days,
               count(DISTINCT date_trunc('day', created_at)) AS active_days
        FROM recent GROUP BY author_id
    ), rated AS (
        SELECT *,
            -- Wilson lower bound: 1/1 must not outrank 40/300.
            (n_toxic * 1.0 / n_posts + 1.9208 / (2 * n_posts)
             - 1.96 * sqrt((n_toxic * 1.0 / n_posts * (1 - n_toxic * 1.0 / n_posts)
                            + 0.9604 / n_posts) / n_posts))
            / (1 + 3.8416 / n_posts) AS toxic_rate_lb,
            toxic_days * 1.0 / greatest(active_days, 1) AS persistence
        FROM acct
        WHERE n_posts >= {int(min_posts)} AND n_toxic >= {int(min_toxic)}
    ), toxic_obj AS (
        SELECT platform_post_id FROM flag WHERE toxic
    ), rt AS (
        SELECT author_id, repost_of_id AS obj FROM lp WHERE repost_of_id IS NOT NULL
        {census}
    ), trt AS (
        SELECT rt.* FROM rt JOIN toxic_obj ON rt.obj = toxic_obj.platform_post_id
    ), amp AS (
        SELECT a.author_id,
               count(DISTINCT b.author_id) AS n_cotoxic_peers,
               count(DISTINCT a.obj) AS n_toxic_objects_amplified
        FROM trt a JOIN trt b USING (obj)
        WHERE a.author_id <> b.author_id
        GROUP BY a.author_id
    ), toxic_replies AS (
        SELECT conversation_id, author_id
        FROM recent
        WHERE toxic AND in_reply_to_id IS NOT NULL AND conversation_id IS NOT NULL
    ), brigades AS (
        SELECT conversation_id FROM toxic_replies
        GROUP BY conversation_id
        HAVING count(DISTINCT author_id) >= {int(BRIGADE_MIN_AUTHORS)}
    ), brig AS (
        SELECT author_id, count(DISTINCT conversation_id) AS n_brigade_convs
        FROM toxic_replies WHERE conversation_id IN (SELECT conversation_id FROM brigades)
        GROUP BY author_id
    ), susp AS (
        SELECT * FROM ({_score_sql(authors_view, posts_view, has_quote)})
    ), la AS (
        SELECT * FROM {authors_view}
        QUALIFY row_number() OVER (
            PARTITION BY platform, platform_user_id ORDER BY collected_at DESC
        ) = 1
    ), joined AS (
        SELECT
            la.handle,
            r.author_id, r.n_posts, r.n_toxic, r.toxic_rate_lb, r.persistence,
            COALESCE(amp.n_cotoxic_peers, 0) AS n_cotoxic_peers,
            COALESCE(amp.n_toxic_objects_amplified, 0) AS n_toxic_objects_amplified,
            COALESCE(brig.n_brigade_convs, 0) AS n_brigade_convs,
            COALESCE(susp.suspicion, 0.0) AS suspicion
        FROM rated r
        JOIN la ON r.author_id = la.platform_user_id
        LEFT JOIN amp USING (author_id)
        LEFT JOIN brig USING (author_id)
        LEFT JOIN susp ON susp.handle = la.handle
        WHERE la.handle IS NOT NULL AND trim(la.handle) <> ''
          AND COALESCE(amp.n_cotoxic_peers, 0) >= {int(min_peers)}
    )
    SELECT *,
        -- Percentile-rank each component so one heavy-tailed count cannot
        -- dominate, then weight. Counts are log1p'd first.
        {HATE_SEED_WEIGHTS["co_amplify"]} * percent_rank() OVER (
            ORDER BY ln(1 + n_cotoxic_peers) + ln(1 + n_toxic_objects_amplified))
      + {HATE_SEED_WEIGHTS["toxic_rate_lb"]} * percent_rank() OVER (ORDER BY toxic_rate_lb)
      + {HATE_SEED_WEIGHTS["brigade"]} * percent_rank() OVER (ORDER BY ln(1 + n_brigade_convs))
      + {HATE_SEED_WEIGHTS["persistence"]} * percent_rank() OVER (ORDER BY persistence)
      + {HATE_SEED_WEIGHTS["suspicion"]} * percent_rank() OVER (ORDER BY suspicion)
        AS hate_seed_score
    FROM joined
    """


def hate_accounts(
    con: duckdb.DuckDBPyConnection,
    hatespeech_view: str,
    posts_view: str,
    authors_view: str,
    engagements_view: str | None = None,
    lookback_days: int = HATE_LOOKBACK_DAYS,
    min_posts: int = HATE_MIN_POSTS,
    min_toxic: int = HATE_MIN_TOXIC_POSTS,
    min_peers: int = HATE_SEED_MIN_PEERS,
    n: int = 50,
) -> list[dict]:
    """Accounts ranked as hate-network seeds, strongest first, with the evidence
    that put them there. Returns [] when nothing has been scored yet."""
    try:
        hate_cols = set(con.sql(f"SELECT * FROM {hatespeech_view} LIMIT 0").columns)
    except duckdb.Error:
        log.info("hate_signal: no hatespeech scores available yet")
        return []
    has_quote = "is_quote" in _post_columns(con, posts_view)
    sql = _hate_account_sql(
        hatespeech_view, posts_view, authors_view, engagements_view,
        _toxic_expr(hate_cols), lookback_days, min_posts, min_toxic, min_peers, has_quote,
    )
    try:
        rel = con.sql(f"SELECT * FROM ({sql}) ORDER BY hate_seed_score DESC LIMIT {int(n)}")
        cols = rel.columns
        return [dict(zip(cols, row)) for row in rel.fetchall()]
    except duckdb.Error:
        log.exception("hate_signal: seed query failed")
        return []


def top_hate_seeds(
    con: duckdb.DuckDBPyConnection,
    hatespeech_view: str,
    posts_view: str,
    authors_view: str,
    engagements_view: str | None = None,
    n: int = 20,
    **kwargs,
) -> list[str]:
    """Handles only - a drop-in for `suspicion.top_suspicious_handles`."""
    return [r["handle"] for r in hate_accounts(
        con, hatespeech_view, posts_view, authors_view, engagements_view, n=n, **kwargs
    )]


# Words too common to be register markers. Kept deliberately small: the
# corpus-frequency floor below does most of the work, and an over-eager
# stoplist would suppress exactly the coded terms we are hunting.
STOPWORDS = {
    # English
    "the", "and", "for", "that", "this", "with", "from", "have", "has", "had",
    "not", "are", "was", "were", "you", "your", "they", "them", "their", "there",
    "will", "would", "should", "could", "can", "but", "all", "any", "our", "who",
    "what", "when", "where", "why", "how", "his", "her", "its", "been", "being",
    "just", "only", "more", "most", "some", "such", "than", "then", "these",
    "those", "about", "into", "over", "after", "before", "now", "one", "two",
    "get", "got", "out", "off", "own", "too", "very", "via", "amp", "rts", "http",
    "https", "com", "www",
    # Swahili / Sheng function words
    "wa", "ya", "na", "ni", "kwa", "za", "la", "cha", "vya", "pa", "kwenye",
    "hii", "hiyo", "hizo", "huyu", "huyo", "hawa", "yule", "wale", "sana",
    "tu", "pia", "lakini", "sasa", "bado", "kama", "ndio", "ndiyo", "hapa",
    "pale", "kule", "yako", "yangu", "yake", "yetu", "wetu", "wao", "mimi",
    "wewe", "yeye", "sisi", "nyinyi", "hata", "kila", "wote", "watu", "mtu",
    "siku", "leo", "kesho", "juzi", "nini", "nani", "gani", "vile", "hivyo",
    "ati", "eti", "haya", "sawa", "asante", "pole", "kwanza", "tena", "zaidi",
    # Generic connectives and discourse filler that survived the first live run
    # (2026-07-28) purely on small-count novelty noise.
    "since", "trying", "without", "haha", "can't", "cant", "dont", "don't",
    "east", "west", "north", "south", "down", "end", "made", "make", "know",
    "think", "said", "say", "says", "see", "even", "still", "back", "good",
    "well", "want", "need", "like", "time", "years", "year", "first", "last",
    "many", "much", "every", "other", "same", "long", "next", "come", "going",
    "let", "put", "take", "give", "look", "thing", "things", "man", "people",
}

# A term must be over-represented in the cohort before its temporal rise means
# anything. Without this, novelty ranks small-count sampling noise: the first
# live run surfaced terms with lift ~0.15 (six times UNDER-represented in the
# cohort) simply because their handful of occurrences happened to be recent.
MIN_LIFT = 1.5


def mine_terms(
    con: duckdb.DuckDBPyConnection,
    posts_view: str,
    authors_view: str,
    cohort_handles: list[str],
    lookback_days: int = HATE_LOOKBACK_DAYS,
    min_count: int = 8,
    min_authors: int = 3,
    min_lift: float = MIN_LIFT,
    novelty_days: int = 3,
    extra_stopwords: set[str] | None = None,
    n: int = 50,
) -> list[dict]:
    """Terms over-represented in a cohort's vocabulary versus the whole corpus.

    This is how the 2026 coded register gets found at all. The classifier
    cannot see it - on known-coded posts its mean `p_hate` drops - so the
    cohort is located by explicit toxicity and coordination, and then its
    *language* is mined by contrast.

    Two-stage on purpose. `min_lift` **gates**: a term must actually be
    over-represented in the cohort to be a candidate at all. `novelty` (rise in
    cohort share over the last `novelty_days` versus the preceding window) then
    **ranks** the survivors, because new coded terms appear suddenly, which is
    far more specific than a term merely being cohort-flavoured. Ranking by
    novelty alone does not work: cohort counts are small, so the top of that
    list fills with sampling noise on terms the cohort barely uses.

    `min_authors` is the other guard that matters: one account repeating a word
    is a verbal tic, not a register. Output is a candidate list for human
    review, never a promotion - see HATE_MINE_AUTOPROMOTE.
    """
    if not cohort_handles:
        return []
    stop = STOPWORDS | (extra_stopwords or set())
    stop_sql = ", ".join("'" + w.replace("'", "''") + "'" for w in sorted(stop))
    handles_sql = ", ".join("'" + h.replace("'", "''") + "'" for h in cohort_handles)
    try:
        rel = con.sql(
            f"""
            WITH lp AS (
                SELECT * FROM {posts_view}
                QUALIFY row_number() OVER (
                    PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
                ) = 1
            ), la AS (
                SELECT * FROM {authors_view}
                QUALIFY row_number() OVER (
                    PARTITION BY platform, platform_user_id ORDER BY collected_at DESC
                ) = 1
            ), cohort AS (
                SELECT DISTINCT platform_user_id AS author_id FROM la
                WHERE lower(handle) IN ({handles_sql.lower()})
            ), recent AS (
                SELECT p.platform_post_id, p.author_id, p.text, p.created_at,
                       (c.author_id IS NOT NULL) AS in_cohort,
                       (p.created_at > now() - INTERVAL {int(novelty_days)} DAY) AS is_new
                FROM lp p LEFT JOIN cohort c USING (author_id)
                WHERE p.text IS NOT NULL
                  AND p.created_at > now() - INTERVAL {int(lookback_days)} DAY
            ), toks AS (
                SELECT author_id, in_cohort, is_new,
                       unnest(regexp_extract_all(lower(text), '[\\p{{L}}][\\p{{L}}''-]{{2,}}')) AS term
                FROM recent
            ), kept AS (
                SELECT * FROM toks WHERE term NOT IN ({stop_sql})
            ), agg AS (
                SELECT term,
                    count(*) FILTER (WHERE in_cohort) AS c_in,
                    count(DISTINCT author_id) FILTER (WHERE in_cohort) AS a_in,
                    count(*) AS c_all,
                    count(*) FILTER (WHERE in_cohort AND is_new) AS c_in_new,
                    count(*) FILTER (WHERE in_cohort AND NOT is_new) AS c_in_old
                FROM kept GROUP BY term
            ), tot AS (
                SELECT sum(c_in) AS n_in, sum(c_all) AS n_all, count(*) AS v,
                       sum(c_in_new) AS n_in_new, sum(c_in_old) AS n_in_old
                FROM agg
            )
            SELECT * FROM (
                SELECT a.term, a.c_in, a.a_in, a.c_all,
                    ((a.c_in + 0.5) / (t.n_in + 0.5 * t.v))
                        / ((a.c_all + 0.5) / (t.n_all + 0.5 * t.v)) AS lift,
                    ((a.c_in_new + 0.5) / (t.n_in_new + 0.5 * t.v))
                        / ((a.c_in_old + 0.5) / (t.n_in_old + 0.5 * t.v)) AS novelty
                FROM agg a, tot t
                WHERE a.c_in >= {int(min_count)} AND a.a_in >= {int(min_authors)}
            )
            WHERE lift >= {float(min_lift)}
            ORDER BY novelty DESC, lift DESC
            LIMIT {int(n)}
            """
        )
        cols = rel.columns
        return [dict(zip(cols, row)) for row in rel.fetchall()]
    except duckdb.Error:
        log.exception("hate_signal: term mining failed")
        return []


def term_examples(
    con: duckdb.DuckDBPyConnection, posts_view: str, term: str, limit: int = 3
) -> list[str]:
    """Example posts containing a mined term, so a human reviewing a candidate
    sees the usage rather than just a score."""
    try:
        return [
            r[0]
            for r in con.sql(
                f"""
                SELECT text FROM {posts_view}
                WHERE text IS NOT NULL
                  AND regexp_matches(lower(text), '\\b' || lower(?) || '\\b')
                LIMIT {int(limit)}
                """,
                params=[term],
            ).fetchall()
        ]
    except duckdb.Error:
        return []


def hot_toxic_objects(
    con: duckdb.DuckDBPyConnection,
    hatespeech_view: str,
    posts_view: str,
    lookback_days: int = 2,
    top_retweeted: int = 15,
    top_conversations: int = 10,
) -> tuple[list[str], list[str], list[str]]:
    """Snowball targets ranked by toxic density rather than raw engagement.

    Same (retweeted_ids, conversation_ids, missing_ids) shape as
    `runner.hot_objects`, so it drops straight into `collect_snowball`. The
    third element is always empty: hydration of referenced-but-uncollected
    originals stays a baseline concern."""
    try:
        hate_cols = set(con.sql(f"SELECT * FROM {hatespeech_view} LIMIT 0").columns)
    except duckdb.Error:
        return [], [], []
    toxic = _toxic_expr(hate_cols)
    base = f"""
        WITH lp AS (
            SELECT * FROM {posts_view}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
            ) = 1
        ), lh AS (
            SELECT * FROM {hatespeech_view}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY scored_at DESC
            ) = 1
        ), flag AS (
            SELECT h.platform_post_id, {toxic} AS toxic, h.p_hate FROM lh h
        ), recent AS (
            SELECT lp.*, COALESCE(f.toxic, FALSE) AS toxic, f.p_hate
            FROM lp LEFT JOIN flag f USING (platform_post_id)
            WHERE lp.created_at > now() - INTERVAL {int(lookback_days)} DAY
        )
    """
    try:
        retweeted = [
            r[0]
            for r in con.sql(
                f"""
                {base}
                SELECT platform_post_id FROM recent
                WHERE toxic AND COALESCE(repost_count, 0) > 0 AND NOT COALESCE(is_repost, FALSE)
                ORDER BY repost_count DESC NULLS LAST, p_hate DESC NULLS LAST
                LIMIT {int(top_retweeted)}
                """
            ).fetchall()
        ]
        conversations = [
            r[0]
            for r in con.sql(
                f"""
                {base}
                SELECT conversation_id FROM recent
                WHERE conversation_id IS NOT NULL
                GROUP BY conversation_id
                HAVING count(*) FILTER (WHERE toxic) >= {int(BRIGADE_MIN_AUTHORS)}
                ORDER BY count(*) FILTER (WHERE toxic) DESC, count(*) DESC
                LIMIT {int(top_conversations)}
                """
            ).fetchall()
        ]
    except duckdb.Error:
        log.exception("hate_signal: hot object query failed")
        return [], [], []
    return retweeted, conversations, []

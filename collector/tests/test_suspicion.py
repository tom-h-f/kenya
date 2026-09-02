from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import duckdb
import pyarrow as pa

from kenya_monitor.suspicion import top_suspicious_handles

NOW = datetime.now(timezone.utc)


def test_top_suspicious_handles_ranks_young_high_ratio_first():
    authors = pa.table(
        {
            "platform": pa.array(["x", "x"], type=pa.string()),
            "platform_user_id": pa.array(["a1", "a2"], type=pa.string()),
            "handle": pa.array(["suspect", "realnews"], type=pa.string()),
            "bio": pa.array(["", "Kenya elections reporter"], type=pa.string()),
            "followers_count": pa.array([10, 500_000], type=pa.int64()),
            "following_count": pa.array([2000, 400], type=pa.int64()),
            "tweet_count": pa.array([500, 10_000], type=pa.int64()),
            "listed_count": pa.array([0, 1000], type=pa.int64()),
            "verified": pa.array([False, True], type=pa.bool_()),
            "blue": pa.array([False, False], type=pa.bool_()),
            "created_at": pa.array([NOW, NOW], type=pa.timestamp("us", tz="UTC")),
            "profile_image_url": pa.array(
                ["https://x.com/default_profile.png", "https://x.com/pic.jpg"],
                type=pa.string(),
            ),
            "collected_at": pa.array([NOW, NOW], type=pa.timestamp("us", tz="UTC")),
        }
    )
    posts = pa.table(
        {
            "platform": pa.array(["x"], type=pa.string()),
            "platform_post_id": pa.array(["p1"], type=pa.string()),
            "author_id": pa.array(["a1"], type=pa.string()),
            "text": pa.array(["spam spam spam"], type=pa.string()),
            "collected_at": pa.array([NOW], type=pa.timestamp("us", tz="UTC")),
        }
    )
    con = duckdb.connect()
    con.register("authors_tbl", authors)
    con.register("posts_tbl", posts)
    handles = top_suspicious_handles(con, "authors_tbl", "posts_tbl", n=1)
    assert handles == ["suspect"]


def _authors(ids: list[str], handles: list[str], collected: list[datetime]) -> pa.Table:
    n = len(ids)
    return pa.table(
        {
            "platform": pa.array(["x"] * n, type=pa.string()),
            "platform_user_id": pa.array(ids, type=pa.string()),
            "handle": pa.array(handles, type=pa.string()),
            "bio": pa.array([""] * n, type=pa.string()),
            "followers_count": pa.array([10] * n, type=pa.int64()),
            "following_count": pa.array([2000] * n, type=pa.int64()),
            "tweet_count": pa.array([500] * n, type=pa.int64()),
            "created_at": pa.array([NOW] * n, type=pa.timestamp("us", tz="UTC")),
            "profile_image_url": pa.array(["https://x.com/pic.jpg"] * n, type=pa.string()),
            "collected_at": pa.array(collected, type=pa.timestamp("us", tz="UTC")),
        }
    )


def _posts(
    ids: list[str], authors: list[str], texts: list[str],
    dts: list[date], collected: list[datetime],
) -> pa.Table:
    return pa.table(
        {
            "platform": pa.array(["x"] * len(ids), type=pa.string()),
            "platform_post_id": pa.array(ids, type=pa.string()),
            "author_id": pa.array(authors, type=pa.string()),
            "text": pa.array(texts, type=pa.string()),
            "dt": pa.array(dts, type=pa.date32()),
            "collected_at": pa.array(collected, type=pa.timestamp("us", tz="UTC")),
        }
    )


def _rank(con: duckdb.DuckDBPyConnection, authors: pa.Table, posts: pa.Table, **kw) -> list[str]:
    con.register("authors_tbl", authors)
    con.register("posts_tbl", posts)
    return top_suspicious_handles(con, "authors_tbl", "posts_tbl", n=10, **kw)


def test_author_posting_only_outside_window_is_not_ranked():
    today, old = date.today(), date.today() - timedelta(days=60)
    authors = _authors(["a1", "a2"], ["recent", "dormant"], [NOW, NOW])
    posts = _posts(
        ["p1", "p2"], ["a1", "a2"], ["hello", "hello"],
        [today, old], [NOW, NOW],
    )
    handles = _rank(duckdb.connect(), authors, posts, lookback_days=30)
    assert handles == ["recent"]


def test_window_is_configurable():
    old = date.today() - timedelta(days=60)
    authors = _authors(["a1"], ["dormant"], [NOW])
    posts = _posts(["p1"], ["a1"], ["hello"], [old], [NOW])
    assert _rank(duckdb.connect(), authors, posts, lookback_days=90) == ["dormant"]


def test_dedup_keeps_the_latest_collection_of_a_post():
    """Two collections of one post, different text. The duplicate ratio must be
    computed from the newest only - the old window dedup took the same row."""
    today = date.today()
    older, newer = NOW - timedelta(hours=2), NOW
    authors = _authors(["a1"], ["acct"], [NOW])
    posts = _posts(
        ["p1", "p1", "p2"], ["a1", "a1", "a1"],
        ["edited away", "same", "same"],
        [today, today, today], [older, newer, newer],
    )
    con = duckdb.connect()
    con.register("authors_tbl", authors)
    con.register("posts_tbl", posts)
    from kenya_monitor.suspicion import materialise

    table = materialise(con, "authors_tbl", "posts_tbl")
    row = con.sql(f"SELECT duplicate_text_ratio FROM {table}").fetchone()
    # p1 resolves to "same", so both surviving posts share one text: 1 - 1/2.
    assert row[0] == 0.5


def test_duplicate_text_ratio_matches_the_distinct_form():
    today = date.today()
    texts = ["a", "a", "b", "  A  ", "c"]
    authors = _authors(["a1"], ["acct"], [NOW])
    posts = _posts(
        [f"p{i}" for i in range(len(texts))], ["a1"] * len(texts), texts,
        [today] * len(texts), [NOW] * len(texts),
    )
    con = duckdb.connect()
    con.register("authors_tbl", authors)
    con.register("posts_tbl", posts)
    from kenya_monitor.suspicion import materialise

    table = materialise(con, "authors_tbl", "posts_tbl")
    got = con.sql(f"SELECT duplicate_text_ratio FROM {table}").fetchone()[0]
    want = con.sql(
        """
        SELECT 1.0 - count(DISTINCT lower(trim(text))) * 1.0 / count(*)
        FROM posts_tbl WHERE text IS NOT NULL
        """
    ).fetchone()[0]
    assert got == want


def test_ranks_without_a_dt_column():
    """Callers register bare tables with no hive partitioning; the window must
    be skipped rather than failing to bind."""
    authors = _authors(["a1"], ["acct"], [NOW])
    posts = pa.table(
        {
            "platform": pa.array(["x"], type=pa.string()),
            "platform_post_id": pa.array(["p1"], type=pa.string()),
            "author_id": pa.array(["a1"], type=pa.string()),
            "text": pa.array(["hello"], type=pa.string()),
            "collected_at": pa.array([NOW], type=pa.timestamp("us", tz="UTC")),
        }
    )
    assert _rank(duckdb.connect(), authors, posts) == ["acct"]


def test_materialise_is_reused_within_the_cache_window():
    """Both seed paths build this table in one cycle; the second must not pay
    for it again."""
    from kenya_monitor.suspicion import materialise

    today = date.today()
    authors = _authors(["a1"], ["acct"], [NOW])
    posts = _posts(["p1"], ["a1"], ["hello"], [today], [NOW])
    con = duckdb.connect()
    con.register("authors_tbl", authors)
    con.register("posts_tbl", posts)

    materialise(con, "authors_tbl", "posts_tbl")
    con.execute("DELETE FROM _susp_scored")
    materialise(con, "authors_tbl", "posts_tbl")
    assert con.sql("SELECT count(*) FROM _susp_scored").fetchone()[0] == 0

    materialise(con, "authors_tbl", "posts_tbl", max_age_minutes=0)
    assert con.sql("SELECT count(*) FROM _susp_scored").fetchone()[0] == 1

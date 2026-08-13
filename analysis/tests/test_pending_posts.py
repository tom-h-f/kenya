"""The shared incremental-scan contract behind every enrichment pass.

Each pass used to pull the whole deduped corpus into pandas, pull the whole
scored-id set into pandas, anti-join in Python, then `.head(limit)` - applying
the cap in arbitrary Parquet order, so freshly collected posts were not
prioritised and the cost grew with the corpus forever.
"""

from __future__ import annotations

import duckdb
import pytest

from kma import db


@pytest.fixture
def con(monkeypatch):
    c = duckdb.connect()
    c.execute(
        "CREATE TABLE _posts (platform VARCHAR, platform_post_id VARCHAR, "
        "text VARCHAR, collected_at TIMESTAMP)"
    )
    c.executemany(
        "INSERT INTO _posts VALUES ('x', ?, ?, ?::TIMESTAMP)",
        [
            ["old", "an old post about nyoka", "2026-07-01 00:00:00"],
            ["mid", "a middling post", "2026-07-10 00:00:00"],
            ["new", "the newest post", "2026-07-20 00:00:00"],
            ["blank", "   ", "2026-07-21 00:00:00"],
            ["nulltext", None, "2026-07-21 00:00:00"],
        ],
    )
    c.execute("CREATE TABLE _scored (platform_post_id VARCHAR)")
    monkeypatch.setattr(db, "posts_source", lambda platform="*", type="*": "_posts")
    return c


def _ids(df) -> list[str]:
    return df["platform_post_id"].tolist()


def test_newest_first(con):
    """A bounded pass must take the freshest work, not whatever the scan hit."""
    assert _ids(db.pending_posts(con, "_scored", limit=2)) == ["new", "mid"]


def test_empty_and_null_text_are_excluded(con):
    got = _ids(db.pending_posts(con, "_scored"))
    assert "blank" not in got and "nulltext" not in got


def test_scored_posts_are_anti_joined_out(con):
    con.execute("INSERT INTO _scored VALUES ('new'), ('mid')")
    assert _ids(db.pending_posts(con, "_scored")) == ["old"]


def test_a_missing_scored_prefix_means_everything_pends(con):
    """A prefix no pass has written yet must not error - it means no work done."""
    assert set(_ids(db.pending_posts(con, "_no_such_table"))) == {"old", "mid", "new"}


def test_none_scored_source_skips_the_anti_join(con):
    con.execute("INSERT INTO _scored VALUES ('new')")
    assert "new" in _ids(db.pending_posts(con, None))


def test_priority_is_applied_before_the_limit(con):
    """Sorting after a cap only reorders rows already chosen. The oldest post is
    the only lexicon hit here, so a limit of 1 must still return it."""
    priority = "regexp_matches(text, '\\bnyoka\\b', 'i')"
    assert _ids(db.pending_posts(con, "_scored", limit=1, priority_sql=priority)) == ["old"]
    # Without the priority the same query takes the newest instead.
    assert _ids(db.pending_posts(con, "_scored", limit=1)) == ["new"]


def test_priority_still_falls_back_to_recency(con):
    priority = "regexp_matches(text, '\\bnyoka\\b', 'i')"
    assert _ids(db.pending_posts(con, "_scored", priority_sql=priority)) == [
        "old", "new", "mid",
    ]


def test_lexicon_priority_sql_matches_the_python_scanner():
    """Ordering in SQL and hit extraction in pandas must agree, or a bounded run
    prioritises posts the scanner then reports as clean."""
    from kma.incitement import _lexicon_hit_sql, scan_text

    c = duckdb.connect()
    hits_sql = _lexicon_hit_sql()
    for text in ("watu wa madoa doa", "nyoka wamerudi", "an ordinary post", ""):
        got = c.execute(
            f"SELECT ({hits_sql}) FROM (SELECT ? AS text)", [text]
        ).fetchone()[0]
        assert bool(got) == bool(scan_text(text)[0]), text


def test_lexicon_patterns_are_sql_safe():
    """`_lexicon_hit_sql` interpolates patterns directly, so a quote in one would
    break the query. They are authored in-repo, so pin it."""
    from kma.incitement import LEXICON

    for terms in LEXICON.values():
        for term, meta in terms.items():
            assert "'" not in meta["pattern"], term


# --- absent prefixes ----------------------------------------------------------


EMPTY_GLOB = "read_parquet('/tmp/kma-no-such-prefix/dt=*/run=*.parquet', union_by_name=true, hive_partitioning=true)"


def test_duckdb_raises_on_a_zero_file_glob():
    """The premise behind every prefix probe in this package. If DuckDB ever
    starts returning an empty relation instead, the guards become dead weight
    and this test says so."""
    c = duckdb.connect()
    with pytest.raises(duckdb.Error):
        c.sql(f"SELECT 1 FROM {EMPTY_GLOB} LIMIT 1").fetchall()


def test_prefix_readable_reports_an_absent_prefix():
    c = duckdb.connect()
    assert db.prefix_readable(c, EMPTY_GLOB) is False


def test_prefix_readable_reports_a_present_one(con):
    assert db.prefix_readable(con, "_scored") is True


def test_refresh_measure_survives_an_absent_incitement_prefix(monkeypatch):
    """`--refresh-measure` exists to migrate rows written before the NLI pass
    ran, so it must work on exactly the corpus where that prefix is empty. The
    read sat inside a CTE, where a zero-file glob fails at resolution rather
    than contributing no rows."""
    from kma import hatespeech

    monkeypatch.setattr(hatespeech, "incitement_source", lambda platform="*": EMPTY_GLOB)

    c = duckdb.connect()
    c.execute(
        "CREATE TABLE _hate (platform_post_id VARCHAR, label VARCHAR, "
        "p_neither DOUBLE, p_offensive DOUBLE, p_hate DOUBLE, hate_flag BOOLEAN, "
        "scored_at TIMESTAMP)"
    )
    c.execute("CREATE TABLE _posts (platform VARCHAR, platform_post_id VARCHAR, "
              "text VARCHAR, collected_at TIMESTAMP)")
    monkeypatch.setattr(hatespeech, "hatespeech_source", lambda platform="*": "_hate")
    monkeypatch.setattr(hatespeech, "posts_source", lambda platform="*": "_posts")

    # No rows to migrate, but it must return cleanly rather than raise.
    assert hatespeech.refresh_measure(c) == 0

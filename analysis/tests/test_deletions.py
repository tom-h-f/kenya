from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd
import pytest

from kma import deletions

NOW = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)


def _metrics(rows):
    return pd.DataFrame(rows, columns=["platform_post_id", "status", "absence_cause", "collected_at", "dt"])


def _posts(rows):
    return pd.DataFrame(rows, columns=["platform_post_id", "author_id", "created_at", "type"])


@pytest.fixture
def con(monkeypatch):
    con = duckdb.connect()
    monkeypatch.setattr(deletions, "metrics_source", lambda platform: "metrics_tbl")
    return con


def _outcomes(con, metrics, posts):
    con.register("metrics_tbl", metrics)
    con.register("posts_tbl", posts)
    return deletions.recheck_outcomes(con, posts_on=lambda platform, first, last: "posts_tbl")


def test_rows_from_before_the_status_column_are_not_counted_as_present(con):
    """Their absences were thrown away at collection; they carry no evidence."""
    metrics = _metrics([
        ("p1", None, None, NOW - timedelta(days=8), NOW.date()),
        ("p2", "present", None, NOW, NOW.date()),
    ])
    posts = _posts([("p1", "a1", NOW, "search"), ("p2", "a2", NOW, "search")])

    got = _outcomes(con, metrics, posts)

    assert list(got["platform_post_id"]) == ["p2"]


def test_absence_is_sticky_and_a_later_return_is_reported(con):
    metrics = _metrics([
        ("p1", "absent", "author_gone", NOW - timedelta(hours=10), NOW.date()),
        ("p1", "present", None, NOW, NOW.date()),
    ])
    posts = _posts([("p1", "a1", NOW - timedelta(days=2), "search")])

    row = _outcomes(con, metrics, posts).iloc[0]

    assert row["ever_absent"]
    assert row["first_cause"] == "author_gone"
    assert row["returned"]


def test_the_cause_is_taken_from_the_first_absence(con):
    metrics = _metrics([
        ("p1", "absent", "author_gone", NOW - timedelta(hours=10), NOW.date()),
        ("p1", "absent", "post_deleted", NOW, NOW.date()),
    ])
    posts = _posts([("p1", "a1", NOW, "search")])

    assert _outcomes(con, metrics, posts).iloc[0]["first_cause"] == "author_gone"


def test_a_post_in_two_partitions_is_counted_in_both(con):
    """Forcing one label would hide the overlap a reader needs to judge the
    comparison."""
    metrics = _metrics([("p1", "absent", "post_deleted", NOW, NOW.date())])
    posts = _posts([("p1", "a1", NOW, "search"), ("p1", "a1", NOW, "hate_timeline")])

    table = deletions.base_rate(_outcomes(con, metrics, posts)).set_index("population")

    assert table.loc["baseline", "absent"] == 1
    assert table.loc["targeted", "absent"] == 1
    assert table.loc["all re-checked", "posts"] == 1


def test_an_empty_arm_has_no_rate_rather_than_a_zero(con):
    metrics = _metrics([("p1", "present", None, NOW, NOW.date())])
    posts = _posts([("p1", "a1", NOW, "search")])

    table = deletions.base_rate(_outcomes(con, metrics, posts)).set_index("population")

    assert pd.isna(table.loc["control", "absent_share"])
    assert table.loc["baseline", "absent_share"] == 0.0


def test_author_rates_withhold_authors_below_the_floor():
    outcomes = pd.DataFrame({
        "author_id": ["a1", "a1", "a1", "a2"],
        "ever_absent": [True, False, False, True],
    })

    got = deletions.author_rates(outcomes, min_checked=3)

    assert list(got["author_id"]) == ["a1"]
    assert got.iloc[0]["absent_share"] == pytest.approx(1 / 3)

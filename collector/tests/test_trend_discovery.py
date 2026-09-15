"""Finding emerging hashtags in an exogenous sample.

The selection rule is the whole design, so that is what is tested: emergence
selects, author concentration does not, and rates normalise by sampled window
rather than by day so our own cadence cannot move them.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from kenya_monitor.trend_discovery import candidates, tags_in

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def _corpus(rows):
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE control (window_start TIMESTAMPTZ, text VARCHAR, "
        "hashtags VARCHAR[], author_id VARCHAR, created_at TIMESTAMPTZ)"
    )
    for r in rows:
        con.execute("INSERT INTO control VALUES (?,?,?,?,?)", r)
    return con


def _post(window, text, tags, author, when):
    return [window, text, tags, author, when]


def test_a_tag_that_was_never_there_before_emerges():
    recent = NOW - timedelta(days=1)
    old = NOW - timedelta(days=10)
    rows = [
        _post(recent, "vote #NewCampaign", ["NewCampaign"], "a", recent),
        _post(recent + timedelta(hours=1), "#NewCampaign again", ["NewCampaign"], "b",
              recent + timedelta(hours=1)),
        _post(old, "ordinary talk", [], "c", old),
    ]
    got = candidates(_corpus(rows), "SELECT * FROM control", now=NOW)

    assert [c.tag for c in got] == ["newcampaign"]
    assert got[0].prior_windows == 0


def test_a_tag_that_was_always_there_does_not():
    """Volume is not emergence. A steady tag is a topic, and the keyword list
    already has the popular topics."""
    rows = []
    for d in range(1, 12):
        w = NOW - timedelta(days=d)
        rows.append(_post(w, "#Steady", ["Steady"], f"a{d}", w))

    got = candidates(_corpus(rows), "SELECT * FROM control", now=NOW)

    assert [c.tag for c in got] == []


def test_a_tag_seen_in_one_window_only_is_not_a_candidate():
    """One window is one moment. A tag seen once is a post, not a campaign."""
    w = NOW - timedelta(days=1)
    rows = [
        _post(w, "#OneOff", ["OneOff"], "a", w),
        _post(w, "#OneOff", ["OneOff"], "b", w),
        _post(w, "#OneOff", ["OneOff"], "c", w),
    ]
    got = candidates(_corpus(rows), "SELECT * FROM control", now=NOW)

    assert [c.tag for c in got] == []


def test_author_concentration_is_recorded_but_does_not_select():
    """Selecting on concentration would assume the conclusion. It has to stay
    evidence about a tag that emergence found."""
    r1, r2 = NOW - timedelta(days=1), NOW - timedelta(days=2)
    rows = [
        _post(r1, "#Pushed", ["Pushed"], "a", r1),
        _post(r1, "#Pushed", ["Pushed"], "a", r1),
        _post(r2, "#Pushed", ["Pushed"], "a", r2),
        _post(r2, "#Pushed", ["Pushed"], "b", r2),
    ]
    got = candidates(_corpus(rows), "SELECT * FROM control", now=NOW)

    assert got[0].tag == "pushed"
    assert got[0].posts == 4 and got[0].authors == 2
    assert got[0].posts_per_author == 2.0


def test_rates_normalise_by_sampled_window_not_by_day():
    """The arm's cadence varies and the horizon rolls, so a per-day rate would
    move with our own sampling rather than with the discourse."""
    w1, w2 = NOW - timedelta(days=1), NOW - timedelta(days=1, hours=1)
    rows = [
        _post(w1, "#T", ["T"], "a", w1),
        _post(w2, "#T", ["T"], "b", w2),
        _post(w1, "other", [], "c", w1),
        _post(w2, "other", [], "d", w2),
    ]
    got = candidates(_corpus(rows), "SELECT * FROM control", now=NOW)

    # Two recent windows, tag in both: rate is 1.0 per window, not per day.
    assert got[0].recent_rate == 1.0
    assert got[0].recent_windows == 2


def test_tags_come_from_the_field_or_the_text():
    """The hashtags field is empty on some collection paths, and a tag missed
    is a campaign missed."""
    assert tags_in("vote #Alpha now", None) == {"alpha"}
    assert tags_in(None, ["Beta"]) == {"beta"}
    assert tags_in("#Alpha", ["Beta"]) == {"alpha", "beta"}
    assert tags_in("#MiXeD", ["MIXED"]) == {"mixed"}

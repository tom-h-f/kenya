"""Unit tests for kma.relevance (no R2).

The learned gate decides `kenya_share`, which every v2 figure rests on, so what
matters here is the seam: the model's call where a post was scored, the keyword
gate's where it was not, and nothing failing when no scores exist at all.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from kma import relevance


@pytest.fixture
def con(monkeypatch):
    c = duckdb.connect()
    c.execute("""
        CREATE TABLE relevance (
            platform_post_id VARCHAR, model VARCHAR, p_kenya DOUBLE, scored_at TIMESTAMPTZ)
    """)
    monkeypatch.setattr(relevance, "relevance_source", lambda platform="x", model="*": "relevance")
    return c


def _score(con, post_id, p, at="2026-09-12"):
    con.execute("INSERT INTO relevance VALUES (?, 'm', ?, ?)", [post_id, p, at])


POSTS = pd.DataFrame({
    "post_id": ["p1", "p2", "p3"],
    "text": [
        "Completely unrelated cooking thread about pasta",   # gate: ambiguous
        "Nairobi traffic is a nightmare today",              # gate: kenya
        "The mood in the room shifted after the ruling",     # gate: ambiguous
    ],
})


def test_the_model_decides_where_it_scored_the_post(con):
    _score(con, "p1", 0.97)
    _score(con, "p2", 0.02)

    got = relevance.buckets(con, POSTS).tolist()

    assert got[0] == "kenya", "scored above the threshold, whatever the lexicon thinks"
    assert got[1] == "offdomain", "the model overrides a lexicon hit too"


def test_an_unscored_post_falls_back_to_the_keyword_gate(con):
    _score(con, "p1", 0.97)

    got = relevance.buckets(con, POSTS)

    assert got.iloc[2] == "ambiguous", "p3 has no score, so the gate's answer stands"
    assert got.iloc[1] == "kenya", "and the gate still catches its own vocabulary"


def test_the_latest_score_wins(con):
    _score(con, "p1", 0.02, at="2026-09-01")
    _score(con, "p1", 0.99, at="2026-09-12")

    assert relevance.scores(con, ["p1"]).loc["p1"] == pytest.approx(0.99)


def test_the_threshold_is_the_callers_choice(con):
    _score(con, "p3", 0.6)

    assert relevance.buckets(con, POSTS, threshold=0.5).iloc[2] == "kenya"
    assert relevance.buckets(con, POSTS, threshold=0.9).iloc[2] == "offdomain"


def test_no_persisted_scores_is_the_gate_alone(monkeypatch):
    """Posts collected since the last scoring pass are the normal case, and a
    project with no scores at all must still produce a Kenya share."""
    c = duckdb.connect()
    monkeypatch.setattr(relevance, "relevance_source", lambda platform="x", model="*": "nothing_here")

    got = relevance.buckets(c, POSTS).tolist()

    assert got == ["ambiguous", "kenya", "ambiguous"]

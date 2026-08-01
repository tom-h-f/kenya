"""Unit tests for coded-term mining and the governance gate on it."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from kenya_monitor import hate_seek as hs
from kenya_monitor import hate_signal as hsig
from kenya_monitor import mined_terms as mt
from kenya_monitor.config import load_hate_terms

NOW = datetime.now(timezone.utc)


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute(
        "CREATE TABLE _posts (platform VARCHAR, platform_post_id VARCHAR, author_id VARCHAR, "
        "text VARCHAR, created_at TIMESTAMPTZ, collected_at TIMESTAMPTZ)"
    )
    c.execute(
        "CREATE TABLE _authors (platform VARCHAR, platform_user_id VARCHAR, handle VARCHAR, "
        "collected_at TIMESTAMPTZ)"
    )
    return c


def add(con, uid, handle):
    con.execute("INSERT INTO _authors VALUES ('x', ?, ?, ?)", [uid, handle, NOW])


def post(con, pid, uid, text, days_ago=0):
    con.execute(
        "INSERT INTO _posts VALUES ('x', ?, ?, ?, ?, ?)",
        [pid, uid, text, NOW - timedelta(days=days_ago), NOW],
    )


def mine(con, handles, **kw):
    return hsig.mine_terms(con, "_posts", "_authors", handles, **kw)


def test_no_cohort_yields_nothing(con):
    assert mine(con, []) == []


def test_term_used_by_one_account_is_not_a_register(con):
    """A single account repeating a word is a verbal tic, not a shared register."""
    add(con, "u1", "seed1")
    add(con, "u2", "seed2")
    add(con, "u3", "seed3")
    for i in range(20):
        post(con, f"p{i}", "u1", "the zzcoded menace is here")
    for i in range(20):
        post(con, f"q{i}", "u2", "ordinary political comment about policy")
    terms = {t["term"] for t in mine(con, ["seed1", "seed2", "seed3"])}
    assert "zzcoded" not in terms


def test_term_shared_across_the_cohort_is_surfaced(con):
    for i in (1, 2, 3):
        add(con, f"u{i}", f"seed{i}")
    add(con, "out", "outsider")
    for uid in ("u1", "u2", "u3"):
        for i in range(10):
            post(con, f"{uid}p{i}", uid, "zzcoded menace rising")
    for i in range(60):
        post(con, f"o{i}", "out", "ordinary political commentary about the budget")
    terms = {t["term"]: t for t in mine(con, ["seed1", "seed2", "seed3"])}
    assert "zzcoded" in terms
    assert terms["zzcoded"]["a_in"] >= 3
    assert terms["zzcoded"]["lift"] > 1.0


def test_novelty_ranks_a_sudden_term_above_a_steady_one(con):
    """New coded terms appear suddenly; that is far more specific than a term
    merely being cohort-flavoured."""
    for i in (1, 2, 3):
        add(con, f"u{i}", f"seed{i}")
    for uid in ("u1", "u2", "u3"):
        for i in range(10):  # steady across the whole window
            post(con, f"{uid}s{i}", uid, "steadyword commentary", days_ago=1 + i)
        for i in range(10):  # only in the last 3 days
            post(con, f"{uid}n{i}", uid, "suddenword commentary", days_ago=0)
    # min_lift=0: every post here is a cohort post, so lift is 1.0 by
    # construction. This test is about the novelty ordering, not the gate.
    ranked = [t["term"] for t in mine(con, ["seed1", "seed2", "seed3"], min_lift=0.0)]
    assert "suddenword" in ranked and "steadyword" in ranked
    assert ranked.index("suddenword") < ranked.index("steadyword")


def test_stopwords_and_common_corpus_terms_are_suppressed(con):
    for i in (1, 2, 3):
        add(con, f"u{i}", f"seed{i}")
    for uid in ("u1", "u2", "u3"):
        for i in range(10):
            post(con, f"{uid}p{i}", uid, "the watu wa kule na sisi wote")
    terms = {t["term"] for t in mine(con, ["seed1", "seed2", "seed3"])}
    assert not terms & {"the", "watu", "wa", "sisi", "wote", "na"}


def test_under_represented_terms_are_gated_out(con):
    """Regression for the first live run (2026-07-28): ranking by novelty alone
    filled the candidate list with terms the cohort barely used, whose handful
    of occurrences merely happened to be recent. Lift gates, novelty ranks."""
    for i in (1, 2, 3):
        add(con, f"u{i}", f"seed{i}")
    add(con, "out", "outsider")
    # `rareword` is recent inside the cohort but far more common outside it.
    # The cohort also needs a broad ordinary vocabulary, otherwise the term is
    # a large share of its tokens purely because it says nothing else.
    for uid in ("u1", "u2", "u3"):
        for i in range(4):
            post(con, f"{uid}r{i}", uid, "rareword commentary", days_ago=0)
        for i in range(40):
            post(con, f"{uid}f{i}", uid, f"filler{i} padding{i} chatter{i}", days_ago=2)
    for i in range(400):
        post(con, f"o{i}", "out", "rareword commentary regarding budgets", days_ago=1)
    terms = {t["term"] for t in mine(con, ["seed1", "seed2", "seed3"])}
    assert "rareword" not in terms

    ungated = {t["term"] for t in mine(con, ["seed1", "seed2", "seed3"], min_lift=0.0)}
    assert "rareword" in ungated  # only the lift gate was keeping it out


def test_generic_filler_is_stoplisted(con):
    for i in (1, 2, 3):
        add(con, f"u{i}", f"seed{i}")
    for uid in ("u1", "u2", "u3"):
        for i in range(10):
            post(con, f"{uid}p{i}", uid, "haha since trying without down end made")
    terms = {t["term"] for t in mine(con, ["seed1", "seed2", "seed3"], min_lift=0.0)}
    assert not terms & {"haha", "since", "trying", "without", "down", "end", "made"}


def test_handles_hashtags_and_urls_are_not_mined(con):
    """A username or campaign tag is an entity, not a register marker. The
    first gated live run returned three usernames among eight candidates
    because their word bodies tokenised like ordinary text."""
    for i in (1, 2, 3):
        add(con, f"u{i}", f"seed{i}")
    for uid in ("u1", "u2", "u3"):
        for i in range(10):
            post(
                con, f"{uid}p{i}", uid,
                "@murimamaestro zzcoded #rutomustgo https://t.co/abcdefg",
            )
    terms = {t["term"] for t in mine(con, ["seed1", "seed2", "seed3"], min_lift=0.0)}
    assert "zzcoded" in terms
    assert not terms & {"murimamaestro", "rutomustgo", "abcdefg", "https"}


def test_examples_are_returned_for_review(con):
    add(con, "u1", "seed1")
    post(con, "p1", "u1", "a post containing zzcoded here")
    assert hsig.term_examples(con, "_posts", "zzcoded") == ["a post containing zzcoded here"]
    assert hsig.term_examples(con, "_posts", "absentword") == []


# --- governance gate --------------------------------------------------------


def test_candidates_are_not_searched_until_approved():
    """The gate: mining writes candidates, it does not promote them."""
    terms = mt.merge([], [{"term": "zzcoded", "novelty": 9.0, "c_in": 30, "a_in": 4}])
    assert terms[0].approved is False
    assert mt.active(terms, autopromote=False) == []
    assert mt.active(terms, autopromote=True) == [{"term": "zzcoded"}]


def test_human_approval_survives_re_mining():
    terms = mt.merge([], [{"term": "zzcoded", "novelty": 9.0}])
    terms[0].approved = True
    again = mt.merge(terms, [{"term": "zzcoded", "novelty": 4.0, "c_in": 99}])
    assert again[0].approved is True
    assert again[0].n_cohort == 99


def test_approved_terms_sort_first():
    terms = mt.merge(
        [], [{"term": "aaa", "novelty": 1.0}, {"term": "bbb", "novelty": 9.0}]
    )
    for t in terms:
        if t.term == "aaa":
            t.approved = True
    assert mt.merge(terms, [])[0].term == "aaa"


def test_stale_candidates_expire():
    old = mt.MinedTerm(term="gone", last_confirmed=(NOW - timedelta(days=30)).isoformat())
    fresh = mt.MinedTerm(term="here", last_confirmed=NOW.isoformat())
    kept = {t.term for t in mt.merge([old, fresh], [])}
    assert kept == {"here"}


def test_active_respects_the_cap():
    terms = mt.merge([], [{"term": f"t{i}", "novelty": float(i)} for i in range(10)])
    for t in terms:
        t.approved = True
    assert len(mt.active(terms, cap=3)) == 3


def test_state_round_trips(tmp_path):
    path = tmp_path / "mined.json"
    assert mt.load(path) == []
    terms = mt.merge([], [{"term": "zzcoded", "novelty": 2.0, "c_in": 12, "a_in": 3}])
    terms[0].examples = ["an example post"]
    mt.save(terms, path)
    back = mt.load(path)
    assert back[0].term == "zzcoded"
    assert back[0].examples == ["an example post"]
    assert back[0].source == mt.SOURCE


def test_mined_terms_always_get_tightest_anchoring():
    """A machine-derived term has no provenance, so it gets maximum scoping -
    which is the SMALLER anchor group, since anchoring is an AND over an OR."""
    register = load_hate_terms()
    terms = mt.merge([], [{"term": "zzcoded"}])
    terms[0].approved = True
    picked = hs.select_terms(
        register, mined=mt.active(terms), max_terms=8, max_targets=0, max_mined=5
    )
    mined_q = [q for q in picked if q.source == hs.MINED]
    assert len(mined_q) == 1
    assert mined_q[0].fp_risk == "high"
    assert hs.anchors_for(mined_q[0], register.anchors) == register.anchors["core"]
    rendered = hs.render(mined_q[0], register.anchors)
    assert "zzcoded (Kenya OR " in rendered

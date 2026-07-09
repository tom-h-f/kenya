"""Unit tests for kma.stories (no R2 required).

The R2-reading entrypoints (candidate_stories, origin, spread) are exercised in
the notebook / verify step against live data; here we cover the pure logic:
centroids, trusted-media corroboration directionality, and the scorecard
composite - monkeypatching the two DB-backed helpers the scorecard calls."""

import numpy as np
import pandas as pd

from kma import stories as st


def _unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v)


def _stories_frame(spec):
    """spec = {story_id: [(author, vec, text, hashtags), ...]} -> members frame."""
    rows = []
    pid = 0
    for sid, members in spec.items():
        for author, vec, text, tags in members:
            rows.append(
                {
                    "platform_post_id": f"p{pid}",
                    "story_id": sid,
                    "author_id": author,
                    "author_handle": author,
                    "text": text,
                    "created_at": pd.Timestamp("2026-07-08", tz="UTC"),
                    "is_repost": False,
                    "hashtags": tags,
                    "conversation_id": None,
                    "embedding": list(_unit(vec)),
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def test_story_centroids_are_renormalised_means():
    s = _stories_frame(
        {0: [("a", [1, 0, 0], "x", []), ("b", [1, 0, 0], "y", [])]}
    )
    c = st.story_centroids(s)
    assert set(c) == {0}
    np.testing.assert_allclose(c[0], [1, 0, 0], atol=1e-6)
    assert abs(np.linalg.norm(c[0]) - 1.0) < 1e-6


def test_corroboration_high_when_trusted_matches(monkeypatch):
    s = _stories_frame({0: [("a", [1, 0, 0], "iebc rigged", [])]})
    trusted = pd.DataFrame(
        {
            "platform_post_id": ["t1", "t2"],
            "author_handle": ["NationAfrica", "StandardKenya"],
            "text": ["IEBC denies rigging claim", "unrelated weather report"],
            "created_at": [pd.Timestamp("2026-07-08", tz="UTC")] * 2,
            "embedding": [list(_unit([0.99, 0.14, 0])), list(_unit([0, 0, 1]))],
        }
    )
    monkeypatch.setattr(st, "_trusted_posts", lambda *a, **k: trusted)
    out = st.corroboration(None, s)
    assert out.loc[0, "corrob_sim"] > 0.95
    assert out.loc[0, "nearest_handle"] == "NationAfrica"


def test_corroboration_low_when_no_trusted_match(monkeypatch):
    s = _stories_frame({0: [("a", [1, 0, 0], "fabricated rumour", [])]})
    trusted = pd.DataFrame(
        {
            "platform_post_id": ["t1"],
            "author_handle": ["ntvkenya"],
            "text": ["something entirely different"],
            "created_at": [pd.Timestamp("2026-07-08", tz="UTC")],
            "embedding": [list(_unit([0, 1, 0]))],
        }
    )
    monkeypatch.setattr(st, "_trusted_posts", lambda *a, **k: trusted)
    out = st.corroboration(None, s)
    assert out.loc[0, "corrob_sim"] < 0.2


def test_corroboration_empty_trusted_is_maximal_gap(monkeypatch):
    s = _stories_frame({0: [("a", [1, 0, 0], "x", [])]})
    monkeypatch.setattr(
        st, "_trusted_posts", lambda *a, **k: pd.DataFrame(columns=["embedding"])
    )
    out = st.corroboration(None, s)
    assert out.loc[0, "corrob_sim"] == 0.0
    assert pd.isna(out.loc[0, "nearest_handle"])


def test_story_hashtags_counted_and_prefixed():
    s = _stories_frame(
        {0: [("a", [1, 0, 0], "x", ["Rigged", "rigged"]), ("b", [1, 0, 0], "y", ["#Rigged"])]}
    )
    tags = st._story_hashtags(s)
    assert tags[0][0] == "#rigged"


def test_story_keywords_grouped_by_story():
    s = _stories_frame(
        {
            0: [("a", [1, 0, 0], "iebc rigged election", []),
                ("b", [1, 0, 0], "iebc rigged again", [])],
            1: [("c", [0, 1, 0], "weather sunny today", [])],
        }
    )
    kw = st._story_keywords(s)
    assert set(kw) == {0, 1}
    assert all(isinstance(terms, list) for terms in kw.values())


def test_scorecard_ranks_uncorroborated_botty_story_first(monkeypatch):
    # story 0: bot-amplified, coordinated, uncorroborated. story 1: clean, corroborated.
    s = _stories_frame(
        {
            0: [("bot1", [1, 0, 0], "claim a", []), ("bot2", [1, 0, 0], "claim a too", []),
                ("bot3", [1, 0, 0], "claim a again", [])],
            1: [("real1", [0, 1, 0], "claim b", []), ("real2", [0, 1, 0], "claim b too", [])],
        }
    )
    corrob = pd.DataFrame(
        {
            "story_id": [0, 1],
            "corrob_sim": [0.05, 0.95],
            "nearest_handle": ["NationAfrica", "NationAfrica"],
            "nearest_text": ["x", "y"],
            "nearest_post_id": ["t1", "t2"],
        }
    )
    fake_auth = pd.DataFrame(
        {
            "platform_user_id": ["bot1", "bot2", "bot3", "real1", "real2"],
            "suspicion": [0.9, 0.85, 0.88, 0.1, 0.15],
        }
    )
    monkeypatch.setattr(
        "kma.authenticity.authenticity_score", lambda *a, **k: fake_auth
    )
    monkeypatch.setattr(st, "_coordination_author_ids", lambda *a, **k: {"bot1", "bot2"})

    cards = st.story_scorecard(None, s, corrob)
    assert list(cards["story_id"]) == [0, 1]  # botty/uncorroborated ranks first
    assert cards.loc[0, "story_suspicion_index"] > cards.loc[1, "story_suspicion_index"]
    assert cards.loc[0, "corroboration_gap"] > cards.loc[1, "corroboration_gap"]
    # component columns are exposed for transparency
    assert {f"ix_{k}" for k in st.STORY_WEIGHTS} <= set(cards.columns)


def test_story_weights_sum_to_one():
    assert abs(sum(st.STORY_WEIGHTS.values()) - 1.0) < 1e-9


def test_empty_stories_return_empty_frames():
    empty = pd.DataFrame(columns=["story_id", "embedding"])
    assert st.corroboration(None, empty).empty
    assert st.story_scorecard(None, empty).empty

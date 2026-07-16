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
    # trusted post shares the claim vocabulary (iebc/rigged/election), so it corroborates
    s = _stories_frame({0: [("a", [1, 0, 0], "iebc rigged election results", [])]})
    trusted = pd.DataFrame(
        {
            "platform_post_id": ["t1", "t2"],
            "author_handle": ["NationAfrica", "StandardKenya"],
            "text": ["IEBC denies rigged election claim", "unrelated weather report"],
            "created_at": [pd.Timestamp("2026-07-08", tz="UTC")] * 2,
            "embedding": [list(_unit([0.99, 0.14, 0])), list(_unit([0, 0, 1]))],
        }
    )
    monkeypatch.setattr(st, "_trusted_posts", lambda *a, **k: trusted)
    out = st.corroboration(None, s)
    assert out.loc[0, "corrob_sim"] > 0.95
    assert out.loc[0, "nearest_handle"] == "NationAfrica"


def test_corroboration_gated_out_when_only_topic_overlaps(monkeypatch):
    # same topic (accident), different claim: shares only "accident" -> not corroboration
    s = _stories_frame(
        {0: [("a", [1, 0, 0], "ruto motorcade crash accident kills bodyguard", [])]}
    )
    trusted = pd.DataFrame(
        {
            "platform_post_id": ["t1"],
            "author_handle": ["citizentvkenya"],
            "text": ["eight dead in road accident on nakuru eldoret highway matatu"],
            "created_at": [pd.Timestamp("2026-07-08", tz="UTC")],
            "embedding": [list(_unit([0.97, 0.24, 0]))],  # embeds close (same topic)
        }
    )
    monkeypatch.setattr(st, "_trusted_posts", lambda *a, **k: trusted)
    out = st.corroboration(None, s)
    assert out.loc[0, "corrob_sim"] == 0.0  # topic bleed does not count as coverage
    assert pd.isna(out.loc[0, "nearest_handle"])


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


def test_stable_story_id_deterministic_for_same_members():
    ids_a = ["p3", "p1", "p2"]
    ids_b = ["p1", "p2", "p3"]
    assert st.stable_story_id(ids_a) == st.stable_story_id(ids_b)
    assert st.stable_story_id(ids_a) != st.stable_story_id(["p1", "p2"])


def test_attach_stable_story_ids_on_members():
    s = _stories_frame(
        {
            0: [("a", [1, 0, 0], "claim one alpha", []),
                ("b", [1, 0, 0], "claim one beta", [])],
            1: [("c", [0, 1, 0], "other claim gamma", [])],
        }
    )
    out = st.attach_stable_story_ids(s)
    assert "stable_story_id" in out.columns
    g0 = out.loc[out["story_id"] == 0, "stable_story_id"].unique()
    g1 = out.loc[out["story_id"] == 1, "stable_story_id"].unique()
    assert len(g0) == 1 and len(g1) == 1
    assert g0[0] != g1[0]
    assert g0[0] == st.stable_story_id(
        out.loc[out["story_id"] == 0, "platform_post_id"].tolist()
    )


def test_assign_tiers_thin_high_gap_not_high_suspicion_without_amp(monkeypatch):
    # 2-author maximal-gap claim: thin_evidence, not high_suspicion
    s = _stories_frame(
        {
            0: [
                ("a", [1, 0, 0], "Ruto motorcade crash Embu highway kills guard", []),
                ("b", [1, 0, 0], "Ruto motorcade crash Embu highway injured", []),
            ],
        }
    )
    s = st.attach_stable_story_ids(s)
    corrob = pd.DataFrame(
        {
            "story_id": [0],
            "corrob_sim": [0.0],
            "nearest_handle": [None],
            "nearest_text": [None],
            "nearest_post_id": [None],
        }
    )
    fake_auth = pd.DataFrame(
        {"platform_user_id": ["a", "b"], "suspicion": [0.1, 0.12]}
    )
    monkeypatch.setattr(
        "kma.authenticity.authenticity_score", lambda *a, **k: fake_auth
    )
    monkeypatch.setattr(st, "_coordination_author_ids", lambda *a, **k: set())
    cards = st.story_scorecard(None, s, corrob)
    cards = st.assign_tiers(s, cards)
    assert cards.loc[0, "tier"] == st.TIER_THIN
    assert cards.loc[0, "high_suspicion"] is False or cards.loc[0, "high_suspicion"] == False
    assert cards.loc[0, "stable_story_id"] == s.loc[0, "stable_story_id"]


def test_assign_tiers_main_lane_size_three():
    s = _stories_frame(
        {
            0: [
                ("a", [1, 0, 0], "iebc rigged election results claim", []),
                ("b", [1, 0, 0], "iebc rigged election results again", []),
                ("c", [1, 0, 0], "iebc rigged election results third", []),
            ],
        }
    )
    s = st.attach_stable_story_ids(s)
    cards = pd.DataFrame(
        {
            "story_id": [0],
            "size": [3],
            "corroboration_gap": [0.8],
            "amplifier_botness": [0.2],
            "coordination_overlap": [0.0],
            "story_suspicion_index": [0.6],
        }
    )
    out = st.assign_tiers(s, cards)
    assert out.loc[0, "tier"] == st.TIER_MAIN
    assert out.loc[0, "stable_story_id"] == s.loc[s["story_id"] == 0, "stable_story_id"].iloc[0]



def test_persist_columns_include_tier_and_stable_id():
    cards = pd.DataFrame(
        {
            "story_id": [0],
            "stable_story_id": ["abc123"],
            "tier": [st.TIER_THIN],
            "size": [2],
            "n_posts": [2],
            "keywords": [["ruto"]],
            "hashtags": [[]],
            "representative_text": ["x"],
            "representative_post_id": ["p0"],
            "member_post_ids": [["p0", "p1"]],
            "corrob_sim": [0.0],
            "corroboration_gap": [1.0],
            "amplifier_botness": [0.1],
            "coordination_overlap": [0.0],
            "source_concentration": [1.0],
            "story_suspicion_index": [0.4],
            "high_suspicion": [False],
        }
    )
    cols = st.persist_story_columns(cards)
    assert "stable_story_id" in cols
    assert "tier" in cols
    assert "high_suspicion" in cols

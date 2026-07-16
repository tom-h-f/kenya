"""Offline regression tests for the story-detection fixes (no R2 required).

Covers the de-chaining filters (Fix A), the claim-level + entity-gated corroboration,
and the ground-truth registry - the defects the ground-truth eval exposed. The live
end-to-end recall of the two real cases is exercised by analysis/run_eval.py against R2."""

import numpy as np
import pandas as pd

from kma import eval as gt
from kma import stories as st


def _unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v)


# --- Fix A: low-information filter -----------------------------------------------

def test_content_words_ignores_links_and_mentions():
    assert st._content_words("https://t.co/abc") == 0
    assert st._content_words("@alice @bob") == 0
    assert st._content_words("🔥🔥🔥") == 0
    assert st._content_words("Ruto motorcade crashes on Embu highway today") >= 5


def test_drop_low_information_removes_junk_keeps_claims():
    df = pd.DataFrame(
        {
            "text": [
                "https://t.co/x",                              # bare link
                "lol",                                          # one-liner
                "President Ruto motorcade crashes near Embu",   # real claim
            ],
            "author_id": ["a", "b", "c"],
        }
    )
    out = st._drop_low_information(df)
    assert list(out["text"]) == ["President Ruto motorcade crashes near Embu"]


# --- Fix A: chaining-blob rejection ----------------------------------------------

def test_reject_chaining_blobs_drops_large_dispersed_component():
    rng = np.random.default_rng(0)
    # one big DISPERSED component (random unit vectors) = a chaining hairball
    blob = rng.normal(size=(300, 16)).astype("float32")
    blob /= np.linalg.norm(blob, axis=1, keepdims=True)
    # one small TIGHT component (near-identical vectors) = a real claim cluster
    base = _unit(rng.normal(size=16))
    tight = np.array([_unit(base + 0.02 * rng.normal(size=16)) for _ in range(5)],
                     dtype="float32")
    x = np.vstack([blob, tight])
    labels = np.array([0] * 300 + [1] * 5)
    authors = np.array([f"b{i}" for i in range(300)] + [f"t{i}" for i in range(5)])

    keep = st._reject_chaining_blobs(x, labels, authors, max_authors=150, min_cohesion=0.80)
    assert keep[300:].all()          # tight cluster survives
    assert not keep[:300].any()      # dispersed blob dropped


def test_reject_chaining_blobs_keeps_large_but_cohesive_component():
    # a genuinely viral story: many authors, but tightly clustered -> must be kept
    rng = np.random.default_rng(1)
    base = _unit(rng.normal(size=16))
    v = np.array([_unit(base + 0.01 * rng.normal(size=16)) for _ in range(300)],
                 dtype="float32")
    labels = np.zeros(300, dtype=int)
    authors = np.array([f"a{i}" for i in range(300)])
    keep = st._reject_chaining_blobs(v, labels, authors, max_authors=150, min_cohesion=0.80)
    assert keep.all()


# --- Claim-level + entity-gated corroboration ------------------------------------

def _members(spec):
    rows, pid = [], 0
    for sid, members in spec.items():
        for author, vec, text in members:
            rows.append({
                "platform_post_id": f"p{pid}", "story_id": sid, "author_id": author,
                "author_handle": author, "text": text,
                "created_at": pd.Timestamp("2026-07-08", tz="UTC"),
                "is_repost": False, "hashtags": [], "conversation_id": None,
                "embedding": list(_unit(vec)),
            })
            pid += 1
    return pd.DataFrame(rows)


def test_entity_gate_rejects_same_topic_different_entities(monkeypatch):
    # fabricated motorcade crash vs real, unrelated accident: embeds close, shares
    # generic accident words, but no entities -> not corroboration (maximal gap).
    s = _members({0: [
        ("a", [1, 0, 0], "President Ruto motorcade crashes head on near Embu, several injured"),
        ("b", [1, 0, 0], "Speeding Ruto security vehicle involved in accident heading to Embu"),
    ]})
    trusted = pd.DataFrame({
        "platform_post_id": ["t1"],
        "author_handle": ["citizentvkenya"],
        "text": ["Eight killed in road accident on Nakuru Eldoret highway, several injured"],
        "created_at": [pd.Timestamp("2026-07-08", tz="UTC")],
        "embedding": [list(_unit([0.97, 0.24, 0]))],
    })
    monkeypatch.setattr(st, "_trusted_posts", lambda *a, **k: trusted)
    out = st.corroboration(None, s)
    assert out.loc[0, "corrob_sim"] == 0.0
    assert pd.isna(out.loc[0, "nearest_handle"])


def test_entity_gate_allows_genuine_same_claim_match(monkeypatch):
    # same claim, shared entities (SACCO, Government) -> corroborated despite being disinfo
    s = _members({0: [
        ("a", [1, 0, 0], "Government plans to tap SACCO savings for the Infrastructure Fund"),
        ("b", [1, 0, 0], "Kenya Government to borrow SACCO trillions for Infrastructure Fund"),
    ]})
    trusted = pd.DataFrame({
        "platform_post_id": ["t1"],
        "author_handle": ["ntvkenya"],
        "text": ["Government moves to tap SACCO trillions for Infrastructure Fund development"],
        "created_at": [pd.Timestamp("2026-07-08", tz="UTC")],
        "embedding": [list(_unit([0.99, 0.1, 0]))],
    })
    monkeypatch.setattr(st, "_trusted_posts", lambda *a, **k: trusted)
    out = st.corroboration(None, s)
    assert out.loc[0, "corrob_sim"] > 0.95
    assert out.loc[0, "nearest_handle"] == "ntvkenya"


def test_entity_terms_extracts_proper_nouns():
    ents = st._entity_terms("President Ruto's motorcade near Embu https://t.co/x @foo")
    assert {"ruto", "embu"} <= ents
    assert "https" not in ents and "foo" not in ents


# --- Ground-truth registry -------------------------------------------------------

def test_ground_truth_registry_well_formed():
    names = [c.name for c in gt.GROUND_TRUTH]
    assert "sacco-savings-borrow" in names and "ruto-motorcade-crash" in names
    assert all(c.expect in (gt.EXPECT_SURFACE, gt.EXPECT_KNOWN_LIMITATION)
               for c in gt.GROUND_TRUTH)
    assert all(c.lane in ("main", "thin", "either") for c in gt.GROUND_TRUTH)
    assert any(c.expect == gt.EXPECT_SURFACE for c in gt.GROUND_TRUTH)
    moto = next(c for c in gt.GROUND_TRUTH if c.name == "ruto-motorcade-crash")
    assert moto.expect == gt.EXPECT_SURFACE
    assert moto.lane == "thin"
    sacco = next(c for c in gt.GROUND_TRUTH if c.name == "sacco-savings-borrow")
    assert sacco.lane == "main"


def test_triage_cut_is_fraction_with_floor():
    assert gt._triage_cut(1000) == 150            # 15%
    assert gt._triage_cut(10) == gt.TOP_MIN       # floored on tiny runs


def test_drop_stage_thin_lane_surfaces_without_main_rank():
    row = {
        "n_present": 2,
        "n_embedded": 2,
        "story_id": 7,
        "is_blob": False,
        "tier": st.TIER_THIN,
        "rank": None,
        "n_main_stories": 40,
    }
    assert gt._drop_stage(row, lane="thin") is None
    row["tier"] = st.TIER_MAIN
    assert gt._drop_stage(row, lane="thin") == "tiering (not thin_evidence)"


def test_drop_stage_main_requires_rank_within_cut():
    row = {
        "n_present": 5,
        "n_embedded": 5,
        "story_id": 1,
        "is_blob": False,
        "tier": st.TIER_MAIN,
        "rank": 3,
        "n_main_stories": 40,
    }
    assert gt._drop_stage(row, lane="main") is None
    row["rank"] = 100
    assert gt._drop_stage(row, lane="main") == "scoring"

"""Unit tests for kma.dossier (no R2).

The dossier is the one artifact in the repo that deliberately carries post text
and account handles, because an adjudicator cannot judge intent without them.
So these tests cover two things: that the evidence is actually assembled, and
that it stays on the private side of the line `dashboard.py` draws.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from kma import coordination as co
from kma import dossier


@pytest.fixture
def con(monkeypatch):
    c = duckdb.connect()
    c.execute("""
        CREATE TABLE posts (
            platform VARCHAR, platform_post_id VARCHAR, author_id VARCHAR,
            author_handle VARCHAR, text VARCHAR, created_at TIMESTAMPTZ,
            collected_at TIMESTAMPTZ, is_repost BOOLEAN,
            like_count INT, reply_count INT, repost_count INT, quote_count INT,
            dt DATE GENERATED ALWAYS AS (CAST(collected_at AS DATE)))
    """)
    c.execute("""
        CREATE TABLE engagements (
            platform VARCHAR, platform_post_id VARCHAR,
            platform_user_id VARCHAR, kind VARCHAR, collected_at TIMESTAMPTZ)
    """)
    c.execute("""
        CREATE TABLE authors (
            platform VARCHAR, platform_user_id VARCHAR, handle VARCHAR,
            bio VARCHAR, profile_image_url VARCHAR, created_at TIMESTAMPTZ,
            collected_at TIMESTAMPTZ, followers_count INT, following_count INT)
    """)
    c.execute("""
        CREATE TABLE hatespeech (
            platform_post_id VARCHAR, domain VARCHAR, scored_at TIMESTAMPTZ)
    """)
    c.execute("""
        CREATE TABLE embeddings (
            platform_post_id VARCHAR, embedding FLOAT[], embedded_at TIMESTAMPTZ)
    """)
    monkeypatch.setattr(dossier, "embeddings_source", lambda platform="x", model="*": "embeddings")

    def post(pid, author, handle, text, engagement=0, repost=False):
        c.execute(
            "INSERT INTO posts (platform, platform_post_id, author_id, author_handle,"
            " text, created_at, collected_at, is_repost, like_count, reply_count,"
            " repost_count, quote_count)"
            " VALUES ('x', ?, ?, ?, ?, now(), now(), ?, ?, 0, 0, 0)",
            [pid, author, handle, text, repost, engagement],
        )

    # Two members who jointly retweet OBJ1; a third object only one touches.
    post("m1", "a1", "member_one", "Members own post about the election", 50)
    post("m2", "a2", "member_two", "Another member post", 10)
    post("OBJ1", "outsider", "big_account", "The thing they both amplified")
    post("OBJ2", "outsider", "big_account", "Only one member touched this")
    for uid in ("a1", "a2"):
        c.execute("INSERT INTO engagements VALUES ('x', 'OBJ1', ?, 'retweet', now())", [uid])
    c.execute("INSERT INTO engagements VALUES ('x', 'OBJ2', 'a1', 'retweet', now())")

    for uid, handle, bio in (("a1", "member_one", ""), ("a2", "member_two", "hi")):
        c.execute(
            "INSERT INTO authors VALUES ('x', ?, ?, ?, 'http://img/default_profile.png',"
            " TIMESTAMPTZ '2020-01-01', now(), 100, 200)",
            [uid, handle, bio],
        )
    c.execute("INSERT INTO hatespeech VALUES ('m1', 'kenya', now())")
    c.execute("INSERT INTO hatespeech VALUES ('m2', 'offdomain', now())")

    monkeypatch.setattr(co, "posts_source", lambda platform="x", type="*": "posts")
    monkeypatch.setattr(dossier, "engagements_source", lambda platform="x": "engagements")
    monkeypatch.setattr(dossier, "hatespeech_source", lambda platform="x": "hatespeech")
    monkeypatch.setattr("kma.db.authors_source", lambda platform="x": "authors")
    return c


MEMBERS = pd.DataFrame([
    {"cluster_id": 0, "author_id": "a1"},
    {"cluster_id": 0, "author_id": "a2"},
])


def _post(con, pid, author, handle, text):
    con.execute(
        "INSERT INTO posts (platform, platform_post_id, author_id, author_handle,"
        " text, created_at, collected_at, is_repost, like_count, reply_count,"
        " repost_count, quote_count)"
        " VALUES ('x', ?, ?, ?, ?, now(), now(), FALSE, 0, 0, 0, 0)",
        [pid, author, handle, text],
    )


def _embed(con, pid, vector):
    con.execute("INSERT INTO embeddings VALUES (?, ?, now())", [pid, vector])


def test_near_identical_posts_by_two_members_are_an_exhibit(con):
    """v2's text trace links accounts by what they WROTE; that co-action has to
    reach the reader, or a text-linked group arrives with no joint evidence."""
    text = "Vote for real change across the whole county this August"
    for pid, author, handle in (("d1", "a1", "member_one"), ("d2", "a2", "member_two")):
        _post(con, pid, author, handle, text)
        _embed(con, pid, [0.0, 1.0, 0.0])
    _embed(con, "m1", [1.0, 0.0, 0.0])

    packet = dossier.build(con, MEMBERS)[0]

    assert len(packet["shared_texts"]) == 1
    family = packet["shared_texts"][0]
    assert family["n_members"] == 2 and family["n_posts"] == 2
    assert family["text"] == text


def test_a_packet_without_embeddings_still_builds(con):
    packet = dossier.build(con, MEMBERS)[0]

    assert packet["shared_texts"] == []


def test_one_member_repeating_themselves_is_not_co_action():
    import numpy as np

    posts = pd.DataFrame({
        "author_id": ["a1", "a1"], "author_handle": ["h", "h"],
        "text": ["same words said twice here", "same words said twice here"],
        "created_at": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
    })

    got = dossier.near_duplicate_families(posts, np.array([[1.0, 0.0], [1.0, 0.0]]))

    assert got.empty


def test_families_join_transitively_and_rank_by_members():
    """A~B and B~C is one family even when A and C sit below the cut - the same
    connectivity the text trace itself builds on."""
    import numpy as np

    angle = np.deg2rad([0, 20, 40])
    vectors = np.vstack([
        np.column_stack([np.cos(angle), np.sin(angle), np.zeros(3)]),
        [[0.0, 0.0, 1.0], [0.0, 0.1, 1.0]],
    ])
    posts = pd.DataFrame({
        "author_id": ["a", "b", "c", "d", "e"],
        "author_handle": ["a", "b", "c", "d", "e"],
        "text": ["t1", "t2", "t3", "t4", "t5"],
        "created_at": pd.date_range("2026-01-01", periods=5, tz="UTC"),
    })

    got = dossier.near_duplicate_families(posts, vectors)

    assert got["n_members"].tolist() == [3, 2]
    assert got.iloc[0]["text"] == "t1"


def test_a_packet_carries_the_evidence_a_reader_needs(con):
    """Intent is not in the co-action graph, so the packet has to carry the
    content the graph fired on - otherwise there is nothing to adjudicate."""
    packets = dossier.build(con, MEMBERS)

    assert len(packets) == 1
    p = packets[0]
    assert p["cluster_id"] == 0 and p["size"] == 2
    assert p["shared_objects"], "what they jointly amplified"
    assert p["representative_posts"], "what they themselves post"
    assert p["amplification_targets"], "whose content they push"
    assert "provenance" in p and "kenya" in p


def test_only_JOINTLY_amplified_objects_are_exhibits(con):
    """An object one member touched is not evidence of coordination. Showing it
    beside the shared ones would invite a reader to over-read the packet."""
    objects = dossier.build(con, MEMBERS)[0]["shared_objects"]

    ids = {o["platform_post_id"] for o in objects}
    assert "OBJ1" in ids
    assert "OBJ2" not in ids
    assert next(o for o in objects if o["platform_post_id"] == "OBJ1")["n_members"] == 2


def test_targets_are_marked_member_or_outsider(con):
    """A cluster amplifying itself and one amplifying a politician are different
    claims, and the flag is what lets a reader tell them apart at a glance."""
    targets = dossier.build(con, MEMBERS)[0]["amplification_targets"]

    outsider = next(t for t in targets if t["target_handle"] == "big_account")
    assert outsider["target_is_member"] is False


def test_provenance_is_aggregated_not_per_account(con):
    """The question is 'were these provisioned together', which is a property of
    the group. Per-account profiling is authenticity.py's job."""
    prov = dossier.build(con, MEMBERS)[0]["provenance"]

    assert prov["accounts_profiled"] == 2
    assert prov["share_empty_bio"] == 0.5
    assert prov["share_default_image"] == 1.0
    assert "author_id" not in prov and "handle" not in prov


def test_kenya_share_is_evidence_not_a_filter(con):
    """The gate has documented blind spots, so the packet reports the share and
    leaves the judgement to the reader - it must not drop the cluster."""
    p = dossier.build(con, MEMBERS)[0]

    assert p["kenya"]["kenya_share"] == 0.5
    assert p["shared_objects"], "an off-topic cluster is still packaged"


def test_cluster_ids_restrict_the_build(con):
    """Adjudication is the expensive step; callers narrow first."""
    assert dossier.build(con, MEMBERS, cluster_ids=[999]) == []
    assert len(dossier.build(con, MEMBERS, cluster_ids=[0])) == 1


def test_no_clusters_is_an_empty_list(con):
    assert dossier.build(con, pd.DataFrame(columns=["cluster_id", "author_id"])) == []


def test_provenance_scopes_the_author_read_before_deduplicating(con):
    """Profiling ~300 cluster members must not deduplicate all ~461k known
    accounts first. Measured at 383s of a 1,370s build before the semi-join
    moved inside the CTE. Asserted on behaviour rather than on SQL text: an
    author outside the cluster must never reach the packet."""
    con.execute(
        "INSERT INTO authors VALUES ('x', 'stranger', 'nobody', 'bio',"
        " 'http://img/x.png', TIMESTAMPTZ '2019-01-01', now(), 5, 5)"
    )

    prov = dossier.build(con, MEMBERS)[0]["provenance"]

    assert prov["accounts_profiled"] == 2, "the stranger must not be profiled"


def test_object_scope_is_joined_before_the_dedup_window():
    """Same invariant as everywhere else: QUALIFY runs after the window, so a
    scope applied outside deduplicates the whole corpus and then discards it.
    Measured on the dossier path, the unscoped version took a four-cluster
    build past 22 minutes."""
    sql = co._latest_posts_cte("x", lookback_days=0, post_scope="_ids")

    assert "SEMI JOIN _ids USING (platform_post_id)" in sql
    assert sql.index("SEMI JOIN") < sql.index("QUALIFY")


def test_both_scopes_can_apply_at_once():
    sql = co._latest_posts_cte("x", author_scope="_a", post_scope="_p")

    assert "SEMI JOIN _a USING (author_id)" in sql
    assert "SEMI JOIN _p USING (platform_post_id)" in sql
    assert max(sql.index("_a"), sql.index("_p")) < sql.index("QUALIFY")


def test_the_dashboard_payload_never_carries_dossier_fields():
    """The safety boundary. `dashboard.py` is public-by-construction and this
    module is the opposite; nothing here may be copied across."""
    from kma import dashboard as d

    forbidden = {
        "shared_objects", "representative_posts", "amplification_targets",
        "object_text", "object_author", "target_handle", "author_handle", "text",
    }
    assert not (set(d.TRIAGE_COLUMNS) & forbidden)

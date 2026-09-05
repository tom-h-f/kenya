"""Unit tests for kma.adjudicate_run (no R2, no API calls).

Layer three on a timer. What matters here is the queueing - which clusters get
a reader's attention and in what order - and that a missing API key fails loudly
rather than quietly producing nothing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kma import adjudicate_run as ar


def _members(spec: dict) -> pd.DataFrame:
    """spec: {cluster_id: (size, n_channels)}"""
    rows = []
    for cid, (size, n_ch) in spec.items():
        for i in range(size):
            rows.append({"cluster_id": cid, "author_id": f"c{cid}a{i}",
                         "n_channels": n_ch})
    return pd.DataFrame(rows)


def test_only_corroborated_clusters_are_queued():
    """Corroborated is the population the dashboard already publishes, so it is
    the one a reader's time is worth spending on first."""
    got = ar.candidates(_members({0: (10, 2), 1: (8, 1), 2: (5, 2)}), None)

    assert set(got) == {0, 2}


def test_candidates_are_ranked_by_size_not_by_index():
    """The inauthenticity index is a within-run percentile blend that flatters
    tiny clusters - a 3-account group trivially looks concealed and homogeneous.
    The ground-truth work found the substantive clusters are the large ones."""
    got = ar.candidates(_members({0: (5, 2), 1: (30, 2), 2: (12, 2)}), None)

    assert got == [1, 2, 0]


def test_the_queue_is_capped():
    """Reading is the expensive step and the cap is what bounds the bill."""
    got = ar.candidates(
        _members({i: (10 + i, 2) for i in range(20)}), None, max_clusters=5
    )

    assert len(got) == 5


def test_a_kenya_relevant_single_channel_cluster_can_be_dragged_in():
    """The corroboration gate is about statistical support, not topic. A
    single-channel cluster that is clearly about Kenya is worth a look, and the
    ground-truth work showed corroborated clusters are the LEAST on-topic."""
    members = _members({0: (10, 2), 1: (8, 1)})
    cards = pd.DataFrame([{"cluster_id": 0, "kenya_share": 0.02},
                          {"cluster_id": 1, "kenya_share": 0.60}])

    assert set(ar.candidates(members, cards, min_kenya=0.5)) == {0, 1}
    assert set(ar.candidates(members, cards, min_kenya=0.0)) == {0}


def test_no_clusters_is_an_empty_queue():
    assert ar.candidates(pd.DataFrame(columns=["cluster_id", "author_id"]), None) == []


def test_the_triage_clustering_is_preferred_over_the_published_one(monkeypatch):
    """That is where the recall is: 18.6% -> ~40% against confirmed operations
    at zero measured null yield. The published strict clustering exists for
    series continuity, not because it finds more."""
    def fake_latest(con, kind="clusters", platform="x", **kw):
        frame = (
            _members({0: (5, 2)}) if kind == "triage_clusters"
            else _members({9: (4, 2)})
        )
        return type("R", (), {"df": staticmethod(lambda: frame)})()

    monkeypatch.setattr(ar, "coordination_run_latest", fake_latest)
    monkeypatch.setattr("kma.db.prefix_readable", lambda con, src: True)

    frame, source = ar._cluster_source(None, "x")

    assert source == "triage_clusters"
    assert set(frame["cluster_id"]) == {0}


def test_it_falls_back_to_the_published_clustering(monkeypatch):
    """Every run before the dual-resolution change wrote only kind=clusters, so
    a bucket without a triage prefix must still adjudicate."""
    def fake_latest(con, kind="clusters", platform="x", **kw):
        frame = pd.DataFrame() if kind == "triage_clusters" else _members({9: (4, 2)})
        return type("R", (), {"df": staticmethod(lambda: frame)})()

    monkeypatch.setattr(ar, "coordination_run_latest", fake_latest)
    monkeypatch.setattr("kma.db.prefix_readable", lambda con, src: True)

    frame, source = ar._cluster_source(None, "x")

    assert source == "clusters"
    assert set(frame["cluster_id"]) == {9}


def test_an_empty_bucket_yields_no_source(monkeypatch):
    monkeypatch.setattr("kma.db.prefix_readable", lambda con, src: False)

    frame, source = ar._cluster_source(None, "x")

    assert source == "none"
    assert frame.empty


def test_a_missing_api_key_fails_loudly(monkeypatch):
    """A scheduled pass that silently produces no verdicts looks exactly like a
    pass that found nothing to say. It must fail instead."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        ar, "_cluster_source",
        lambda con, platform: (_members({0: (5, 2)}), "triage_clusters"),
    )
    monkeypatch.setattr(
        ar, "coordination_run_latest",
        lambda con, kind="clusters", platform="x", **kw: type(
            "R", (), {"df": staticmethod(pd.DataFrame)})(),
    )
    monkeypatch.setattr(ar.dossier, "build", lambda *a, **k: [{"cluster_id": 0}])

    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        ar.run(None, persist=False)


def test_dossiers_are_built_in_ONE_batched_call(monkeypatch):
    """A dossier build is three fixed whole-prefix scans whose cost does not
    grow with cluster count - 1,333s for four clusters, about the same for
    forty. Per-cluster builds would pay that repeatedly."""
    calls = {"n": 0, "ids": None}

    def fake_build(con, members, scorecards=None, platform="x", cluster_ids=None, **kw):
        calls["n"] += 1
        calls["ids"] = cluster_ids
        return [{"cluster_id": c} for c in cluster_ids]

    monkeypatch.setattr(
        ar, "_cluster_source",
        lambda con, platform: (_members({0: (9, 2), 1: (7, 2)}), "triage_clusters"),
    )
    monkeypatch.setattr(
        ar, "coordination_run_latest",
        lambda con, kind="clusters", platform="x", **kw: type(
            "R", (), {"df": staticmethod(pd.DataFrame)})(),
    )
    monkeypatch.setattr(ar.dossier, "build", fake_build)
    monkeypatch.setattr(
        ar.adj, "judge_all",
        lambda packets, client=None: [
            {"cluster_id": p["cluster_id"], "cluster_type": "engagement_pod"}
            for p in packets
        ],
    )

    out = ar.run(None, persist=False, client=object())

    assert calls["n"] == 1, "one build for every candidate, not one per cluster"
    assert set(calls["ids"]) == {0, 1}
    assert out["n_verdicts"] == 2
    assert out["by_type"] == {"engagement_pod": 2}


def test_a_dry_run_writes_nothing(monkeypatch):
    monkeypatch.setattr(
        ar, "_cluster_source",
        lambda con, platform: (_members({0: (5, 2)}), "triage_clusters"),
    )
    monkeypatch.setattr(
        ar, "coordination_run_latest",
        lambda con, kind="clusters", platform="x", **kw: type(
            "R", (), {"df": staticmethod(pd.DataFrame)})(),
    )
    monkeypatch.setattr(ar.dossier, "build", lambda *a, **k: [{"cluster_id": 0}])
    monkeypatch.setattr(
        ar.adj, "judge_all",
        lambda packets, client=None: [{"cluster_id": 0, "cluster_type": "unclear"}],
    )

    def boom(*a, **k):
        raise AssertionError("a dry run must not persist")

    monkeypatch.setattr(ar.co, "persist_verdicts", boom)

    assert ar.run(None, persist=False, client=object())["verdicts_key"] is None

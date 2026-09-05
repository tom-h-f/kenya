"""Unit tests for kma.scorecard_run (no R2, no scoring).

Scoring lives behind its own entrypoint because it costs ~40x a coordination
pass and reads the run back out of R2 rather than rebuilding it. These tests
pin that contract: the projection is never recomputed, a dead channel does not
take the pass down, and nothing is written without --persist.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from kma import coordination as co
from kma import scorecard_run as sr

MEMBERS = pd.DataFrame(
    [{"cluster_id": 0, "author_id": a, "size": 3} for a in ("a", "b", "c")]
    + [{"cluster_id": 1, "author_id": a, "size": 2} for a in ("d", "e")]
)
EDGES = pd.DataFrame([{"src": "a", "dst": "b", "weight": 3.0}])
CARDS = pd.DataFrame(
    [
        {"cluster_id": 0, "size": 3, "inauthenticity_index": 0.9},
        {"cluster_id": 1, "size": 2, "inauthenticity_index": 0.2},
    ]
)


class _Rel:
    """`coordination_run_latest` returns a relation and every caller here only
    ever calls `.df()` on it."""

    def __init__(self, frame):
        self._frame = frame

    def df(self):
        return self._frame.copy()


@pytest.fixture
def stub(monkeypatch):
    calls = {"scorecards": [], "persist": 0, "build_layers": 0}

    def fake_latest(con, kind="clusters", platform="x", channel="*", method="*"):
        if kind == "clusters":
            return _Rel(MEMBERS)
        if channel == "co_reply":
            raise duckdb.IOException("no files found")
        return _Rel(EDGES)

    def fake_scorecards(con, members, layers, platform="x", **kw):
        calls["scorecards"].append((len(members), sorted(layers)))
        return CARDS

    def fake_persist(con, cards, members, platform="x"):
        calls["persist"] += 1
        return "coordination/platform=x/kind=scorecards/dt=2026-08-14/run=r.parquet"

    def boom_build_layers(*a, **k):
        calls["build_layers"] += 1
        raise AssertionError("scorecard_run must not rebuild the projection")

    monkeypatch.setattr(sr, "coordination_run_latest", fake_latest)
    monkeypatch.setattr(co, "scorecards", fake_scorecards)
    monkeypatch.setattr(co, "persist_scorecards", fake_persist)
    monkeypatch.setattr(co, "build_layers", boom_build_layers)
    monkeypatch.setattr(sr.cr, "tune", lambda con: None)
    return calls


def test_dry_run_scores_but_writes_nothing(stub):
    out = sr.run(None, persist=False)

    assert stub["scorecards"], "it still scores - the dry run is for inspection"
    assert stub["persist"] == 0
    assert out["scorecards_key"] is None
    assert out["n_clusters"] == 2


def test_persist_writes_one_scorecard_run(stub):
    out = sr.run(None, persist=True)

    assert stub["persist"] == 1
    assert "kind=scorecards" in out["scorecards_key"]


def test_the_projection_is_never_rebuilt(stub):
    """The entire reason this is a separate entrypoint: a coordination pass has
    already done the expensive part and persisted the result."""
    sr.run(None, persist=True)

    assert stub["build_layers"] == 0


def test_members_are_deduplicated_to_one_row_per_account(stub):
    """`kind=clusters` is one row per member and carries per-cluster columns on
    every row; passing it through unfiltered would double-count accounts."""
    sr.run(None, persist=False)

    n_members, _ = stub["scorecards"][0]
    assert n_members == 5


def test_a_channel_with_no_persisted_partition_is_skipped(stub):
    """A channel that validated no edges has no partition, and its glob raises
    rather than returning nothing. One dead layer must not kill the pass."""
    out = sr.run(None, channels=["co_retweet", "co_reply"], persist=False)

    _, channels = stub["scorecards"][0]
    assert channels == ["co_retweet"]
    assert out["channels"] == ["co_retweet"]


def test_no_persisted_clusters_is_a_clean_noop(stub, monkeypatch):
    monkeypatch.setattr(
        sr, "coordination_run_latest", lambda *a, **k: _Rel(MEMBERS.iloc[:0])
    )

    out = sr.run(None, persist=True)

    assert out["n_clusters"] == 0
    assert out["scorecards_key"] is None
    assert stub["persist"] == 0


def test_cli_rejects_unknown_channel():
    with pytest.raises(SystemExit):
        sr.main(["--channels", "co_retweet,not_a_channel"])

"""Reading ONE coordination run back, rather than a union of every run.

`cluster_id` is a per-run Leiden label, so the `latest_coordination_*` helpers -
which partition per entity and keep the newest row for each - never collapse to
a single clustering. Measured on the live corpus 2026-08-12: 44,585 member rows
across 1,078 cluster ids drawn from 47 passes, against a true latest run of
1,043 rows and 128 clusters.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd
import pytest

from kma import coordination as co
from kma import db

RUN_1 = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
RUN_2 = RUN_1 + timedelta(hours=6)


@pytest.fixture
def con(monkeypatch):
    c = duckdb.connect()
    # Two passes over the same three accounts. Leiden relabels between runs, so
    # the same group is cluster 0 in run 1 and cluster 7 in run 2, and a fourth
    # account joins. This is exactly what the live prefix looks like.
    members = pd.DataFrame(
        [
            {"cluster_id": 0, "author_id": "a", "size": 3, "n_channels": 1, "computed_at": RUN_1},
            {"cluster_id": 0, "author_id": "b", "size": 3, "n_channels": 1, "computed_at": RUN_1},
            {"cluster_id": 0, "author_id": "c", "size": 3, "n_channels": 1, "computed_at": RUN_1},
            {"cluster_id": 7, "author_id": "a", "size": 4, "n_channels": 2, "computed_at": RUN_2},
            {"cluster_id": 7, "author_id": "b", "size": 4, "n_channels": 2, "computed_at": RUN_2},
            {"cluster_id": 7, "author_id": "c", "size": 4, "n_channels": 2, "computed_at": RUN_2},
            {"cluster_id": 7, "author_id": "d", "size": 4, "n_channels": 2, "computed_at": RUN_2},
        ]
    )
    edges = pd.DataFrame(
        [
            # Validated in run 1 and never again - the stickiness case.
            {"src": "a", "dst": "z", "channel": "co_retweet", "method": "svn_bonf",
             "weight": 2, "computed_at": RUN_1},
            {"src": "a", "dst": "b", "channel": "co_retweet", "method": "svn_bonf",
             "weight": 5, "computed_at": RUN_1},
            {"src": "a", "dst": "b", "channel": "co_retweet", "method": "svn_bonf",
             "weight": 6, "computed_at": RUN_2},
        ]
    )
    c.register("_members", members)
    c.register("_edges", edges)

    def fake_source(kind="edges", platform="*", channel="*", method="*"):
        return "_members" if kind == "clusters" else "_edges"

    monkeypatch.setattr(db, "coordination_source", fake_source)
    return c


def test_run_latest_returns_exactly_one_run(con):
    got = db.coordination_run_latest(con, "clusters").df()

    assert got["computed_at"].nunique() == 1
    assert set(got["author_id"]) == {"a", "b", "c", "d"}
    assert set(got["cluster_id"]) == {7}


def test_sticky_helper_unions_every_run(con):
    """The defect, pinned so nobody 'fixes' the new helper back into this."""
    got = db.latest_coordination_clusters(con).df()

    assert got["computed_at"].nunique() == 2
    assert set(got["cluster_id"]) == {0, 7}
    # `a` appears once per cluster id it was ever assigned.
    assert (got["author_id"] == "a").sum() == 2
    assert len(got) > len(db.coordination_run_latest(con, "clusters").df())


def test_run_latest_edges_drop_a_pair_that_stopped_validating(con):
    got = db.coordination_run_latest(con, "edges").df()

    pairs = set(zip(got["src"], got["dst"]))
    assert ("a", "b") in pairs
    assert ("a", "z") not in pairs, "an edge only validated in an older run is not current"

    sticky = set(
        zip(*db.latest_coordination_edges(con).df()[["src", "dst"]].values.T)
    )
    assert ("a", "z") in sticky, "the sticky helper keeps it - that is its documented flaw"


def test_run_latest_edges_keep_every_channel(con):
    """`persist_edges` takes its own `now` per channel x method, so ranking them
    together would return whichever partition was written last."""
    edges = pd.DataFrame(
        [
            {"src": "a", "dst": "b", "channel": "co_retweet", "method": "svn_bonf",
             "computed_at": RUN_2},
            {"src": "c", "dst": "d", "channel": "co_reply", "method": "svn_bonf",
             "computed_at": RUN_2 + timedelta(seconds=3)},
        ]
    )
    con.register("_edges2", edges)

    got = con.sql(
        """
        SELECT * FROM _edges2
        QUALIFY dense_rank() OVER (
            PARTITION BY channel, method ORDER BY computed_at DESC) = 1
        """
    ).df()
    assert set(got["channel"]) == {"co_retweet", "co_reply"}


def test_stable_cluster_id_tracks_membership_not_the_leiden_label():
    run1 = pd.DataFrame(
        [{"cluster_id": 0, "author_id": a} for a in ("a", "b", "c")]
    )
    run2 = pd.DataFrame(
        [{"cluster_id": 7, "author_id": a} for a in ("c", "a", "b")]
    )

    id1 = co.attach_stable_cluster_ids(run1)["stable_cluster_id"].unique()
    id2 = co.attach_stable_cluster_ids(run2)["stable_cluster_id"].unique()

    assert len(id1) == len(id2) == 1
    assert id1[0] == id2[0], "same members, different Leiden label -> same stable id"


def test_stable_cluster_id_changes_when_membership_changes():
    a = pd.DataFrame([{"cluster_id": 0, "author_id": x} for x in ("a", "b", "c")])
    b = pd.DataFrame([{"cluster_id": 0, "author_id": x} for x in ("a", "b", "c", "d")])

    assert (
        co.attach_stable_cluster_ids(a)["stable_cluster_id"].iloc[0]
        != co.attach_stable_cluster_ids(b)["stable_cluster_id"].iloc[0]
    )


def test_attach_stable_cluster_ids_handles_an_empty_frame():
    out = co.attach_stable_cluster_ids(pd.DataFrame(columns=["cluster_id", "author_id"]))
    assert out.empty
    assert "stable_cluster_id" in out.columns

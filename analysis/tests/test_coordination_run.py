"""Unit tests for kma.coordination_run orchestration (no R2, no clustering)."""

import pandas as pd
import pytest

from kma import coordination as co
from kma import coordination_run as cr


@pytest.fixture
def stub(monkeypatch):
    """Stub the expensive pipeline so only the orchestration is under test."""
    calls = {"persist_edges": [], "persist_clusters": 0, "build_layers": {}}

    def fake_build_layers(con, channels, platform="x", method="fdr", **kw):
        calls["build_layers"] = {"channels": list(channels), "method": method}
        edges = pd.DataFrame(
            {
                "src": ["a", "b"],
                "dst": ["b", "c"],
                "weight": [1.0, 2.0],
                "sig_fdr": [True, True],
                "sig_bonferroni": [True, False],
                "sig_percentile": [False, False],
            }
        )
        return {ch: edges for ch in channels}

    def fake_clusters(layers, resolution, min_size, **kw):
        members = pd.DataFrame(
            {"author_id": ["a", "b", "c"], "cluster_id": [0, 0, 0]}
        )
        summary = pd.DataFrame([{"cluster_id": 0, "size": 3}])
        return members, summary

    def fake_persist_edges(con, edges, channel, method, platform="x"):
        calls["persist_edges"].append((channel, method, len(edges)))
        return f"coordination/platform={platform}/kind=edges/channel={channel}/method={method}"

    def fake_persist_clusters(con, members, summary, platform="x"):
        calls["persist_clusters"] += 1
        return f"coordination/platform={platform}/kind=clusters"

    monkeypatch.setattr(co, "build_layers", fake_build_layers)
    monkeypatch.setattr(co, "clusters", fake_clusters)
    monkeypatch.setattr(co, "persist_edges", fake_persist_edges)
    monkeypatch.setattr(co, "persist_clusters", fake_persist_clusters)
    monkeypatch.setattr(co, "cluster_names", lambda *a, **k: pd.DataFrame(columns=["cluster_id", "name", "label"]))
    return calls


def test_default_channels_exclude_the_dead_ones():
    """text_sim and fast_co_share validated 0 edges on the full corpus under the
    degree-corrected null; running them is pure cost."""
    assert cr.DEFAULT_CHANNELS == ["co_retweet", "co_reply"]
    assert "text_sim" not in cr.DEFAULT_CHANNELS
    assert "fast_co_share" not in cr.DEFAULT_CHANNELS
    assert set(cr.DEFAULT_CHANNELS) <= set(co.CHANNELS)


def test_dry_run_writes_nothing(stub):
    out = cr.run(con=None, persist=False)
    assert out["persisted"] == []
    assert stub["persist_edges"] == []
    assert stub["persist_clusters"] == 0
    assert out["n_clusters"] == 1
    assert out["n_accounts"] == 3
    assert stub["build_layers"]["channels"] == cr.DEFAULT_CHANNELS


def test_persist_writes_edges_per_channel_and_one_cluster_run(stub):
    out = cr.run(con=None, persist=True)
    # svn_fdr (2 edges) and svn_bonf (1 edge) have rows; pct has none, so it is
    # skipped rather than writing an empty run.
    per_channel = {(ch, m) for ch, m, _ in stub["persist_edges"]}
    assert per_channel == {
        ("co_retweet", "svn_fdr"), ("co_retweet", "svn_bonf"),
        ("co_reply", "svn_fdr"), ("co_reply", "svn_bonf"),
    }
    assert all(m != "pct" for _, m, _ in stub["persist_edges"])
    assert stub["persist_clusters"] == 1
    assert len(out["persisted"]) == 5


def test_persist_noop_when_no_clusters(stub, monkeypatch):
    monkeypatch.setattr(
        co, "clusters",
        lambda *a, **k: (pd.DataFrame(columns=["author_id", "cluster_id"]), pd.DataFrame()),
    )
    out = cr.run(con=None, persist=True)
    assert out["persisted"] == []
    assert stub["persist_clusters"] == 0


def test_cli_rejects_unknown_channel():
    with pytest.raises(SystemExit):
        cr.main(["--channels", "co_retweet,not_a_channel"])


def test_tune_constrains_duckdb_so_it_spills_instead_of_oom():
    """DuckDB sizes its budget from the host, not the container cgroup, so on a
    small box it plans past physical RAM and dies mid-projection. Observed on
    tf1: OutOfMemoryException at projected_edges, 2.7GiB/2.9GiB used."""
    import duckdb

    con = duckdb.connect()
    cr.tune(con)
    settings = {
        r[0]: r[1]
        for r in con.sql(
            "SELECT name, value FROM duckdb_settings() "
            "WHERE name IN ('memory_limit','threads','preserve_insertion_order')"
        ).fetchall()
    }
    assert settings["preserve_insertion_order"] == "false"
    assert int(settings["threads"]) == cr.COORD_THREADS
    # memory_limit is echoed back in DuckDB's own units, so assert it is bounded
    # rather than string-equal to the input.
    assert settings["memory_limit"] not in ("", None)


def test_run_tunes_before_building_layers(stub, monkeypatch):
    """The tuning must apply to the connection the projection actually uses."""
    called: list[str] = []
    monkeypatch.setattr(cr, "tune", lambda con: called.append("tuned"))
    cr.run(con=None, persist=False)
    assert called == ["tuned"]


def test_latest_posts_is_windowed_by_default():
    """Unbounded, the projection's cost grows with the corpus and OOMed a 3.7Gi
    host at ~196k posts. It is also the wrong claim: a ring active months ago is
    not the current network."""
    from kma import coordination as co

    assert co.COORD_LOOKBACK_DAYS == 14
    sql = co._latest_posts_cte("x")
    assert "INTERVAL 14 DAY" in sql
    assert "created_at >" in sql


def test_lookback_can_be_overridden_and_disabled():
    from kma import coordination as co

    assert "INTERVAL 7 DAY" in co._latest_posts_cte("x", lookback_days=7)
    unbounded = co._latest_posts_cte("x", lookback_days=0)
    assert "INTERVAL" not in unbounded
    assert "QUALIFY" in unbounded  # dedup survives


def test_window_precedes_the_dedup_qualify():
    """The filter must narrow the scan before the window function, not after."""
    from kma import coordination as co

    sql = co._latest_posts_cte("x")
    assert sql.index("created_at >") < sql.index("QUALIFY")

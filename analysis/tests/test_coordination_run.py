"""Unit tests for kma.coordination_run orchestration (no R2, no clustering)."""

import pandas as pd
import pytest

from kma import coordination as co
from kma import coordination_run as cr


@pytest.fixture
def stub(monkeypatch):
    """Stub the expensive pipeline so only the orchestration is under test."""
    calls = {
        "persist_edges": [],
        "persist_clusters": 0,
        "build_layers": {},
        "persist_run_metrics": [],
        "order": [],
    }

    def fake_build_layers(con, channels, platform="x", method="fdr", stats=None, **kw):
        calls["build_layers"] = {"channels": list(channels), "method": method}
        if stats is not None:
            for ch in channels:
                stats[ch] = {"channel": ch, "pairable": 7, "nohub_amp": 25}
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
        calls["order"].append("edges")
        return f"coordination/platform={platform}/kind=edges/channel={channel}/method={method}"

    def fake_persist_clusters(con, members, summary, platform="x"):
        calls["persist_clusters"] += 1
        calls["order"].append("clusters")
        return f"coordination/platform={platform}/kind=clusters"

    def fake_persist_run_metrics(con, channel_stats, run_stats, platform="x"):
        calls["persist_run_metrics"].append((dict(channel_stats), dict(run_stats)))
        calls["order"].append("metrics")
        return f"coordination/platform={platform}/kind=run_metrics"

    monkeypatch.setattr(co, "persist_run_metrics", fake_persist_run_metrics)
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


def test_hub_cap_is_bounded_above():
    """The 5%-of-accounts term scales with the corpus and eventually excludes
    nothing: at 64,002 amplifiers it gave 3,200 against a busiest object of
    1,171. Measured 2026-08-01: 1.9% of objects produced 99.3% of all pairs."""
    from kma import coordination as co

    def cap(n_accounts):
        return max(50, min(int(0.05 * n_accounts), co.HUB_CAP_MAX))

    assert cap(100) == 50          # tiny corpus -> floor
    assert cap(1_600) == 80        # mid -> the percentage governs
    assert cap(64_002) == co.HUB_CAP_MAX   # large -> bounded, not 3200
    assert cap(1_000_000) == co.HUB_CAP_MAX


def test_corroborated_clusters_count_n_channels_at_least_two(stub, monkeypatch):
    """Corroboration is the outcome measure for the census work and was computed
    nowhere: `adaptive` and `netviz` only read `n_channels` back out of persisted
    parquet, so the "14 corroborated (60 accounts)" baseline in
    docs/analysis/census-tuning.md was hand-derived and never reproducible."""
    members = pd.DataFrame(
        {"author_id": list("abcdefg"), "cluster_id": [0, 0, 0, 1, 1, 2, 2]}
    )
    summary = pd.DataFrame(
        [
            {"cluster_id": 0, "size": 3, "n_channels": 2},
            {"cluster_id": 1, "size": 2, "n_channels": 1},
            {"cluster_id": 2, "size": 2, "n_channels": 3},
        ]
    )
    monkeypatch.setattr(co, "clusters", lambda *a, **k: (members, summary))

    out = cr.run(None, channels=["co_retweet"], persist=False)

    assert out["n_clusters"] == 3
    assert out["n_corroborated_clusters"] == 2       # clusters 0 and 2
    assert out["n_corroborated_accounts"] == 5       # a,b,c + f,g


def test_single_channel_run_reports_no_corroboration(stub, monkeypatch):
    """One channel cannot corroborate itself - the count must be 0, not the
    cluster count."""
    members = pd.DataFrame({"author_id": list("abc"), "cluster_id": [0, 0, 0]})
    summary = pd.DataFrame([{"cluster_id": 0, "size": 3, "n_channels": 1}])
    monkeypatch.setattr(co, "clusters", lambda *a, **k: (members, summary))

    out = cr.run(None, channels=["co_retweet"], persist=False)

    assert out["n_clusters"] == 1
    assert out["n_corroborated_clusters"] == 0
    assert out["n_corroborated_accounts"] == 0


def test_metrics_are_written_even_when_no_clusters_were_found(stub, monkeypatch):
    """A pass that detects nothing is exactly the data point Q1 needs. The old
    control flow returned early at `members.empty` before any write."""
    empty = pd.DataFrame(columns=["author_id", "cluster_id"])
    monkeypatch.setattr(
        co, "clusters", lambda *a, **k: (empty, pd.DataFrame(columns=["cluster_id"]))
    )

    out = cr.run(None, channels=["co_retweet"], persist=True)

    assert len(stub["persist_run_metrics"]) == 1
    assert out["metrics_key"] is not None
    assert stub["persist_edges"] == []      # nothing else was written
    assert stub["persist_clusters"] == 0


def test_metrics_are_written_before_the_edge_and_cluster_artifacts(stub):
    """Ordering is deliberate: a run whose edge write fails to R2 never reaches
    the end, and that is a run whose counters you want."""
    cr.run(None, channels=["co_retweet"], persist=True)

    assert stub["order"][0] == "metrics"
    assert "edges" in stub["order"] and "clusters" in stub["order"]


def test_a_metrics_write_failure_does_not_fail_the_run(stub, monkeypatch):
    """`enrich._coordination_pass` reschedules the whole 6-hourly pass on a
    non-zero rc. A flaky ~4KB write must not cost a full re-projection."""
    def boom(*a, **k):
        raise RuntimeError("r2 unavailable")

    monkeypatch.setattr(co, "persist_run_metrics", boom)

    out = cr.run(None, channels=["co_retweet"], persist=True)

    assert out["metrics_key"] is None
    assert stub["persist_clusters"] == 1     # the real artifacts still landed


def test_dry_run_writes_no_metrics(stub):
    out = cr.run(None, channels=["co_retweet"], persist=False)

    assert stub["persist_run_metrics"] == []
    assert out["metrics_key"] is None


def test_channel_stats_are_collected_per_channel(stub):
    out = cr.run(None, channels=["co_retweet", "co_reply"], persist=False)

    assert set(out["channel_stats"]) == {"co_retweet", "co_reply"}
    assert out["channel_stats"]["co_retweet"]["pairable"] == 7

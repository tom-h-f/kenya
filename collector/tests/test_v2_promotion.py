"""Promotion re-pointed at v2 centrality ranks, and the cap that bounds it."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import duckdb
import pyarrow as pa

from kenya_monitor.adaptive import DynamicEntry, cap_new_accounts, promote, v2_accounts

NOW = datetime.now(timezone.utc)


def _scores(rows: list[tuple[str, float, float, datetime]]) -> pa.Table:
    return pa.table(
        {
            "user_id": pa.array([r[0] for r in rows], type=pa.string()),
            "centrality": pa.array([r[1] for r in rows], type=pa.float64()),
            "kenya_share": pa.array([r[2] for r in rows], type=pa.float64()),
            "computed_at": pa.array([r[3] for r in rows], type=pa.timestamp("us", tz="UTC")),
        }
    )


def _authors(rows: list[tuple[str, str]]) -> pa.Table:
    return pa.table(
        {
            "platform": pa.array(["x"] * len(rows), type=pa.string()),
            "platform_user_id": pa.array([r[0] for r in rows], type=pa.string()),
            "handle": pa.array([r[1] for r in rows], type=pa.string()),
            "collected_at": pa.array([NOW] * len(rows), type=pa.timestamp("us", tz="UTC")),
        }
    )


def _con(scores: pa.Table, authors: pa.Table) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("scores_tbl", scores)
    con.register("authors_tbl", authors)
    return con


def test_latest_run_only_ordered_by_centrality_and_kenya_gated(tmp_path):
    old = NOW - timedelta(hours=30)
    con = _con(
        _scores([
            ("9", 9.0, 0.9, old),
            ("1", 0.2, 0.9, NOW),
            ("2", 0.8, 0.9, NOW),
            ("3", 0.5, 0.01, NOW),
        ]),
        _authors([("1", "low"), ("2", "high"), ("3", "offtopic"), ("9", "stale")]),
    )

    got = v2_accounts(con, "scores_tbl", "authors_tbl", state_path=tmp_path / "v2.json")

    assert got == ["high", "low"]


def test_a_stale_run_promotes_nobody(tmp_path):
    con = _con(
        _scores([("1", 1.0, 0.9, NOW - timedelta(days=8))]),
        _authors([("1", "a")]),
    )

    assert v2_accounts(con, "scores_tbl", "authors_tbl", state_path=tmp_path / "v2.json") == []


def test_handles_are_resolved_once_per_scores_run(tmp_path):
    """The authors join is a full-prefix scan in production, so a second call
    on the same run must not repeat it."""
    state = tmp_path / "v2.json"
    con = _con(_scores([("1", 1.0, 0.9, NOW)]), _authors([("1", "a")]))
    assert v2_accounts(con, "scores_tbl", "authors_tbl", state_path=state) == ["a"]

    con.unregister("authors_tbl")
    assert v2_accounts(con, "scores_tbl", "authors_tbl", state_path=state) == ["a"]
    assert json.loads(state.read_text())["handles"] == ["a"]


def test_cap_keeps_live_accounts_and_admits_only_n_new():
    live = [DynamicEntry("b", "account", "v2-centrality", NOW.isoformat(), NOW.isoformat())]

    assert cap_new_accounts(["a", "B", "c", "d"], live, max_new=2) == ["a", "B", "c"]


def test_promote_adds_v2_accounts_under_the_cap_and_tags_their_source(tmp_path, monkeypatch):
    import kenya_monitor.adaptive as ad

    monkeypatch.setattr(ad, "bursting_hashtags", lambda *a, **k: [])
    monkeypatch.setattr(ad, "cluster_accounts", lambda *a, **k: [])
    monkeypatch.setattr(ad, "V2_PROMOTION_STATE_PATH", tmp_path / "v2.json")
    ids = [str(i) for i in range(30)]
    con = _con(
        _scores([(i, float(i), 0.9, NOW) for i in ids]),
        _authors([(i, f"h{i}") for i in ids]),
    )
    monkeypatch.setattr(
        ad, "v2_accounts",
        lambda c, s, a, **k: v2_accounts(c, s, a, state_path=tmp_path / "v2.json"),
    )

    entries = promote(
        con, "posts_unused", "clusters_unused", "authors_tbl",
        state_path=tmp_path / "dynamic.json",
        scores_view="scores_tbl", v2_enabled=True, v2_max_new=5,
    )

    accounts = [e for e in entries if e.kind == "account"]
    assert len(accounts) == 5
    assert {e.source for e in accounts} == {"v2-centrality"}
    assert {e.value for e in accounts} == {"h29", "h28", "h27", "h26", "h25"}


def test_promote_ignores_v2_when_disabled(tmp_path, monkeypatch):
    import kenya_monitor.adaptive as ad

    monkeypatch.setattr(ad, "bursting_hashtags", lambda *a, **k: [])
    monkeypatch.setattr(ad, "cluster_accounts", lambda *a, **k: [])

    def boom(*a, **k):
        raise AssertionError("v2 path must not run when disabled")

    monkeypatch.setattr(ad, "v2_accounts", boom)

    entries = promote(
        duckdb.connect(), "p", "c", "a",
        state_path=tmp_path / "dynamic.json", scores_view="s", v2_enabled=False,
    )
    assert entries == []

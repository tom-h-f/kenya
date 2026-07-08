from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pyarrow as pa

from kenya_monitor.follow_crawl import (
    CrawlEntry,
    discover_from_edges,
    is_due,
    load_crawl_state,
    save_crawl_state,
)

NOW = datetime.now(timezone.utc)


def test_crawl_state_roundtrip(tmp_path):
    path = tmp_path / "follow_crawl.json"
    entries = {"u1": CrawlEntry(handle="alice", crawled_at=NOW.isoformat(), edge_count=10)}
    save_crawl_state(entries, path)
    loaded = load_crawl_state(path)
    assert loaded["u1"].handle == "alice"
    assert loaded["u1"].edge_count == 10


def test_is_due_respects_refresh_window():
    fresh = CrawlEntry(handle="a", crawled_at=NOW.isoformat())
    stale = CrawlEntry(
        handle="b",
        crawled_at=(NOW - timedelta(days=40)).isoformat(),
    )
    assert not is_due(fresh, refresh_days=30)
    assert is_due(stale, refresh_days=30)
    assert is_due(None, refresh_days=30)


def test_discover_from_edges_finds_uncrawled():
    authors = pa.table(
        {
            "platform": pa.array(["x", "x"], type=pa.string()),
            "platform_user_id": pa.array(["1", "2"], type=pa.string()),
            "handle": pa.array(["alice", "bob"], type=pa.string()),
            "collected_at": pa.array([NOW, NOW], type=pa.timestamp("us", tz="UTC")),
        }
    )
    follows = pa.table(
        {
            "platform": pa.array(["x"], type=pa.string()),
            "follower_id": pa.array(["1"], type=pa.string()),
            "followed_id": pa.array(["2"], type=pa.string()),
            "collected_at": pa.array([NOW], type=pa.timestamp("us", tz="UTC")),
        }
    )
    con = duckdb.connect()
    con.register("authors_tbl", authors)
    con.register("follows_tbl", follows)
    entries = {
        "1": CrawlEntry(handle="alice", crawled_at=NOW.isoformat()),
    }
    found = discover_from_edges(con, "follows_tbl", "authors_tbl", entries, refresh_days=30)
    assert found == [("2", "bob")]

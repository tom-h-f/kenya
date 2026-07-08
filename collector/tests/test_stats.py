from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pyarrow as pa

from kenya_monitor.stats import format_stats, gather_stats

NOW = datetime.now(timezone.utc)


def _storage_with_posts(rows: list[dict]):
    schema = pa.schema(
        [
            ("platform_post_id", pa.string()),
            ("author_id", pa.string()),
            ("collected_at", pa.timestamp("us", tz="UTC")),
            ("type", pa.string()),
            ("run", pa.string()),
            ("dt", pa.string()),
        ]
    )
    defaults = {
        "platform_post_id": "p1",
        "author_id": "a1",
        "collected_at": NOW,
        "type": "search",
        "run": "run1",
        "dt": NOW.date().isoformat(),
    }
    table = pa.Table.from_pylist([{**defaults, **r} for r in rows], schema=schema)
    con = duckdb.connect()
    con.register("posts_tbl", table)
    con.register("authors_tbl", pa.table({"platform_user_id": pa.array([], type=pa.string()), "collected_at": pa.array([], type=pa.timestamp("us", tz="UTC"))}))
    con.register("metrics_tbl", pa.table({"collected_at": pa.array([], type=pa.timestamp("us", tz="UTC"))}))
    con.register("engagements_tbl", pa.table({"collected_at": pa.array([], type=pa.timestamp("us", tz="UTC"))}))
    con.register("follows_tbl", pa.table({"collected_at": pa.array([], type=pa.timestamp("us", tz="UTC"))}))

    class FakeStorage:
        def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
            self.con = connection

        def posts_view(self, platform: str = "*") -> str:
            return "posts_tbl"

        def authors_view(self, platform: str = "*") -> str:
            return "authors_tbl"

        def metrics_view(self, platform: str = "*") -> str:
            return "metrics_tbl"

        def engagements_view(self, platform: str = "*") -> str:
            return "engagements_tbl"

        def follows_view(self, platform: str = "*") -> str:
            return "follows_tbl"

    return FakeStorage(con)


def test_gather_stats_counts_and_recent():
    yesterday = NOW - timedelta(days=1)
    storage = _storage_with_posts(
        [
            {"platform_post_id": "p1", "type": "search", "run": "r1"},
            {"platform_post_id": "p1", "type": "search", "run": "r2"},
            {"platform_post_id": "p2", "type": "timeline", "run": "r2"},
            {
                "platform_post_id": "old",
                "type": "search",
                "run": "old",
                "collected_at": yesterday,
                "dt": yesterday.date().isoformat(),
            },
        ]
    )
    stats = gather_stats(storage, recent_hours=6, daily_days=2)
    assert stats.posts_raw_total == 4
    assert stats.posts_unique_total == 3
    assert stats.recent_posts_raw == 3
    assert stats.recent_runs == 2
    assert any(t.name == "search" and t.raw_rows == 3 for t in stats.posts)


def test_format_stats_renders_sections():
    storage = _storage_with_posts([{"platform_post_id": "p1"}])
    text = format_stats(gather_stats(storage))
    assert "TOTAL" in text
    assert "RECENT" in text
    assert "posts (raw rows)" in text

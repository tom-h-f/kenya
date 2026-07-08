from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import duckdb
import pyarrow as pa

from kenya_monitor.collectors.base import Author, Engagement, FollowEdge, MetricSnapshot, Post
from kenya_monitor.config import R2Config

POST_SCHEMA = pa.schema(
    [
        ("platform", pa.string()),
        ("platform_post_id", pa.string()),
        ("author_id", pa.string()),
        ("author_handle", pa.string()),
        ("text", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("url", pa.string()),
        ("source_query", pa.string()),
        ("lang", pa.string()),
        ("in_reply_to_id", pa.string()),
        ("is_repost", pa.bool_()),
        ("repost_of_id", pa.string()),
        ("like_count", pa.int64()),
        ("reply_count", pa.int64()),
        ("repost_count", pa.int64()),
        ("quote_count", pa.int64()),
        ("view_count", pa.int64()),
        ("hashtags", pa.list_(pa.string())),
        ("cashtags", pa.list_(pa.string())),
        ("mentions", pa.list_(pa.string())),
        ("urls", pa.list_(pa.string())),
        ("quoted_post_id", pa.string()),
        ("is_quote", pa.bool_()),
        ("conversation_id", pa.string()),
        ("in_reply_to_user_id", pa.string()),
        ("source_label", pa.string()),
        ("place_name", pa.string()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("has_media", pa.bool_()),
        ("media_count", pa.int64()),
        ("media_urls", pa.list_(pa.string())),
        ("collected_at", pa.timestamp("us", tz="UTC")),
    ]
)

METRIC_SCHEMA = pa.schema(
    [
        ("platform", pa.string()),
        ("platform_post_id", pa.string()),
        ("like_count", pa.int64()),
        ("reply_count", pa.int64()),
        ("repost_count", pa.int64()),
        ("quote_count", pa.int64()),
        ("view_count", pa.int64()),
        ("collected_at", pa.timestamp("us", tz="UTC")),
    ]
)

AUTHOR_SCHEMA = pa.schema(
    [
        ("platform", pa.string()),
        ("platform_user_id", pa.string()),
        ("handle", pa.string()),
        ("display_name", pa.string()),
        ("bio", pa.string()),
        ("location", pa.string()),
        ("followers_count", pa.int64()),
        ("following_count", pa.int64()),
        ("tweet_count", pa.int64()),
        ("listed_count", pa.int64()),
        ("verified", pa.bool_()),
        ("blue", pa.bool_()),
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("profile_image_url", pa.string()),
        ("collected_at", pa.timestamp("us", tz="UTC")),
    ]
)


ENGAGEMENT_SCHEMA = pa.schema(
    [
        ("platform", pa.string()),
        ("platform_post_id", pa.string()),
        ("platform_user_id", pa.string()),
        ("kind", pa.string()),
        ("collected_at", pa.timestamp("us", tz="UTC")),
    ]
)

FOLLOW_SCHEMA = pa.schema(
    [
        ("platform", pa.string()),
        ("follower_id", pa.string()),
        ("followed_id", pa.string()),
        ("collected_at", pa.timestamp("us", tz="UTC")),
    ]
)


def run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def _dt_partition(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d")


class Storage:
    """Writes immutable Parquet run-files to R2 and reads them back via DuckDB."""

    def __init__(self, cfg: R2Config, con: duckdb.DuckDBPyConnection | None = None):
        self.cfg = cfg
        self.con = con or duckdb.connect()
        self._init_r2()

    def _init_r2(self) -> None:
        self.con.execute("INSTALL httpfs; LOAD httpfs;")
        self.con.execute(
            """
            CREATE OR REPLACE SECRET r2 (
                TYPE r2,
                KEY_ID ?,
                SECRET ?,
                ACCOUNT_ID ?
            );
            """,
            [self.cfg.access_key_id, self.cfg.secret_access_key, self.cfg.account_id],
        )

    def _uri(self, key: str) -> str:
        return f"r2://{self.cfg.bucket}/{key}"

    def _copy_table(self, table: pa.Table, key: str) -> None:
        self.con.register("_write_buf", table)
        try:
            self.con.execute(
                f"COPY _write_buf TO '{self._uri(key)}' (FORMAT parquet, COMPRESSION zstd);"
            )
        finally:
            self.con.unregister("_write_buf")

    def write_posts(self, posts: Sequence[Post], target_type: str, now: datetime | None = None) -> str | None:
        if not posts:
            return None
        platform = posts[0].platform
        table = pa.Table.from_pylist([p.as_row() for p in posts], schema=POST_SCHEMA)
        key = (
            f"posts/platform={platform}/type={target_type}"
            f"/dt={_dt_partition(now)}/run={run_id(now)}.parquet"
        )
        self._copy_table(table, key)
        return key

    def write_metrics(self, metrics: Sequence[MetricSnapshot], now: datetime | None = None) -> str | None:
        if not metrics:
            return None
        platform = metrics[0].platform
        table = pa.Table.from_pylist([m.as_row() for m in metrics], schema=METRIC_SCHEMA)
        key = f"metrics/platform={platform}/dt={_dt_partition(now)}/run={run_id(now)}.parquet"
        self._copy_table(table, key)
        return key

    def write_authors(self, authors: Sequence[Author], now: datetime | None = None) -> str | None:
        if not authors:
            return None
        platform = authors[0].platform
        table = pa.Table.from_pylist([a.as_row() for a in authors], schema=AUTHOR_SCHEMA)
        key = f"authors/platform={platform}/dt={_dt_partition(now)}/run={run_id(now)}.parquet"
        self._copy_table(table, key)
        return key

    def write_engagements(
        self, engagements: Sequence[Engagement], now: datetime | None = None
    ) -> str | None:
        if not engagements:
            return None
        platform = engagements[0].platform
        table = pa.Table.from_pylist([e.as_row() for e in engagements], schema=ENGAGEMENT_SCHEMA)
        key = f"engagements/platform={platform}/dt={_dt_partition(now)}/run={run_id(now)}.parquet"
        self._copy_table(table, key)
        return key

    def write_follows(self, edges: Sequence[FollowEdge], now: datetime | None = None) -> str | None:
        if not edges:
            return None
        platform = edges[0].platform
        table = pa.Table.from_pylist([e.as_row() for e in edges], schema=FOLLOW_SCHEMA)
        key = f"follows/platform={platform}/dt={_dt_partition(now)}/run={run_id(now)}.parquet"
        self._copy_table(table, key)
        return key

    def healthcheck(self) -> int:
        """Round-trip a probe under a fixed key outside posts/ (overwritten each call)."""
        table = pa.table({"ok": [1], "at": [datetime.now(timezone.utc)]})
        key = "_healthcheck/probe.parquet"
        self._copy_table(table, key)
        return self.con.sql(f"SELECT count(*) FROM read_parquet('{self._uri(key)}')").fetchone()[0]

    def query(self, sql: str) -> duckdb.DuckDBPyRelation:
        return self.con.sql(sql)

    def posts_view(self, platform: str = "*", target_type: str = "*") -> str:
        """A read_parquet glob expression for use in SQL (latest-state dedup left to caller)."""
        glob = self._uri(f"posts/platform={platform}/type={target_type}/dt=*/run=*.parquet")
        return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"

    def authors_view(self, platform: str = "*") -> str:
        glob = self._uri(f"authors/platform={platform}/dt=*/run=*.parquet")
        return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"

    def engagements_view(self, platform: str = "*") -> str:
        glob = self._uri(f"engagements/platform={platform}/dt=*/run=*.parquet")
        return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"

    def metrics_view(self, platform: str = "*") -> str:
        glob = self._uri(f"metrics/platform={platform}/dt=*/run=*.parquet")
        return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"

    def follows_view(self, platform: str = "*") -> str:
        glob = self._uri(f"follows/platform={platform}/dt=*/run=*.parquet")
        return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"

    def clusters_view(self, platform: str = "*") -> str:
        """Persisted coordination clusters (written by the analysis side)."""
        glob = self._uri(f"coordination/platform={platform}/kind=clusters/dt=*/run=*.parquet")
        return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"

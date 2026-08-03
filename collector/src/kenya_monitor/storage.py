from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime, timezone

import logging

import duckdb
import pyarrow as pa

from kenya_monitor.collectors.base import Author, Engagement, FollowEdge, MetricSnapshot, Post
from kenya_monitor.config import (
    COLLECTOR_MEMORY_LIMIT,
    COLLECTOR_THREADS,
    R2Config,
)

log = logging.getLogger("kenya_monitor")

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

# Audit trail for targeted collection: the only place the RENDERED query string
# is recorded. `posts.source_query` keeps the bare term, so without this there
# is no way to reconstruct what was actually asked of the platform, or when.
# For a project that searches for ethnic hate speech this is the difference
# between "we monitored hate speech" and "here is exactly what we searched for".
COLLECTION_RUN_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("target_type", pa.string()),
        ("term", pa.string()),
        ("source", pa.string()),
        ("fp_risk", pa.string()),
        ("query", pa.string()),
        ("since", pa.string()),
        ("until", pa.string()),
        ("n_posts", pa.int64()),
        ("collected_at", pa.timestamp("us", tz="UTC")),
    ]
)


# One row per census pass. Object-level degree is already recoverable from the
# engagements/ prefix by grouping on platform_post_id; what is NOT recoverable
# after the fact is the denominator - how many in-band candidates existed, how
# many were selected, and how many were skipped because their TTL had not
# lapsed. Without those, "coverage has saturated" is unfalsifiable, and that is
# open question Q3 in docs/analysis/census-tuning.md.
#
# Separate prefix from collection_runs/ for the same reason incitement/ is
# separate from labels/: different grain and schema under one glob makes
# union_by_name reads ambiguous.
CENSUS_RUN_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("platform", pa.string()),
        ("pass_kind", pa.string()),  # baseline | toxic | merged
        ("code_version", pa.string()),
        # parameters in force for this pass
        ("lookback_days", pa.int64()),
        ("band_min", pa.int64()),
        ("band_max", pa.int64()),
        ("refresh_hours", pa.int64()),
        ("retweeters_limit", pa.int64()),
        ("top_retweeted", pa.int64()),
        # supply: the whole in-band population, and the part not yet worked
        ("candidates_in_band", pa.int64()),
        ("candidates_uncensused", pa.int64()),
        # selection -> work actually done
        ("selected_retweeted", pa.int64()),
        ("selected_deg_min", pa.int64()),
        ("selected_deg_max", pa.int64()),
        ("due_retweeted", pa.int64()),
        ("fetched_retweeted", pa.int64()),
        ("skipped_ttl_retweeted", pa.int64()),
        # what came back. NOTE: degree_* and n_over_band_max measure the FETCH,
        # which SNOWBALL_RETWEETERS_LIMIT truncates - a 5,000-retweet object
        # registers as 300. Sound as a regression guard, not as a distribution.
        ("engagement_rows", pa.int64()),
        ("degree_p50", pa.int64()),
        ("degree_p90", pa.int64()),
        ("degree_max", pa.int64()),
        ("n_over_band_max", pa.int64()),
        # the other arms of the same pass
        ("selected_conversations", pa.int64()),
        ("selected_missing", pa.int64()),
        ("due_conversations", pa.int64()),
        ("reply_rows", pa.int64()),
        ("hydrated_rows", pa.int64()),
        ("authors_written", pa.int64()),
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
        self._tune()
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

    def _tune(self) -> None:
        """Bound DuckDB so a big scan spills instead of getting OOM-killed.

        DuckDB sizes its budget from the HOST, not the container cgroup, so on
        pi0 it plans against ~4GB while `mem_limit: 1g` kills it well before
        that. The census candidate query already reached 467MB of the 1GB
        limit; without a bound, a growing corpus takes the collector down
        rather than running slower.

        Best effort: setting names drift between DuckDB versions, and a tuning
        failure must not stop collection."""
        for stmt in (
            f"SET memory_limit='{COLLECTOR_MEMORY_LIMIT}'",
            f"SET threads={int(COLLECTOR_THREADS)}",
            "SET preserve_insertion_order=false",
        ):
            try:
                self.con.execute(stmt)
            except Exception:
                log.warning("storage: could not apply %r", stmt)

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

    def write_collection_run(
        self, rows: Sequence[dict], platform: str = "x", now: datetime | None = None
    ) -> str | None:
        """Append the query manifest for one targeted-collection pass."""
        if not rows:
            return None
        now = now or datetime.now(timezone.utc)
        rid = run_id(now)
        buf = [
            {
                "run_id": rid,
                "target_type": r.get("target_type"),
                "term": r.get("term"),
                "source": r.get("source"),
                "fp_risk": r.get("fp_risk"),
                "query": r.get("query"),
                "since": r.get("since"),
                "until": r.get("until"),
                "n_posts": int(r.get("n_posts", 0)),
                "collected_at": now,
            }
            for r in rows
        ]
        table = pa.Table.from_pylist(buf, schema=COLLECTION_RUN_SCHEMA)
        key = f"collection_runs/platform={platform}/dt={_dt_partition(now)}/run={rid}.parquet"
        self._copy_table(table, key)
        return key

    def write_census_run(
        self, stats: dict, platform: str = "x", now: datetime | None = None
    ) -> str | None:
        """One census pass's counters -> census_runs/ prefix.

        Unlike the other writers this does NOT return early on an empty pass. A
        pass that selected nothing, or fetched nothing because every candidate
        was inside its TTL, is a real observation - it is the saturation signal
        Q3 is watching for, and skipping it would make saturation look like an
        outage."""
        if not stats:
            return None
        now = now or datetime.now(timezone.utc)
        rid = run_id(now)
        row = {f.name: None for f in CENSUS_RUN_SCHEMA}
        row.update({k: v for k, v in stats.items() if k in row})
        row.update(
            {
                "run_id": rid,
                "platform": platform,
                "code_version": os.getenv("GIT_SHA", ""),
                "collected_at": now,
            }
        )
        table = pa.Table.from_pylist([row], schema=CENSUS_RUN_SCHEMA)
        key = f"census_runs/platform={platform}/dt={_dt_partition(now)}/run={rid}.parquet"
        self._copy_table(table, key)
        return key

    def census_runs_view(self, platform: str = "*") -> str:
        """Per-pass census counters (supply, selection, degree distribution)."""
        glob = self._uri(f"census_runs/platform={platform}/dt=*/run=*.parquet")
        return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"

    def collection_runs_view(self, platform: str = "*") -> str:
        """Rendered-query audit trail for targeted collection passes."""
        glob = self._uri(f"collection_runs/platform={platform}/dt=*/run=*.parquet")
        return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"

    def hatespeech_view(self, platform: str = "*") -> str:
        """Analysis-written hate scores (kma.hatespeech). Read-only here: the
        collector runs no models, it selects targets from scored parquet."""
        glob = self._uri(f"hatespeech/platform={platform}/dt=*/run=*.parquet")
        return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"

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

    def coordination_edges_view(
        self, platform: str = "*", channel: str = "*", method: str = "svn_fdr"
    ) -> str:
        """Persisted, statistically validated coordination edges (analysis-side).

        These are hub-capped, repetition-filtered and FDR-corrected by
        `kma.coordination`; preferring them over a local projection is what
        keeps the collector from re-deriving a weaker version of the same
        signal. Defaults to the `svn_fdr` method - the one the analysis side
        treats as its working edge set."""
        glob = self._uri(
            f"coordination/platform={platform}/kind=edges"
            f"/channel={channel}/method={method}/dt=*/run=*.parquet"
        )
        return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"

    def clusters_view(self, platform: str = "*") -> str:
        """Persisted coordination clusters (written by the analysis side)."""
        glob = self._uri(f"coordination/platform={platform}/kind=clusters/dt=*/run=*.parquet")
        return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"

    def stories_view(self, platform: str = "*") -> str:
        """Persisted flagged stories (written by the analysis side, Phase 4)."""
        glob = self._uri(f"stories/platform={platform}/dt=*/run=*.parquet")
        return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"

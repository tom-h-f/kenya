"""Tests for the Phase-4 story-flag promotion path (no R2 required)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pyarrow as pa

from kenya_monitor.adaptive import flagged_story_keywords, promote

NOW = datetime.now(timezone.utc)

_STORY_SCHEMA = pa.schema(
    [
        ("story_id", pa.int64()),
        ("keywords", pa.list_(pa.string())),
        ("hashtags", pa.list_(pa.string())),
        ("story_suspicion_index", pa.float64()),
        ("computed_at", pa.timestamp("us", tz="UTC")),
    ]
)


def _con_with_stories(rows: list[dict]) -> tuple[duckdb.DuckDBPyConnection, str]:
    con = duckdb.connect()
    con.register("stories_tbl", pa.Table.from_pylist(rows, schema=_STORY_SCHEMA))
    return con, "stories_tbl"


def test_flagged_story_keywords_filters_and_unnests():
    con, view = _con_with_stories(
        [
            {"story_id": 0, "keywords": ["iebc", "rigging"], "hashtags": ["#rigged"],
             "story_suspicion_index": 0.8, "computed_at": NOW},
            {"story_id": 1, "keywords": ["weather"], "hashtags": ["#sunny"],
             "story_suspicion_index": 0.3, "computed_at": NOW},
        ]
    )
    terms = set(flagged_story_keywords(con, view, min_index=0.6))
    assert terms == {"iebc", "rigging", "#rigged"}  # below-cutoff story dropped


def test_flagged_story_keywords_uses_latest_run_only():
    old = NOW - timedelta(days=1)
    con, view = _con_with_stories(
        [
            {"story_id": 0, "keywords": ["stale"], "hashtags": [],
             "story_suspicion_index": 0.9, "computed_at": old},
            {"story_id": 0, "keywords": ["fresh"], "hashtags": [],
             "story_suspicion_index": 0.9, "computed_at": NOW},
        ]
    )
    assert flagged_story_keywords(con, view, min_index=0.6) == ["fresh"]


def test_flagged_story_keywords_missing_view_is_empty():
    con = duckdb.connect()
    assert flagged_story_keywords(con, "no_such_view") == []


_POST_SCHEMA = pa.schema(
    [
        ("platform", pa.string()),
        ("platform_post_id", pa.string()),
        ("collected_at", pa.timestamp("us", tz="UTC")),
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("hashtags", pa.list_(pa.string())),
    ]
)
_CLUSTER_SCHEMA = pa.schema(
    [("author_id", pa.string()), ("computed_at", pa.timestamp("us", tz="UTC"))]
)
_AUTHOR_SCHEMA = pa.schema(
    [
        ("platform", pa.string()),
        ("platform_user_id", pa.string()),
        ("handle", pa.string()),
        ("collected_at", pa.timestamp("us", tz="UTC")),
    ]
)


def test_promote_tags_story_keywords(tmp_path):
    con = duckdb.connect()
    con.register(
        "posts_tbl",
        pa.Table.from_pylist(
            [{"platform": "x", "platform_post_id": "p0", "collected_at": NOW,
              "created_at": NOW, "hashtags": []}],
            schema=_POST_SCHEMA,
        ),
    )
    con.register("clusters_tbl", pa.Table.from_pylist([], schema=_CLUSTER_SCHEMA))
    con.register("authors_tbl", pa.Table.from_pylist([], schema=_AUTHOR_SCHEMA))
    con.register(
        "stories_tbl",
        pa.Table.from_pylist(
            [{"story_id": 0, "keywords": ["iebc"], "hashtags": ["#rigged"],
              "story_suspicion_index": 0.9, "computed_at": NOW}],
            schema=_STORY_SCHEMA,
        ),
    )
    entries = promote(
        con, "posts_tbl", "clusters_tbl", "authors_tbl", stories_view="stories_tbl",
        state_path=tmp_path / "dynamic.json", dry_run=True,
    )
    story_entries = {e.value: e for e in entries if e.source == "story-flag"}
    assert set(story_entries) == {"iebc", "#rigged"}
    assert all(e.kind == "keyword" for e in story_entries.values())

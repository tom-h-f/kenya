from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pyarrow as pa

from kenya_monitor.adaptive import (
    DynamicEntry,
    bursting_hashtags,
    detect_burst,
    load_state,
    merge_targets,
    refresh_entries,
    save_state,
)
from kenya_monitor.collectors.x import build_query
from kenya_monitor.config import PlatformTargets
from kenya_monitor.runner import _due, hot_objects

NOW = datetime.now(timezone.utc)


def test_build_query_full():
    q = build_query("IEBC", min_faves=5, since="2026-07-01", until="2026-07-02", include_retweets=True)
    assert q == "IEBC min_faves:5 since:2026-07-01 until:2026-07-02 include:nativeretweets"


def test_build_query_plain():
    assert build_query("Wantam") == "Wantam"


_TEST_SCHEMA = pa.schema(
    [
        ("platform", pa.string()),
        ("platform_post_id", pa.string()),
        ("author_id", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("collected_at", pa.timestamp("us", tz="UTC")),
        ("repost_of_id", pa.string()),
        ("quoted_post_id", pa.string()),
        ("in_reply_to_id", pa.string()),
        ("conversation_id", pa.string()),
        ("repost_count", pa.int64()),
        ("reply_count", pa.int64()),
        ("quote_count", pa.int64()),
        ("hashtags", pa.list_(pa.string())),
    ]
)


def _con_with_posts(rows: list[dict]) -> tuple[duckdb.DuckDBPyConnection, str]:
    defaults = {
        "platform": "x",
        "platform_post_id": "p0",
        "author_id": "a0",
        "created_at": NOW,
        "collected_at": NOW,
        "repost_of_id": None,
        "quoted_post_id": None,
        "in_reply_to_id": None,
        "conversation_id": None,
        "repost_count": 0,
        "reply_count": 0,
        "quote_count": 0,
        "hashtags": [],
    }
    table = pa.Table.from_pylist([{**defaults, **r} for r in rows], schema=_TEST_SCHEMA)
    con = duckdb.connect()
    con.register("posts_tbl", table)
    return con, "posts_tbl"


def test_hot_objects_selection_and_missing_refs():
    rows = [
        # two accounts retweeting the same hot object (never itself collected)
        {"platform_post_id": "1", "author_id": "a", "repost_of_id": "X", "repost_count": 500},
        {"platform_post_id": "2", "author_id": "b", "repost_of_id": "X", "repost_count": 500},
        # a quieter retweeted object whose original IS collected
        {"platform_post_id": "3", "author_id": "c", "repost_of_id": "4", "repost_count": 10},
        {"platform_post_id": "4", "author_id": "d", "repost_count": 10},
        # busy conversation
        {"platform_post_id": "5", "author_id": "e", "conversation_id": "5", "reply_count": 99},
    ]
    con, view = _con_with_posts(rows)
    retweeted, conversations, missing = hot_objects(
        con, view, lookback_days=2, top_retweeted=1, top_conversations=1, hydrate_limit=10
    )
    assert retweeted == ["X"]
    assert conversations == ["5"]
    assert "X" in missing and "4" not in missing


def test_hot_objects_respects_lookback():
    old = NOW - timedelta(days=10)
    rows = [
        {"platform_post_id": "1", "author_id": "a", "repost_of_id": "X",
         "repost_count": 500, "created_at": old},
    ]
    con, view = _con_with_posts(rows)
    retweeted, _, _ = hot_objects(con, view, lookback_days=2, top_retweeted=5)
    assert retweeted == []


def test_due_ttl():
    fresh = (NOW - timedelta(hours=1)).isoformat()
    stale = (NOW - timedelta(hours=24)).isoformat()
    state = {"a": fresh, "b": stale}
    assert _due(["a", "b", "c"], state, refresh_hours=12) == ["b", "c"]


def test_bursting_hashtags_new_and_ratio():
    burst_rows = [
        {"platform_post_id": f"n{i}", "hashtags": ["newtag"], "created_at": NOW - timedelta(hours=1)}
        for i in range(25)
    ]
    steady_rows = [
        {"platform_post_id": f"s{i}", "hashtags": ["steady"],
         "created_at": NOW - timedelta(days=(i % 8), hours=2)}
        for i in range(80)
    ]
    con, view = _con_with_posts(burst_rows + steady_rows)
    tags = bursting_hashtags(con, view, min_count=20, ratio=5.0)
    assert "#newtag" in tags
    assert "#steady" not in tags


def test_refresh_entries_caps_expiry_and_confirmation():
    old = (NOW - timedelta(days=10)).isoformat()
    recent = (NOW - timedelta(days=1)).isoformat()
    existing = [
        DynamicEntry("#expired", "keyword", "hashtag-burst", old, old),
        DynamicEntry("#confirmed", "keyword", "hashtag-burst", recent, recent),
        DynamicEntry("olduser", "account", "coordination-cluster", recent, recent),
    ]
    out = refresh_entries(
        existing,
        keywords=["#confirmed", "#new1", "#new2"],
        accounts=["newuser"],
        max_keywords=2,
        max_accounts=5,
        expiry_days=7,
        now=NOW,
    )
    values = {(e.kind, e.value) for e in out}
    assert ("keyword", "#expired") not in values
    assert ("account", "olduser") in values and ("account", "newuser") in values
    assert sum(1 for e in out if e.kind == "keyword") == 2  # cap enforced
    confirmed = next(e for e in out if e.value == "#confirmed")
    assert confirmed.last_confirmed == NOW.isoformat()
    assert confirmed.added_at == recent  # added_at survives confirmation


def test_merge_targets_dedupes_case_insensitively():
    static = PlatformTargets(accounts=["WilliamsRuto"], keywords=["IEBC"])
    entries = [
        DynamicEntry("iebc", "keyword", "hashtag-burst", NOW.isoformat(), NOW.isoformat()),
        DynamicEntry("#newtag", "keyword", "hashtag-burst", NOW.isoformat(), NOW.isoformat()),
        DynamicEntry("williamsruto", "account", "coordination-cluster", NOW.isoformat(), NOW.isoformat()),
        DynamicEntry("suspect1", "account", "coordination-cluster", NOW.isoformat(), NOW.isoformat()),
    ]
    merged = merge_targets(static, entries)
    assert merged.keywords == ["IEBC", "#newtag"]
    assert merged.accounts == ["WilliamsRuto", "suspect1"]


def test_state_roundtrip(tmp_path):
    path = tmp_path / "dynamic.json"
    entries = [DynamicEntry("#t", "keyword", "hashtag-burst", NOW.isoformat(), NOW.isoformat())]
    save_state(entries, path)
    assert load_state(path) == entries


def test_detect_burst_fires_on_spike():
    rows = []
    pid = 0
    for h in range(2, 40):  # steady baseline: 5 posts/hour
        for _ in range(5):
            rows.append({"platform_post_id": str(pid := pid + 1),
                         "created_at": NOW - timedelta(hours=h, minutes=30)})
    for _ in range(150):  # spike in the last complete hour
        rows.append({"platform_post_id": str(pid := pid + 1),
                     "created_at": NOW - timedelta(hours=1, minutes=30)})
    con, view = _con_with_posts(rows)
    bursting, z, n = detect_burst(con, view, zscore=3.0, min_posts=100)
    assert bursting and z > 3.0 and n >= 150


def test_detect_burst_quiet_on_steady_volume():
    rows = []
    pid = 0
    for h in range(1, 40):
        for _ in range(5):
            rows.append({"platform_post_id": str(pid := pid + 1),
                         "created_at": NOW - timedelta(hours=h, minutes=30)})
    con, view = _con_with_posts(rows)
    bursting, _, _ = detect_burst(con, view, zscore=3.0, min_posts=100)
    assert not bursting

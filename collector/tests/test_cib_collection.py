from __future__ import annotations

import asyncio
import json

import pytest

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
from kenya_monitor.collectors.base import Engagement
from kenya_monitor.runner import _due, collect_snowball, hot_objects

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


def test_promoted_accounts_are_separated_from_baseline_targets():
    """Coordination-promoted accounts must not reach the baseline `timeline`
    partition: they were selected for looking coordinated, so collecting them
    as baseline biases every prevalence measured downstream. `cib_timeline` is
    already in kma.db.TARGETED_TYPES; nothing was writing it."""
    from kenya_monitor.adaptive import DynamicEntry, merge_targets
    from kenya_monitor.config import PlatformTargets

    entries = [
        DynamicEntry("#burst", "keyword", "hashtag-burst", "t", "t"),
        DynamicEntry("suspect1", "account", "coordination-cluster", "t", "t"),
    ]
    static = PlatformTargets(accounts=["WilliamsRuto"], keywords=["IEBC"])

    keyword_entries = [e for e in entries if e.kind == "keyword"]
    promoted_accounts = [e.value for e in entries if e.kind == "account"]
    merged = merge_targets(static, keyword_entries)

    assert "#burst" in merged.keywords          # keywords still widen baseline
    assert "suspect1" not in merged.accounts    # accounts do NOT
    assert merged.accounts == ["WilliamsRuto"]
    assert promoted_accounts == ["suspect1"]


# --- snowball write batching -------------------------------------------------


class _FakeStorage:
    """Records writes; view methods are unused because `objects` is supplied."""

    def __init__(self):
        self.engagement_writes: list[int] = []
        self.post_writes: list[tuple[str, int]] = []
        self.con = None

    def write_engagements(self, rows, now=None):
        if not rows:
            return None
        self.engagement_writes.append(len(rows))
        return f"engagements/run={len(self.engagement_writes)}.parquet"

    def write_posts(self, posts, target_type, now=None):
        if not posts:
            return None
        self.post_writes.append((target_type, len(posts)))
        return f"posts/{target_type}.parquet"

    def write_authors(self, authors, now=None):
        return None


class _FakeCollector:
    platform = "x"

    def __init__(self, fail_on: str | None = None, per_object: int = 3):
        self.fail_on = fail_on
        self.per_object = per_object
        self.fetched: list[str] = []

    async def retweeters(self, post_id, limit):
        self.fetched.append(post_id)
        if post_id == self.fail_on:
            raise RuntimeError("rate limit abort")
        for i in range(self.per_object):
            yield Engagement(
                platform="x", platform_post_id=post_id,
                platform_user_id=f"{post_id}_u{i}", kind="retweet",
            )

    async def replies(self, post_id, limit):
        return
        yield  # pragma: no cover

    async def hydrate(self, post_ids):
        return
        yield  # pragma: no cover

    def collected_authors(self):
        return []


def _run_snowball(tmp_path, collector, storage, oids, flush_every):
    return asyncio.run(
        collect_snowball(
            collector, storage,
            objects=(oids, [], []),
            state_path=tmp_path / "snowball.json",
            flush_every=flush_every,
        )
    )


def test_snowball_flushes_in_batches_not_once_at_the_end(tmp_path):
    """A single end-of-pass write meant a restart mid-census discarded every API
    call made since it began - ~40 minutes of rate-limited pool budget at 250
    objects per pass."""
    storage, collector = _FakeStorage(), _FakeCollector()
    oids = [f"o{i}" for i in range(10)]
    counts = _run_snowball(tmp_path, collector, storage, oids, flush_every=4)
    assert storage.engagement_writes == [12, 12, 6]   # 4 + 4 + 2 objects x 3 rows
    assert counts["retweeters"] == 30


def test_completed_batches_survive_a_mid_pass_failure(tmp_path):
    """The durability property: work already flushed stays written, and only the
    unflushed objects are retried."""
    storage = _FakeStorage()
    collector = _FakeCollector(fail_on="o5")
    oids = [f"o{i}" for i in range(10)]
    with pytest.raises(RuntimeError):
        _run_snowball(tmp_path, collector, storage, oids, flush_every=4)

    assert storage.engagement_writes == [12]          # first batch persisted
    state = json.loads((tmp_path / "snowball.json").read_text())
    assert set(state) == {"o0", "o1", "o2", "o3"}     # only flushed objects marked
    assert "o4" not in state, "unflushed work must not be marked done"


def test_objects_are_marked_only_after_their_write(tmp_path):
    """Ordering guard: marking before writing would lose data silently, because
    a marked object is skipped by the TTL on the next pass."""
    storage, collector = _FakeStorage(), _FakeCollector()
    _run_snowball(tmp_path, collector, storage, ["a", "b"], flush_every=1)
    state = json.loads((tmp_path / "snowball.json").read_text())
    assert set(state) == {"a", "b"}
    assert storage.engagement_writes == [3, 3]

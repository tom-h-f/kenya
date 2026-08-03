from __future__ import annotations

import asyncio
import json
import logging

import pytest
import time

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
import pathlib

import kenya_monitor.config
from kenya_monitor.config import PlatformTargets, SNOWBALL_BAND_MAX
from kenya_monitor.collectors.base import Engagement
from kenya_monitor import runner
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
    # X has 500 reposts - above the band, so the coordination hub cap would
    # discard every pair it produced. The quieter object is the useful one.
    assert retweeted == ["4"]
    assert conversations == ["5"]
    assert "X" in missing and "4" not in missing


def test_hot_objects_excludes_hubs_and_singletons():
    """Measured 2026-08-01: 420 of 422 censused objects were hubs (>100
    amplifiers) and contributed 0 usable pairs, because the projection discards
    them. A hub costs ~3 paginated requests for ~300 accounts that are all
    thrown away; a mid-band object costs 1 request for ~27 that all count."""
    rows = [
        {"platform_post_id": "h1", "author_id": "a", "repost_of_id": "HUB", "repost_count": 5000},
        {"platform_post_id": "m1", "author_id": "b", "repost_of_id": "MID", "repost_count": 42},
        {"platform_post_id": "s1", "author_id": "c", "repost_of_id": "LONE", "repost_count": 1},
    ]
    con, view = _con_with_posts(rows)
    retweeted, _, _ = hot_objects(con, view, lookback_days=2, top_retweeted=10)
    assert retweeted == ["MID"]
    assert "HUB" not in retweeted, "above the hub cap - pairs would be discarded"
    assert "LONE" not in retweeted, "below the floor - unpairable"


def test_census_band_is_wired_to_the_analysis_hub_cap_env_var():
    """The band max must track `kma.coordination.HUB_CAP_MAX`, or the census
    silently refills with objects the projection will discard.

    The collector cannot import `kma` - that separation is deliberate and load
    bearing - so the coupling is the shared env var name. Asserted against the
    source because reloading the module to test it rebuilds `HateTerm` and
    breaks `isinstance` for anything already constructed."""
    src = (
        pathlib.Path(kenya_monitor.config.__file__).read_text().splitlines()
    )
    line = next(ln for ln in src if ln.startswith("SNOWBALL_BAND_MAX"))
    assert 'os.getenv("HUB_CAP_MAX"' in line, line
    assert SNOWBALL_BAND_MAX == 100  # matches coordination.HUB_CAP_MAX default


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
        self.census_runs: list[dict] = []
        self.con = None

    def write_census_run(self, stats, platform="x", now=None):
        self.census_runs.append(dict(stats))
        return f"census_runs/run={len(self.census_runs)}.parquet"

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


def test_hate_cadence_survives_restarts():
    """`cycle` resets to 0 on every container restart, so a `cycle % N` schedule
    pushed the hate steps back ~3h on each redeploy. Observed on pi0: zero hate
    executions across a whole container lifetime. The schedule is now wall-clock."""
    from kenya_monitor.scheduler import _cycle_estimate_s

    # Before any cycle completes we cannot know the period; fall back, do not divide by zero.
    assert _cycle_estimate_s(0, time.monotonic()) == 1800.0
    # After cycles complete, estimate from elapsed time, floored so a fast cycle
    # cannot collapse the cadence to nothing.
    assert _cycle_estimate_s(4, time.monotonic() - 7200) == pytest.approx(1800, rel=0.1)
    assert _cycle_estimate_s(1000, time.monotonic() - 10) >= 60.0


def test_selection_excludes_recently_censused_objects():
    """Selection is deterministic (repost_count DESC), so without a TTL-aware
    filter the same objects are re-picked every pass and then discarded by
    `_due`. Observed on pi0: 495 selected, 10-35 actually fetched, while ~2,900
    uncensused in-band objects went untouched."""
    rows = [
        {"platform_post_id": "r1", "author_id": "a", "repost_of_id": "DONE", "repost_count": 99},
        {"platform_post_id": "r2", "author_id": "b", "repost_of_id": "FRESH", "repost_count": 50},
    ]
    con, view = _con_with_posts(rows)
    eng = pa.table({
        "platform": pa.array(["x"], type=pa.string()),
        "platform_post_id": pa.array(["DONE"], type=pa.string()),
        "platform_user_id": pa.array(["u1"], type=pa.string()),
        "kind": pa.array(["retweet"], type=pa.string()),
        "collected_at": pa.array([NOW], type=pa.timestamp("us", tz="UTC")),
    })
    con.register("eng_tbl", eng)

    # Without the filter, DONE outranks FRESH and consumes the budget.
    unfiltered, _, _ = hot_objects(con, view, lookback_days=2, top_retweeted=1)
    assert unfiltered == ["DONE"]

    # With it, the budget goes to work that has not been done.
    filtered, _, _ = hot_objects(
        con, view, lookback_days=2, top_retweeted=1,
        engagements_view="eng_tbl", refresh_hours=12,
    )
    assert filtered == ["FRESH"]


def test_selection_filter_tolerates_a_missing_engagements_view():
    rows = [{"platform_post_id": "r1", "author_id": "a", "repost_of_id": "X", "repost_count": 50}]
    con, view = _con_with_posts(rows)
    got, _, _ = hot_objects(con, view, lookback_days=2, engagements_view="_nope")
    assert got == ["X"]


def test_censused_objects_return_once_the_ttl_lapses():
    """The filter is a TTL, not a permanent exclusion - hot objects still get
    re-censused to pick up new amplifiers."""
    rows = [{"platform_post_id": "r1", "author_id": "a", "repost_of_id": "OLD", "repost_count": 50}]
    con, view = _con_with_posts(rows)
    eng = pa.table({
        "platform": pa.array(["x"], type=pa.string()),
        "platform_post_id": pa.array(["OLD"], type=pa.string()),
        "platform_user_id": pa.array(["u1"], type=pa.string()),
        "kind": pa.array(["retweet"], type=pa.string()),
        "collected_at": pa.array([NOW - timedelta(hours=20)], type=pa.timestamp("us", tz="UTC")),
    })
    con.register("eng_tbl", eng)
    assert hot_objects(con, view, lookback_days=2, engagements_view="eng_tbl", refresh_hours=12)[0] == ["OLD"]
    assert hot_objects(con, view, lookback_days=2, engagements_view="eng_tbl", refresh_hours=48)[0] == []


# --- census pass telemetry (census-tuning measurement layer) ----------------


def _eng_table(pairs):
    """(object_id, collected_at) -> a one-row-per-pair engagements table."""
    return pa.table({
        "platform": pa.array(["x"] * len(pairs), type=pa.string()),
        "platform_post_id": pa.array([o for o, _ in pairs], type=pa.string()),
        "platform_user_id": pa.array([f"u{i}" for i in range(len(pairs))], type=pa.string()),
        "kind": pa.array(["retweet"] * len(pairs), type=pa.string()),
        "collected_at": pa.array([t for _, t in pairs], type=pa.timestamp("us", tz="UTC")),
    })


def test_backlog_count_is_not_capped_by_the_selection_limit():
    """`candidates_uncensused` is the Q3 saturation series. Capping it at the
    per-pass budget would make a full backlog indistinguishable from a drained
    one, which is the exact thing being watched for."""
    rows = [
        {"platform_post_id": f"r{i}", "author_id": f"a{i}",
         "repost_of_id": f"O{i}", "repost_count": 10 + i}
        for i in range(5)
    ]
    con, view = _con_with_posts(rows)
    stats: dict = {}

    got, _, _ = hot_objects(con, view, lookback_days=2, top_retweeted=1, stats=stats)

    assert len(got) == 1                       # budget honoured
    assert stats["candidates_in_band"] == 5    # supply is not truncated by it
    assert stats["candidates_uncensused"] == 5
    assert stats["selected_retweeted"] == 1


def test_backlog_separates_the_whole_band_from_the_unworked_part():
    rows = [
        {"platform_post_id": "r1", "author_id": "a", "repost_of_id": "DONE", "repost_count": 99},
        {"platform_post_id": "r2", "author_id": "b", "repost_of_id": "F1", "repost_count": 50},
        {"platform_post_id": "r3", "author_id": "c", "repost_of_id": "F2", "repost_count": 40},
    ]
    con, view = _con_with_posts(rows)
    con.register("eng_tbl", _eng_table([("DONE", NOW)]))
    stats: dict = {}

    got, _, _ = hot_objects(
        con, view, lookback_days=2, engagements_view="eng_tbl",
        refresh_hours=12, stats=stats,
    )

    assert got == ["F1", "F2"]
    assert stats["candidates_in_band"] == 3
    assert stats["candidates_uncensused"] == 2


def test_backlog_is_reported_when_the_whole_band_is_already_censused():
    """The saturated state. An empty selection must not report an empty band -
    that would read as "no supply" when the truth is "all of it is done"."""
    rows = [
        {"platform_post_id": "r1", "author_id": "a", "repost_of_id": "D1", "repost_count": 99},
        {"platform_post_id": "r2", "author_id": "b", "repost_of_id": "D2", "repost_count": 50},
    ]
    con, view = _con_with_posts(rows)
    con.register("eng_tbl", _eng_table([("D1", NOW), ("D2", NOW)]))
    stats: dict = {}

    got, _, _ = hot_objects(
        con, view, lookback_days=2, engagements_view="eng_tbl",
        refresh_hours=12, stats=stats,
    )

    assert got == []
    assert stats["candidates_in_band"] == 2
    assert stats["candidates_uncensused"] == 0


def test_censused_filter_survives_a_null_object_id():
    """`NOT IN` is NULL-annihilating: under three-valued logic one NULL
    platform_post_id in engagements/ returns NULL for every candidate and the
    selector yields nothing - a total stall that looks like a drained backlog."""
    rows = [{"platform_post_id": "r1", "author_id": "a",
             "repost_of_id": "X", "repost_count": 50}]
    con, view = _con_with_posts(rows)
    eng = pa.table({
        "platform": pa.array(["x", "x"], type=pa.string()),
        "platform_post_id": pa.array([None, "OTHER"], type=pa.string()),
        "platform_user_id": pa.array(["u1", "u2"], type=pa.string()),
        "kind": pa.array(["retweet", "retweet"], type=pa.string()),
        "collected_at": pa.array([NOW, NOW], type=pa.timestamp("us", tz="UTC")),
    })
    con.register("eng_tbl", eng)

    got, _, _ = hot_objects(
        con, view, lookback_days=2, engagements_view="eng_tbl", refresh_hours=12
    )

    assert got == ["X"]


def test_census_run_records_ttl_skips_and_degree_shape(tmp_path):
    storage = _FakeStorage()
    collector = _FakeCollector(per_object=4)
    state = {"done1": NOW.isoformat(), "done2": NOW.isoformat()}
    (tmp_path / "snowball.json").write_text(json.dumps(state))

    asyncio.run(
        collect_snowball(
            collector, storage,
            objects=(["done1", "done2", "fresh1", "fresh2", "fresh3"], [], []),
            state_path=tmp_path / "snowball.json",
            refresh_hours=12,
        )
    )

    run = storage.census_runs[-1]
    assert run["selected_retweeted"] == 5
    assert run["due_retweeted"] == 3
    assert run["skipped_ttl_retweeted"] == 2
    assert run["fetched_retweeted"] == 3
    assert run["degree_p50"] == 4
    assert run["degree_max"] == 4
    assert run["engagement_rows"] == 12


def test_census_run_counts_objects_returned_above_the_band_max(tmp_path):
    """The Q2 measurement, now standing. Objects above the band come back to an
    analysis hub cap that discards them, so the requests buy nothing."""
    storage = _FakeStorage()

    class _Varying(_FakeCollector):
        degrees = {"small": 5, "mid": 40, "hub": 150}

        async def retweeters(self, post_id, limit):
            for i in range(self.degrees[post_id]):
                yield Engagement(
                    platform="x", platform_post_id=post_id,
                    platform_user_id=f"{post_id}_u{i}", kind="retweet",
                )

    asyncio.run(
        collect_snowball(
            _Varying(), storage,
            objects=(["small", "mid", "hub"], [], []),
            state_path=tmp_path / "snowball.json",
            band_max=100,
        )
    )

    run = storage.census_runs[-1]
    assert run["n_over_band_max"] == 1
    assert run["degree_p50"] == 40
    assert run["degree_max"] == 150
    assert run["band_max"] == 100


def test_census_over_band_share_warns(tmp_path, caplog):
    """Pre-banding 99.5% of censused objects were hubs; post-banding 6.4%. The
    guard has to trip on a return to the old behaviour."""
    storage = _FakeStorage()

    class _AllHubs(_FakeCollector):
        async def retweeters(self, post_id, limit):
            for i in range(200):
                yield Engagement(
                    platform="x", platform_post_id=post_id,
                    platform_user_id=f"{post_id}_u{i}", kind="retweet",
                )

    with caplog.at_level(logging.WARNING, logger="kenya_monitor"):
        asyncio.run(
            collect_snowball(
                _AllHubs(), storage,
                objects=(["a", "b"], [], []),
                state_path=tmp_path / "snowball.json",
                band_max=100,
            )
        )

    assert any("above the band max" in r.message for r in caplog.records)


def test_census_over_band_share_is_quiet_at_the_measured_healthy_rate(tmp_path):
    storage = _FakeStorage()
    collector = _FakeCollector(per_object=30)   # every object well inside the band

    asyncio.run(
        collect_snowball(
            collector, storage,
            objects=(["a", "b", "c"], [], []),
            state_path=tmp_path / "snowball.json",
            band_max=100,
        )
    )

    assert storage.census_runs[-1]["n_over_band_max"] == 0


def test_census_run_is_written_even_when_nothing_was_due(tmp_path):
    """A pass with no work is the saturation observation, not an outage."""
    storage = _FakeStorage()
    state = {"done": NOW.isoformat()}
    (tmp_path / "snowball.json").write_text(json.dumps(state))

    asyncio.run(
        collect_snowball(
            _FakeCollector(), storage,
            objects=(["done"], [], []),
            state_path=tmp_path / "snowball.json",
            refresh_hours=12,
        )
    )

    assert len(storage.census_runs) == 1
    assert storage.census_runs[0]["fetched_retweeted"] == 0
    assert storage.census_runs[0]["skipped_ttl_retweeted"] == 1


def test_hydration_prioritises_censused_objects_over_amplified_ones():
    """A censused object with no post row has no created_at, so the coordination
    window cannot place it and every retweeter trace it carries is unusable.
    Measured 2026-08-02: 2,805 of 4,679 censused objects were in that state,
    holding 30% of the engagement arm. Ordering by engagement spent these slots
    on the most-amplified refs - the hubs the census no longer collects."""
    rows = [
        # a heavily amplified ref we never collected, and never censused
        {"platform_post_id": "r1", "author_id": "a", "repost_of_id": "LOUD",
         "repost_count": 9000},
        # a quiet ref we DID census but never hydrated
        {"platform_post_id": "r2", "author_id": "b", "repost_of_id": "CENSUSED",
         "repost_count": 12},
    ]
    con, view = _con_with_posts(rows)
    con.register("eng_tbl", _eng_table([("CENSUSED", NOW - timedelta(days=3))]))

    _, _, missing = hot_objects(
        con, view, lookback_days=2, hydrate_limit=1, engagements_view="eng_tbl"
    )

    assert missing == ["CENSUSED"], "censused-but-unhydrated must win the budget"


def test_hydration_still_falls_back_to_engagement_order():
    """With nothing censused, behaviour is unchanged: most-amplified first."""
    rows = [
        {"platform_post_id": "r1", "author_id": "a", "repost_of_id": "LOUD",
         "repost_count": 9000},
        {"platform_post_id": "r2", "author_id": "b", "repost_of_id": "QUIET",
         "repost_count": 12},
    ]
    con, view = _con_with_posts(rows)

    _, _, missing = hot_objects(con, view, lookback_days=2, hydrate_limit=1)

    assert missing == ["LOUD"]


def test_hydration_skips_refs_whose_original_is_already_collected():
    rows = [
        {"platform_post_id": "r1", "author_id": "a", "repost_of_id": "HAVE",
         "repost_count": 50},
        {"platform_post_id": "HAVE", "author_id": "z", "repost_count": 50},
    ]
    con, view = _con_with_posts(rows)

    _, _, missing = hot_objects(con, view, lookback_days=2, hydrate_limit=10)

    assert "HAVE" not in missing


# --- census-discovered account timelines -------------------------------------


def _con_for_census_accounts(engagement_uids, author_rows, post_author_ids):
    con = duckdb.connect()
    con.execute("CREATE TABLE eng (platform VARCHAR, platform_post_id VARCHAR,"
                " platform_user_id VARCHAR, kind VARCHAR, collected_at TIMESTAMPTZ)")
    for uid in engagement_uids:
        con.execute("INSERT INTO eng VALUES ('x','o1',?,'retweet',now())", [uid])
    con.execute("CREATE TABLE auth (platform VARCHAR, platform_user_id VARCHAR,"
                " handle VARCHAR, collected_at TIMESTAMPTZ)")
    for uid, handle in author_rows:
        con.execute("INSERT INTO auth VALUES ('x',?,?,now())", [uid, handle])
    con.execute("CREATE TABLE po (author_id VARCHAR)")
    for aid in post_author_ids:
        con.execute("INSERT INTO po VALUES (?)", [aid])
    return con


def test_census_timeline_candidates_come_from_the_census():
    """Candidates are accounts seen retweeting. The posts anti-join was dropped
    on purpose: it doubled the query cost (464s vs 215s at 800MB/2 threads) to
    exclude ~1,000 of ~180,000 accounts, since 179,106 censused accounts have no
    posts at all. The attempted map is what actually prevents repeats."""
    con = _con_for_census_accounts(
        engagement_uids=["u1", "u2"],
        author_rows=[("u1", "alice"), ("u2", "bob")],
        post_author_ids=["u2"],
    )

    got = runner.census_discovered_handles(con, "eng", "auth", "po", limit=10)

    assert set(got) == {"alice", "bob"}


def test_census_timeline_selection_is_random_not_by_coordination():
    """Load bearing. Selecting by cluster membership would give the suspected
    accounts more posts, traces and edges, manufacturing the corroboration it is
    meant to measure - and the cib_timeline quarantine does not help, because
    coordination traces are built from all post types."""
    import inspect

    src = inspect.getsource(runner._refresh_census_pool)
    body = '"""'.join(src.split('"""')[2:])  # the SQL uses triple quotes too

    assert "ORDER BY random()" in body
    for forbidden in ("suspicion", "n_channels", "clusters_view", "cluster_id"):
        assert forbidden not in body, f"selection must not condition on {forbidden}"


def test_census_timeline_state_stops_refetching_accounts_with_no_tweets():
    """An account with no tweets never gains a post row, so it would otherwise
    be re-picked forever."""
    con = _con_for_census_accounts(
        engagement_uids=["u1"], author_rows=[("u1", "alice")], post_author_ids=[]
    )
    state: dict = {}

    first = runner.census_discovered_handles(con, "eng", "auth", "po", limit=5, state=state)
    second = runner.census_discovered_handles(con, "eng", "auth", "po", limit=5, state=state)

    assert first == ["alice"]
    assert second == [], "already attempted inside the retry window"
    assert "u1" in state["attempted"]


def test_census_timeline_pool_is_cached_between_cycles():
    """The candidate query scans the parquet globs over the network - 215s at
    800MB/2 threads - so it must not run once per cycle to pick 20 handles."""
    con = _con_for_census_accounts(
        engagement_uids=[f"u{i}" for i in range(10)],
        author_rows=[(f"u{i}", f"h{i}") for i in range(10)],
        post_author_ids=[],
    )
    state: dict = {}
    calls = {"n": 0}
    real = runner._refresh_census_pool

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    runner._refresh_census_pool = counting
    try:
        first = runner.census_discovered_handles(
            con, "eng", "auth", "po", limit=3, state=state, pool_size=10
        )
        second = runner.census_discovered_handles(
            con, "eng", "auth", "po", limit=3, state=state, pool_size=10
        )
    finally:
        runner._refresh_census_pool = real

    assert len(first) == 3 and len(second) == 3
    assert not set(first) & set(second), "must not hand back the same accounts"
    assert calls["n"] == 1, "pool refreshed once, spent across cycles"


def test_census_timeline_pool_refreshes_once_stale():
    con = _con_for_census_accounts(
        engagement_uids=["u1"], author_rows=[("u1", "alice")], post_author_ids=[]
    )
    state = {
        "pool": [["u9", "stale_handle"]],
        "pool_refreshed_at": (NOW - timedelta(hours=48)).isoformat(),
        "attempted": {},
    }

    got = runner.census_discovered_handles(
        con, "eng", "auth", "po", limit=5, state=state, pool_hours=12
    )

    assert got == ["alice"], "stale pool discarded and redrawn"


def test_census_timeline_skips_accounts_without_a_handle():
    con = _con_for_census_accounts(
        engagement_uids=["u1", "u2"],
        author_rows=[("u1", None), ("u2", "bob")],
        post_author_ids=[],
    )

    got = runner.census_discovered_handles(con, "eng", "auth", "po", limit=10)

    assert got == ["bob"]

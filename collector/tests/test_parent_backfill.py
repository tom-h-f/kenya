"""Offline tests for the missing-retweet-parent backfill.

No R2: `posts_view` is a registered in-memory table and storage is a stub, so
the same SQL that runs against the parquet globs is exercised locally.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import duckdb
import pyarrow as pa
import pytest

from kenya_monitor.collectors.base import Author, Post
from kenya_monitor.parent_backfill import (
    TARGET_TYPE,
    BackfillEntry,
    backfill_parents,
    backfill_summary,
    blocked_ids,
    candidate_parents,
    is_retryable,
    load_state,
    prune_state,
    save_state,
)

NOW = datetime.now(timezone.utc)

# `dt` mirrors the hive partition key every R2 read exposes; the amplifier stage
# prunes on it, so a fixture without it cannot catch a wrong pruning predicate.
_POSTS_SCHEMA = pa.schema(
    [
        ("platform", pa.string()),
        ("platform_post_id", pa.string()),
        ("author_id", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("collected_at", pa.timestamp("us", tz="UTC")),
        ("repost_of_id", pa.string()),
        ("repost_count", pa.int64()),
        ("dt", pa.date32()),
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
        "repost_count": 0,
    }
    rows = [{**defaults, **r} for r in rows]
    for row in rows:
        row.setdefault("dt", row["collected_at"].date())
    con = duckdb.connect()
    con.register("posts_tbl", pa.Table.from_pylist(rows, schema=_POSTS_SCHEMA))
    return con, "posts_tbl"


def _retweets(parent: str, authors: list[str], **kw) -> list[dict]:
    return [
        {
            "platform_post_id": f"rt-{parent}-{a}",
            "author_id": a,
            "repost_of_id": parent,
            **kw,
        }
        for a in authors
    ]


# --- candidate selection -----------------------------------------------------


def test_ranking_is_densest_first_and_unbanded():
    """Amplifier count is coverage per request: each amplifier is one retweet
    row that becomes a candidate `fast_retweet` trace row once the parent lands.
    The trace's entity is the retweeted AUTHOR, so even a single-amplifier
    parent contributes, and v2 has no hub cap to discard the dense ones."""
    rows = (
        _retweets("LONE", ["a1"])
        + _retweets("PAIR", ["b1", "b2", "b3"])
        + _retweets("DENSE", [f"c{i}" for i in range(40)])
        + _retweets("HUB", [f"d{i}" for i in range(140)])
    )
    con, view = _con_with_posts(rows)
    stats: dict = {}
    got = candidate_parents(con, view, limit=10, stats=stats)

    assert [oid for oid, _ in got] == ["HUB", "DENSE", "PAIR", "LONE"]
    assert dict(got)["DENSE"] == 40
    # The hub is the single best request available, not the worst. Banding here
    # would have skipped it - measured on live R2 there are exactly 10 such
    # objects in the 167,220-id backlog.
    assert got[0] == ("HUB", 140)
    assert stats["selected_rows"] == 184


def test_band_only_is_an_opt_in_that_drops_the_densest_objects():
    """Kept because it is built, never the default: the census bands because
    `validated_edges` discards hubs, and v2 has no hub cap, so applying it here
    forfeits the highest-coverage requests in the backlog."""
    rows = (
        _retweets("LONE", ["a1"])
        + _retweets("PAIR", ["b1", "b2", "b3"])
        + _retweets("HUB", [f"d{i}" for i in range(140)])
    )
    con, view = _con_with_posts(rows)

    assert [oid for oid, _ in candidate_parents(con, view, limit=10)][0] == "HUB"
    banded = candidate_parents(con, view, limit=10, band_only=True)
    assert [oid for oid, _ in banded] == ["PAIR"]
    assert "HUB" not in dict(banded), "the densest object is forfeited by --band-only"


def test_limit_bounds_the_pass_to_the_top_of_the_ranking():
    rows = (
        _retweets("LONE", ["a1"])
        + _retweets("PAIR", ["b1", "b2", "b3"])
        + _retweets("DENSE", [f"c{i}" for i in range(40)])
    )
    con, view = _con_with_posts(rows)
    assert [oid for oid, _ in candidate_parents(con, view, limit=1)] == ["DENSE"]
    assert [oid for oid, _ in candidate_parents(con, view, limit=2)] == ["DENSE", "PAIR"]


def test_held_parents_are_not_candidates():
    """The arm is self-draining: once the original has a post row it stops
    qualifying, which is what stops a finished backfill refetching itself."""
    rows = _retweets("HELD", ["a1", "a2", "a3"]) + _retweets("GONE", ["b1", "b2", "b3"])
    rows.append({"platform_post_id": "HELD", "author_id": "origin"})
    con, view = _con_with_posts(rows)

    stats: dict = {}
    got = candidate_parents(con, view, limit=10, stats=stats)

    assert [oid for oid, _ in got] == ["GONE"]
    assert stats["retweeted_objects"] == 2
    assert stats["held_parents"] == 1
    assert stats["missing_parents"] == 1
    assert stats["missing_in_band"] == 1


def test_amplifiers_count_distinct_accounts_not_collection_snapshots():
    """Re-collecting the same retweet must not inflate the object's rank. The
    count is DISTINCT (parent, retweeter), which is also why the query needs no
    latest-snapshot window: `repost_of_id` and `author_id` cannot change for a
    given post id."""
    rows = _retweets("X", ["a1", "a2", "a3"])
    resnapshot = [{**r, "collected_at": NOW - timedelta(hours=3)} for r in rows]
    con, view = _con_with_posts(rows + resnapshot)

    assert dict(candidate_parents(con, view, limit=10))["X"] == 3


def test_amplifiers_ignore_the_platform_repost_counter():
    """`repost_count` is what the census bands on, because it is choosing
    objects to fetch retweeters FOR. Here the quantity that decides whether
    hydrating an object can produce a pair is how many amplifiers we HOLD, and a
    platform counter of 5,000 on an object we saw twice buys nothing."""
    rows = _retweets("VIRAL_BUT_THIN", ["a1", "a2"], repost_count=5000) + _retweets(
        "QUIET_BUT_DENSE", [f"b{i}" for i in range(9)], repost_count=9
    )
    con, view = _con_with_posts(rows)
    got = candidate_parents(con, view, limit=10)

    assert got[0] == ("QUIET_BUT_DENSE", 9)
    assert dict(got)["VIRAL_BUT_THIN"] == 2


def test_ledger_blocks_ids_the_anti_join_can_never_learn_about():
    """An absent object never gains a post row, so the anti-join cannot exclude
    it. Selection is deterministic, so without the ledger it occupies the same
    slot at the head of the ranking on every pass forever."""
    rows = _retweets("DEAD", [f"a{i}" for i in range(50)]) + _retweets(
        "ALIVE", [f"b{i}" for i in range(20)]
    )
    con, view = _con_with_posts(rows)

    assert [oid for oid, _ in candidate_parents(con, view, limit=10)] == ["DEAD", "ALIVE"]
    got = candidate_parents(con, view, limit=10, blocked=["DEAD"])
    assert [oid for oid, _ in got] == ["ALIVE"]


def test_held_stage_is_not_windowed():
    """The amplifier scan prunes on `dt`; the held-ids stage must not. A
    windowed answer calls every older post missing and spends the whole budget
    refetching things already in R2, every pass, forever."""
    old = NOW - timedelta(days=60)
    rows = _retweets("OLD", ["a1", "a2", "a3"]) + _retweets("GONE", ["b1", "b2", "b3"])
    rows.append(
        {
            "platform_post_id": "OLD",
            "author_id": "origin",
            "created_at": old,
            "collected_at": old,
        }
    )
    con, view = _con_with_posts(rows)

    got = candidate_parents(con, view, limit=10, lookback_days=2)
    assert [oid for oid, _ in got] == ["GONE"]


def test_window_prunes_the_amplifier_scan():
    """The other half of the same predicate: retweets collected outside the
    window do not contribute amplifiers, so the scan cost stops growing with
    the corpus rather than with the useful window."""
    old = NOW - timedelta(days=60)
    rows = _retweets("STALE", ["a1", "a2", "a3"], collected_at=old, dt=old.date())
    rows += _retweets("FRESH", ["b1", "b2", "b3"])
    con, view = _con_with_posts(rows)

    assert [oid for oid, _ in candidate_parents(con, view, limit=10, lookback_days=2)] == [
        "FRESH"
    ]
    assert {oid for oid, _ in candidate_parents(con, view, limit=10)} == {
        "FRESH",
        "STALE",
    }


def test_a_null_post_id_does_not_stall_selection():
    """`NOT EXISTS`, not `NOT IN`. Under three-valued logic a single NULL id on
    the inner side makes `NOT IN` NULL for every candidate and the selector
    silently returns nothing - the stall `census.censused_expr` documents."""
    rows = _retweets("GONE", ["a1", "a2", "a3"])
    rows.append({"platform_post_id": None, "author_id": "ghost"})
    con, view = _con_with_posts(rows)

    assert [oid for oid, _ in candidate_parents(con, view, limit=10)] == ["GONE"]


def test_each_aggregate_is_staged_and_the_widest_one_is_released():
    """Staged, not one query. A query whose every stage fits can still OOM when
    DuckDB runs the stages concurrently and their peaks add - so each aggregate
    is materialised before the next reads it, and the pair set (the widest
    relation here, one row per retweet rather than per parent) is dropped once
    the amplifier counts are out of it."""
    rows = _retweets("X", ["a1", "a2", "a3"]) + _retweets("Y", ["b1"])
    con, view = _con_with_posts(rows)
    candidate_parents(con, view, limit=10)

    staged = {
        r[0]
        for r in con.sql(
            "SELECT table_name FROM duckdb_tables() WHERE table_name LIKE '_pb_%'"
        ).fetchall()
    }
    assert {"_pb_amps", "_pb_held", "_pb_missing"} <= staged
    assert "_pb_pairs" not in staged


# --- ledger ------------------------------------------------------------------


def test_state_roundtrip(tmp_path):
    path = tmp_path / "parent_backfill.json"
    save_state({"1": BackfillEntry(fetched_at=NOW.isoformat(), amplifiers=7)}, path)
    loaded = load_state(path)
    assert loaded["1"].status == "ok"
    assert loaded["1"].amplifiers == 7
    assert load_state(tmp_path / "absent.json") == {}


def test_absent_is_permanent_but_a_failed_request_retries():
    """The distinction requirement 4 turns on: a deleted tweet will never
    succeed, whereas a rate-limited request says nothing about the object."""
    absent = BackfillEntry(fetched_at=NOW.isoformat(), status="not_found", attempts=1)
    failed_once = BackfillEntry(fetched_at=NOW.isoformat(), status="failed", attempts=1)
    failed_out = BackfillEntry(fetched_at=NOW.isoformat(), status="failed", attempts=3)
    fetched = BackfillEntry(fetched_at=NOW.isoformat(), status="ok", attempts=1)

    assert is_retryable(None)
    assert is_retryable(failed_once, max_attempts=3)
    assert not is_retryable(failed_out, max_attempts=3)
    assert not is_retryable(absent)
    assert not is_retryable(fetched)

    assert set(blocked_ids({"a": absent, "b": failed_once, "c": fetched})) == {"a", "c"}


def test_pruning_keeps_failures_and_drops_stale_successes():
    """A fetched id has a post row, so the anti-join owns it; an absent one has
    no other record anywhere and pruning it re-opens the starvation."""
    stale = (NOW - timedelta(days=30)).isoformat()
    entries = {
        "old_ok": BackfillEntry(fetched_at=stale, status="ok"),
        "new_ok": BackfillEntry(fetched_at=NOW.isoformat(), status="ok"),
        "old_absent": BackfillEntry(fetched_at=stale, status="not_found"),
        "old_failed": BackfillEntry(fetched_at=stale, status="failed", attempts=3),
    }
    kept = prune_state(entries, retain_hours=168)
    assert set(kept) == {"new_ok", "old_absent", "old_failed"}


def test_summary_reports_coverage_bought_not_ids_fetched():
    """`rows_per_request` is the number that decides whether to keep going: the
    ranking is densest-first over a backlog that is 90.4% single-amplifier
    objects, so it decays towards 1.0."""
    entries = {
        "a": BackfillEntry(fetched_at=NOW.isoformat(), status="ok", amplifiers=40),
        "b": BackfillEntry(fetched_at=NOW.isoformat(), status="ok", amplifiers=1),
        # An absent parent unlocks nothing, so it must not enter the average.
        "c": BackfillEntry(fetched_at=NOW.isoformat(), status="not_found", amplifiers=99),
    }
    summary = backfill_summary(entries)
    assert summary["tracked"] == 3
    assert summary["ok"] == 2
    assert summary["not_found"] == 1
    assert summary["rows_unlocked"] == 41
    assert summary["rows_per_request"] == 20.5
    assert backfill_summary({})["rows_unlocked"] == 0


# --- the pass ----------------------------------------------------------------


class _StubStorage:
    """The Storage surface `backfill_parents` touches."""

    def __init__(self, con: duckdb.DuckDBPyConnection | None = None):
        self.con = con or duckdb.connect()
        self.writes: list[tuple[str, int]] = []
        self.written_posts: list[Post] = []
        self.written_authors: list[Author] = []

    def posts_view(self, platform: str = "*", target_type: str = "*") -> str:
        return "posts_tbl"

    def write_posts(self, posts, target_type: str, now=None) -> str | None:
        if not posts:
            return None
        self.writes.append((target_type, len(posts)))
        self.written_posts.extend(posts)
        return f"posts/type={target_type}/run=test.parquet"

    def write_authors(self, authors, now=None) -> str | None:
        self.written_authors.extend(authors)
        return "authors/run=test.parquet" if authors else None


class _StubCollector:
    """`hydrate` per the real contract: yields nothing for an absent object,
    raises for a failed request."""

    platform = "x"

    def __init__(self, absent: set[str] = frozenset(), raising: set[str] = frozenset()):
        self.absent = set(absent)
        self.raising = set(raising)
        self.requested: list[str] = []
        self._authors: dict[str, Author] = {}

    async def hydrate(self, post_ids):
        for pid in post_ids:
            self.requested.append(pid)
            if pid in self.raising:
                raise RuntimeError("rate limited")
            if pid in self.absent:
                continue
            self._authors[f"u-{pid}"] = Author(
                platform="x", platform_user_id=f"u-{pid}", handle=f"h-{pid}"
            )
            yield Post(
                platform="x",
                platform_post_id=pid,
                author_id=f"u-{pid}",
                author_handle=f"h-{pid}",
                text=f"parent {pid}",
                created_at=NOW,
                url=f"https://x.com/h-{pid}/status/{pid}",
            )

    def collected_authors(self) -> list[Author]:
        out = list(self._authors.values())
        self._authors = {}
        return out


def _run(coro):
    return asyncio.run(coro)


def test_rows_land_in_the_quarantined_targeted_partition(tmp_path):
    """The constraint the whole task hangs on: `hydrated` is a BASELINE type and
    167,220 rows there would move the baseline composition further than the
    2026-08-06 conversation widening did."""
    storage, collector = _StubStorage(), _StubCollector()
    counts = _run(
        backfill_parents(
            collector,
            storage,
            limit=2,
            state_path=tmp_path / "s.json",
            candidates=[("A", 40), ("B", 5)],
        )
    )
    assert counts["hydrated"] == 2
    assert {t for t, _ in storage.writes} == {TARGET_TYPE}
    assert TARGET_TYPE == "parent_backfill"
    assert "hydrated" not in {t for t, _ in storage.writes}


def test_authors_are_written_because_fast_retweet_needs_the_parents_author(tmp_path):
    storage, collector = _StubStorage(), _StubCollector()
    _run(
        backfill_parents(
            collector, storage, limit=1, state_path=tmp_path / "s.json",
            candidates=[("A", 40)],
        )
    )
    assert [a.platform_user_id for a in storage.written_authors] == ["u-A"]


def test_ledger_resume_does_not_refetch(tmp_path):
    """A restart must spend its budget on new ids, not on the ones the previous
    pass already paid for.

    Both passes run the real selection query against an unchanged corpus, so the
    ledger is the only thing that can move the second pass off the head of the
    ranking. `posts_tbl` deliberately does not grow with what was hydrated:
    that isolates the ledger from the anti-join, which would otherwise mask it.
    """
    path = tmp_path / "s.json"
    rows = (
        _retweets("A", [f"a{i}" for i in range(40)])
        + _retweets("B", [f"b{i}" for i in range(20)])
        + _retweets("C", ["c1", "c2", "c3", "c4", "c5"])
    )
    con, _ = _con_with_posts(rows)

    first = _StubCollector()
    _run(backfill_parents(first, _StubStorage(con), limit=2, state_path=path))
    assert first.requested == ["A", "B"]
    assert set(blocked_ids(load_state(path))) == {"A", "B"}

    second = _StubCollector()
    _run(backfill_parents(second, _StubStorage(con), limit=3, state_path=path))
    assert second.requested == ["C"]


def test_absent_and_failed_are_recorded_differently(tmp_path):
    path = tmp_path / "s.json"
    collector = _StubCollector(absent={"DEAD"}, raising={"BOOM"})
    counts = _run(
        backfill_parents(
            collector,
            _StubStorage(),
            limit=3,
            state_path=path,
            candidates=[("A", 40), ("DEAD", 30), ("BOOM", 20)],
        )
    )
    assert counts == {
        "selected": 3,
        "hydrated": 1,
        "not_found": 1,
        "failed": 1,
        "authors": 1,
        # Only A came back, so only A's 40 amplifiers are unlocked. The absent
        # and failed ids selected 50 more rows between them and unlocked none.
        "retweet_rows_unlocked": 40,
    }
    entries = load_state(path)
    assert entries["DEAD"].status == "not_found"
    assert entries["BOOM"].status == "failed"
    assert entries["A"].status == "ok"
    # The absence is terminal, the failure is not.
    assert set(blocked_ids(entries)) == {"A", "DEAD"}


def test_a_failure_gives_up_after_max_attempts(tmp_path):
    path = tmp_path / "s.json"
    for _ in range(2):
        _run(
            backfill_parents(
                _StubCollector(raising={"BOOM"}),
                _StubStorage(),
                limit=1,
                state_path=path,
                candidates=[("BOOM", 20)],
                max_attempts=2,
            )
        )
    entries = load_state(path)
    assert entries["BOOM"].attempts == 2
    assert not is_retryable(entries["BOOM"], max_attempts=2)


def test_partial_progress_survives_an_abort_mid_pass(tmp_path):
    """Flushing every `flush_every` ids is what stops a rate-limit abort
    discarding every API call made since the pass started."""
    path = tmp_path / "s.json"
    storage = _StubStorage()

    class _AbortingCollector(_StubCollector):
        async def hydrate(self, post_ids):
            for pid in post_ids:
                if pid == "C":
                    raise KeyboardInterrupt
                async for p in super().hydrate([pid]):
                    yield p

    with pytest.raises(KeyboardInterrupt):
        _run(
            backfill_parents(
                _AbortingCollector(),
                storage,
                limit=4,
                state_path=path,
                flush_every=2,
                candidates=[("A", 40), ("B", 30), ("C", 20), ("D", 10)],
            )
        )
    assert storage.writes == [(TARGET_TYPE, 2)]
    assert set(load_state(path)) == {"A", "B"}


def test_the_pass_is_announced_in_coverage_not_in_ids(tmp_path):
    """`fast_retweet` coverage was 161,146 of 364,287 retweet rows on
    2026-09-05. A pass reported as "500 ids" hides both what it bought and the
    decay: densest-first over a 90.4% single-amplifier backlog means late
    passes unlock roughly one row per request."""
    rows = (
        _retweets("BIG", [f"a{i}" for i in range(30)])
        + _retweets("SMALL", ["b1"])
        + _retweets("HELD", ["c1", "c2"])
    )
    rows.append({"platform_post_id": "HELD", "author_id": "origin"})
    con, _ = _con_with_posts(rows)
    stats: dict = {}
    counts = _run(
        backfill_parents(
            _StubCollector(),
            _StubStorage(con),
            limit=2,
            state_path=tmp_path / "s.json",
            stats=stats,
        )
    )
    assert stats["retweet_rows"] == 33
    assert stats["held_rows"] == 2
    assert stats["coverage"] == round(2 / 33, 4)
    assert stats["selected_rows"] == 31
    assert counts["retweet_rows_unlocked"] == 31


def test_an_empty_candidate_set_is_not_an_error(tmp_path):
    counts = _run(
        backfill_parents(
            _StubCollector(), _StubStorage(), limit=10,
            state_path=tmp_path / "s.json", candidates=[],
        )
    )
    assert counts["selected"] == 0
    assert counts["hydrated"] == 0


def test_pass_selects_through_the_query_when_given_no_candidates(tmp_path):
    """End to end on the real SQL: selection, ranking, ledger and the targeted
    write, with only the network stubbed."""
    rows = (
        _retweets("DENSE", [f"a{i}" for i in range(40)])
        + _retweets("PAIR", ["b1", "b2", "b3"])
        + _retweets("LONE", ["c1"])
        + _retweets("HUB", [f"d{i}" for i in range(130)])
    )
    con, _ = _con_with_posts(rows)
    storage = _StubStorage(con)
    collector = _StubCollector()
    stats: dict = {}
    counts = _run(
        backfill_parents(
            collector,
            storage,
            limit=2,
            state_path=tmp_path / "s.json",
            stats=stats,
        )
    )
    assert collector.requested == ["HUB", "DENSE"]
    assert counts["hydrated"] == 2
    assert counts["retweet_rows_unlocked"] == 170
    assert storage.writes == [(TARGET_TYPE, 2)]
    assert stats["missing_parents"] == 4
    assert stats["missing_in_band"] == 2
    assert stats["band_only"] is False

"""Recording the posts that vanish, and refusing to guess why.

Deletion is the strongest concealment signal this collector can reach, and
before 2026-09-15 `refresh_metrics` observed it and threw it away. What is
tested here is the discipline around it: absence is recorded, a failed request
never becomes a deletion, and a cause is never attributed without checking.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from kenya_monitor.collectors.base import (
    CAUSE_AUTHOR_GONE,
    CAUSE_POST_DELETED,
    CAUSE_UNRESOLVED,
    STATUS_ABSENT,
    STATUS_PRESENT,
)
from kenya_monitor.collectors.x import XCollector


class _Tweet:
    def __init__(self, tid):
        self.id = tid
        self.likeCount = self.replyCount = self.retweetCount = 1
        self.quoteCount = self.viewCount = 1


class _Api:
    """Posts in `present` come back; everything else is gone. `users` decides
    whether the author still resolves. `raises` makes the request fail."""

    def __init__(self, present=(), users=(), raises=()):
        self.present = {str(p) for p in present}
        self.users = {str(u) for u in users}
        self.raises = {str(r) for r in raises}
        self.user_lookups = 0

    async def tweet_details(self, twid, kv=None):
        if str(twid) in self.raises:
            raise RuntimeError("rate limited")
        return _Tweet(twid) if str(twid) in self.present else None

    async def user_by_id(self, uid, kv=None):
        self.user_lookups += 1
        return object() if str(uid) in self.users else None


def _collector(api) -> XCollector:
    c = XCollector.__new__(XCollector)
    c.api = api
    return c


def _run(collector, ids, **kw):
    async def go():
        return [m async for m in collector.refresh_metrics(ids, **kw)]
    return asyncio.run(go())


def test_a_post_that_is_gone_is_recorded_rather_than_skipped():
    """The bug: `if tw is None: continue` observed the disappearance and
    discarded it, so nothing downstream could tell it from "never checked"."""
    got = _run(_collector(_Api(present=["1"])), ["1", "2"])

    assert [m.platform_post_id for m in got] == ["1", "2"]
    assert [m.status for m in got] == [STATUS_PRESENT, STATUS_ABSENT]


def test_an_absent_post_whose_author_is_alive_is_a_deletion():
    api = _Api(present=[], users=["111"])

    got = _run(_collector(api), ["1"], authors={"1": "111"})

    assert got[0].absence_cause == CAUSE_POST_DELETED
    assert api.user_lookups == 1


def test_an_absent_post_whose_author_is_gone_is_a_different_signal():
    """Suspension and deletion are both concealment and they are not the same
    thing; collapsing them would lose the distinction the column exists for."""
    got = _run(_collector(_Api(present=[], users=[])), ["1"], authors={"1": "111"})

    assert got[0].absence_cause == CAUSE_AUTHOR_GONE


def test_absence_stays_unresolved_when_resolution_is_off():
    api = _Api(present=[], users=["111"])

    got = _run(_collector(api), ["1"], authors={"1": "111"}, resolve_absence=False)

    assert got[0].status == STATUS_ABSENT
    assert got[0].absence_cause == CAUSE_UNRESOLVED
    assert api.user_lookups == 0


def test_absence_stays_unresolved_when_the_author_is_unknown():
    api = _Api(present=[], users=["111"])

    got = _run(_collector(api), ["1"])

    assert got[0].absence_cause == CAUSE_UNRESOLVED
    assert api.user_lookups == 0


def test_a_failed_request_is_never_recorded_as_a_deletion():
    """Our own rate limiting must not manufacture deletions - that is the
    failure mode the whole column exists to avoid."""
    got = _run(_collector(_Api(present=["2"], raises=["1"])), ["1", "2"])

    assert [m.platform_post_id for m in got] == ["2"]
    assert got[0].status == STATUS_PRESENT


def test_a_present_post_carries_no_absence_cause():
    got = _run(_collector(_Api(present=["1"])), ["1"])

    assert got[0].status == STATUS_PRESENT
    assert got[0].absence_cause is None


def test_the_candidate_query_carries_the_author_of_every_post():
    """The author is what resolves an absence to a deletion or a suspension, so
    `collect_metrics` selects it alongside the post id. Run against DuckDB
    rather than a stub: the column was missing from the CTE for six days, which
    a mocked storage cannot catch."""
    import duckdb
    import pyarrow as pa

    from kenya_monitor.runner import collect_metrics

    now = datetime.now(timezone.utc)
    posts = pa.table(
        {
            "platform": pa.array(["x", "x"], type=pa.string()),
            "platform_post_id": pa.array(["p1", "p2"], type=pa.string()),
            "author_id": pa.array(["a1", "a2"], type=pa.string()),
            "like_count": pa.array([100, 1], type=pa.int64()),
            "quote_count": pa.array([0, 0], type=pa.int64()),
            "repost_count": pa.array([0, 0], type=pa.int64()),
            "collected_at": pa.array([now, now], type=pa.timestamp("us", tz="UTC")),
        }
    )
    con = duckdb.connect()
    con.register("posts_tbl", posts)

    class _Storage:
        def posts_view(self, platform="*", target_type="*"):
            return "posts_tbl"

        def query(self, sql):
            return con.sql(sql)

        def write_metrics(self, snapshots):
            return None

    class _Collector:
        platform = "x"

        def __init__(self):
            self.authors = None

        async def refresh_metrics(self, ids, authors=None, **kw):
            self.authors = authors
            for pid in ids:
                yield _Snapshot(pid)

    collector = _Collector()
    counts = asyncio.run(collect_metrics(collector, _Storage(), top_pct=1.0))

    assert counts["metrics"] == 2
    assert collector.authors == {"p1": "a1", "p2": "a2"}


class _Snapshot:
    def __init__(self, pid):
        self.platform_post_id = pid
        self.status = STATUS_PRESENT
        self.absence_cause = None

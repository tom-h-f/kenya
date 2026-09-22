"""When the cycle's gated steps run.

Two failures measured on pi0 after the 2026-09-21 deploy motivate these: due
checks evaluated at cycle start skipped a control pass that fell due during a
6.5-hour cycle, and a depth pass that selected nothing reran its 25-31 minute
selection query on every cycle because nothing recorded that it had tried.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

import kenya_monitor.scheduler as sched

NOW = datetime.now(timezone.utc)


@pytest.fixture
def attempts_path(tmp_path, monkeypatch):
    path = tmp_path / "step_attempts.json"
    monkeypatch.setattr(sched, "STEP_ATTEMPTS_PATH", path)
    return path


def test_due_is_decided_when_the_step_is_reached_not_when_the_cycle_starts(attempts_path):
    state = {"due": False}
    ran = []

    async def step():
        ran.append(1)

    gated = sched._gated("control", lambda: state["due"], step, record=False)
    # Built at cycle start while not due; falls due before the cycle reaches it.
    state["due"] = True
    asyncio.run(gated())

    assert ran == [1]


def test_a_skipped_step_records_no_attempt(attempts_path):
    async def step():
        raise AssertionError("must not run")

    asyncio.run(sched._gated("deep_timelines", lambda: False, step)())

    assert sched.load_attempts() == {}


def test_the_attempt_is_recorded_before_the_step_runs_so_a_failure_counts(attempts_path):
    async def step():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        asyncio.run(sched._gated("follow_crawl", lambda: True, step)())

    assert "follow_crawl" in sched.load_attempts()


def test_an_empty_pass_is_not_due_again_until_its_interval_passes(attempts_path, monkeypatch):
    """The ledger's last fetch is days old, so the ledger alone says due. The
    recorded attempt is what stops the rerun."""
    import kenya_monitor.deep_timelines as dt

    old = (NOW - timedelta(days=5)).isoformat()
    monkeypatch.setattr(dt, "load_state", lambda *a, **k: {})
    monkeypatch.setattr(dt, "timeline_summary", lambda *a, **k: {"latest_fetch": old})

    assert sched._accounts_due() is True
    sched.record_attempt("deep_timelines", now=NOW - timedelta(hours=1))
    assert sched._accounts_due() is False
    sched.record_attempt(
        "deep_timelines", now=NOW - timedelta(hours=sched.DEPTH_EVERY_HOURS + 1)
    )
    assert sched._accounts_due() is True


def test_follow_crawl_waits_a_day_from_its_latest_crawl(attempts_path, monkeypatch):
    import kenya_monitor.follow_crawl as fc

    recent = fc.CrawlEntry(handle="a", crawled_at=(NOW - timedelta(hours=3)).isoformat())
    monkeypatch.setattr(fc, "load_crawl_state", lambda *a, **k: {"1": recent})
    assert sched._follow_crawl_due() is False

    stale = fc.CrawlEntry(handle="a", crawled_at=(NOW - timedelta(hours=30)).isoformat())
    monkeypatch.setattr(fc, "load_crawl_state", lambda *a, **k: {"1": stale})
    assert sched._follow_crawl_due() is True


def test_a_corrupt_attempts_file_reads_as_empty(attempts_path):
    attempts_path.write_text("{not json")

    assert sched.load_attempts() == {}

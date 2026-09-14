"""Depth as a standing cost: the cadence gate on the two depth arms.

`run_scheduler` runs forever, so what is tested here is the decision it makes
each cycle - whether an arm is due - and the property that decision exists to
have: it is read from the arm's own ledger, so it survives the restarts that
made a cycle counter useless.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from kenya_monitor.deep_timelines import DeepEntry, load_state, save_state, timeline_summary
from kenya_monitor.parent_backfill import BackfillEntry, backfill_summary
from kenya_monitor.parent_backfill import load_state as load_parent_state
from kenya_monitor.parent_backfill import save_state as save_parent_state
from kenya_monitor.scheduler import _depth_due

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def test_an_arm_that_has_never_run_is_due():
    assert _depth_due(None, 12, now=NOW)
    assert _depth_due("", 12, now=NOW)


def test_an_arm_is_due_once_the_interval_has_elapsed():
    assert not _depth_due((NOW - timedelta(hours=11)).isoformat(), 12, now=NOW)
    assert _depth_due((NOW - timedelta(hours=12)).isoformat(), 12, now=NOW)
    assert _depth_due((NOW - timedelta(hours=30)).isoformat(), 12, now=NOW)


def test_the_deep_timeline_cadence_survives_a_restart(tmp_path):
    """The bug this replaces: `cycle` resets to 0 on every restart, so a
    cycle-modulo schedule deferred the hate steps past every redeploy. A pass
    written to the ledger an hour ago must still block the next process."""
    path = tmp_path / "deep_timeline.json"
    save_state(
        {"7": DeepEntry(fetched_at=(NOW - timedelta(hours=1)).isoformat(), posts=200)},
        path=path,
    )

    latest = timeline_summary(load_state(path=path))["latest_fetch"]

    assert not _depth_due(latest, 12, now=NOW)
    assert _depth_due(latest, 12, now=NOW + timedelta(hours=11))


def test_the_parent_backfill_cadence_reads_its_own_ledger(tmp_path):
    path = tmp_path / "parent_backfill.json"
    save_parent_state(
        {"1": BackfillEntry(fetched_at=(NOW - timedelta(hours=20)).isoformat())},
        path=path,
    )

    latest = backfill_summary(load_parent_state(path=path))["latest_fetch"]

    assert _depth_due(latest, 12, now=NOW)


def test_the_two_arms_are_gated_apart(tmp_path):
    """One arm running does not reset the other's clock: they consume different
    budgets and one can fail while the other is fine."""
    deep = tmp_path / "deep_timeline.json"
    parents = tmp_path / "parent_backfill.json"
    save_state({"7": DeepEntry(fetched_at=NOW.isoformat())}, path=deep)
    save_parent_state(
        {"1": BackfillEntry(fetched_at=(NOW - timedelta(hours=13)).isoformat())},
        path=parents,
    )

    assert not _depth_due(timeline_summary(load_state(path=deep))["latest_fetch"], 12, now=NOW)
    assert _depth_due(
        backfill_summary(load_parent_state(path=parents))["latest_fetch"], 12, now=NOW
    )

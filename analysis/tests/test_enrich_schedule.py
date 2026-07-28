"""Unit tests for the coordination-refresh schedule in the enrich worker."""

import pytest

from kma import enrich


@pytest.fixture
def loop(monkeypatch):
    """Run run_loop for a bounded number of cycles, recording coordination runs."""
    state = {"coord": 0, "cycles": 0}

    monkeypatch.setattr(enrich, "_subprocess_pass", lambda *a, **k: 0)

    def fake_coord():
        state["coord"] += 1
        return True

    monkeypatch.setattr(enrich, "_coordination_pass", fake_coord)

    clock = {"t": 0.0}
    monkeypatch.setattr(enrich.time, "monotonic", lambda: clock["t"])

    def fake_sleep(seconds):
        state["cycles"] += 1
        clock["t"] += seconds
        if state["cycles"] >= state["max_cycles"]:
            raise KeyboardInterrupt

    monkeypatch.setattr(enrich.time, "sleep", fake_sleep)
    monkeypatch.setattr(enrich.random, "uniform", lambda a, b: 3600.0)  # 1h idle

    def run(max_cycles, **kw):
        state["max_cycles"] = max_cycles
        with pytest.raises(KeyboardInterrupt):
            enrich.run_loop(**kw)
        return state

    return run


def test_coordination_runs_on_the_first_cycle(loop):
    """Targeting should not wait `coord_hours` for its first cluster run."""
    assert loop(1, coord_hours=6)["coord"] == 1


def test_coordination_is_rate_limited_not_per_cycle(loop):
    """1h idle per cycle at coord_hours=6: the initial run, then nothing until
    the 7th cycle starts at t=6h. Six cycles must not trigger a second run."""
    assert loop(6, coord_hours=6)["coord"] == 1
    assert loop(7, coord_hours=6)["coord"] == 2


def test_coordination_can_be_disabled(loop):
    assert loop(5, coord_hours=0)["coord"] == 0


def test_coordination_failure_does_not_kill_the_worker(loop, monkeypatch):
    """A stale cluster run degrades targeting; a dead worker stops enrichment."""
    monkeypatch.setattr(enrich, "_coordination_pass", lambda: (_ for _ in ()).throw(RuntimeError))
    with pytest.raises(RuntimeError):
        enrich.run_loop(coord_hours=6)


def test_default_refresh_interval_is_hours_not_minutes():
    assert enrich.COORD_REFRESH_HOURS >= 1

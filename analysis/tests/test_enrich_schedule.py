"""Unit tests for the pass wiring in the enrich worker."""

import pytest

from kma import enrich


@pytest.fixture
def passes(monkeypatch):
    """Record the flag sets each isolated pass is launched with."""
    seen: list[list[str]] = []
    monkeypatch.setattr(
        enrich, "_subprocess_pass", lambda flags, *a, **k: seen.append(list(flags)) or 0
    )

    def stop(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(enrich.time, "sleep", stop)
    monkeypatch.setattr(enrich.random, "uniform", lambda a, b: 1.0)

    def run(**kw):
        with pytest.raises(KeyboardInterrupt):
            enrich.run_loop(**kw)
        return seen

    return run


def test_incitement_runs_before_hate(passes):
    """`hatespeech._measure_frame` resolves coded_suspect at scoring time and
    writes a non-null False when no NLI row exists, so a post scored before its
    NLI is permanently marked not-coded. Order is the fix, not a preference."""
    launched = passes()
    order = [next(p for p in enrich.PASSES if f"--no-{p}" not in flags) for flags in launched]

    assert order == list(enrich.PASSES)
    assert order.index("incitement") < order.index("hate")


def test_each_pass_disables_every_other_one(passes):
    """Isolation is the peak-memory guard; a missed flag runs two models in one
    process, which is the failure it exists to prevent."""
    for flags in passes():
        enabled = [p for p in enrich.PASSES if f"--no-{p}" not in flags]
        assert len(enabled) == 1, f"{flags} leaves {enabled} enabled"


def test_a_disabled_pass_is_not_launched(passes):
    launched = passes(incitement=False)
    assert not any("--no-incitement" not in f for f in launched)
    assert len(launched) == len(enrich.PASSES) - 1


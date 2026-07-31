"""Unit tests for the hate-expansion frontier ledger (pure, no R2)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from kenya_monitor import hate_frontier as hf

NOW = datetime.now(timezone.utc)


def seed(uid: str, handle: str, score: float = 0.5) -> dict:
    return {"author_id": uid, "handle": handle, "hate_seed_score": score, "n_posts": 20}


def entry(uid_handle: str, days_ago: float = 0, status: str = "ok", attempts: int = 0):
    return hf.ExpandEntry(
        handle=uid_handle,
        expanded_at=(NOW - timedelta(days=days_ago)).isoformat(),
        status=status,
        attempts=attempts,
    )


def test_state_round_trips(tmp_path):
    path = tmp_path / "hate_expand.json"
    assert hf.load(path) == {}
    entries = {"u1": entry("alpha")}
    hf.save(entries, path)
    back = hf.load(path)
    assert back["u1"].handle == "alpha"
    assert back["u1"].status == "ok"


def test_save_is_atomic(tmp_path):
    """A crash mid-write must leave the previous ledger intact, not a truncated
    file - every other state module in the collector gets this wrong."""
    path = tmp_path / "hate_expand.json"
    hf.save({"u1": entry("alpha")}, path)
    original = path.read_text()
    hf.save({"u1": entry("alpha"), "u2": entry("beta")}, path)
    assert len(hf.load(path)) == 2
    assert not list(tmp_path.glob("*.tmp"))  # temp file cleaned up
    assert original != path.read_text()


def test_unreadable_state_starts_fresh_rather_than_crashing(tmp_path):
    path = tmp_path / "hate_expand.json"
    path.write_text("{ this is not json")
    assert hf.load(path) == {}


def test_never_expanded_is_due():
    assert hf.is_due(None) is True


def test_recently_expanded_is_not_due():
    assert hf.is_due(entry("a", days_ago=0), refresh_days=3) is False
    assert hf.is_due(entry("a", days_ago=5), refresh_days=3) is True


def test_repeated_failures_stop_being_retried():
    """follow_crawl retries failures forever with no backoff; this must not."""
    assert hf.is_due(entry("a", days_ago=99, status="failed", attempts=1), max_attempts=3) is True
    assert hf.is_due(entry("a", days_ago=99, status="failed", attempts=3), max_attempts=3) is False


def test_frontier_preserves_caller_ranking():
    """The seed order IS the priority - that is the point of a frontier over the
    FIFO queue follow_crawl uses."""
    ranked = [seed(f"u{i}", f"h{i}", score=1.0 - i / 10) for i in range(6)]
    got = hf.select_frontier(ranked, {}, max_accounts=3)
    assert [g["handle"] for g in got] == ["h0", "h1", "h2"]


def test_frontier_advances_past_expanded_accounts():
    """The bug this exists for: without a ledger the same top-N is re-pulled
    every pass and the frontier never moves outward."""
    ranked = [seed(f"u{i}", f"h{i}") for i in range(6)]
    entries: dict[str, hf.ExpandEntry] = {}
    first = hf.select_frontier(ranked, entries, max_accounts=3)
    for row in first:
        entries = hf.record(entries, row)
    second = hf.select_frontier(ranked, entries, max_accounts=3)
    assert [g["handle"] for g in first] == ["h0", "h1", "h2"]
    assert [g["handle"] for g in second] == ["h3", "h4", "h5"]
    assert set(first[0].keys()) & {"author_id", "handle"}


def test_exhausted_accounts_are_skipped_by_the_frontier():
    ranked = [seed("u0", "h0"), seed("u1", "h1")]
    entries = {"u0": entry("h0", days_ago=99, status="failed", attempts=3)}
    got = hf.select_frontier(ranked, entries, max_accounts=5)
    assert [g["handle"] for g in got] == ["h1"]


def test_record_accumulates_attempts_on_failure_and_resets_on_success():
    entries: dict[str, hf.ExpandEntry] = {}
    row = seed("u1", "alpha")
    for _ in range(2):
        entries = hf.record(entries, row, status="failed")
    assert entries["u1"].attempts == 2
    entries = hf.record(entries, row, status="ok")
    assert entries["u1"].attempts == 0
    assert entries["u1"].status == "ok"


def test_prune_drops_stale_entries():
    entries = {"old": entry("a", days_ago=99), "new": entry("b", days_ago=1)}
    kept = hf.prune(entries, keep_days=30)
    assert set(kept) == {"new"}


def test_summary_counts_exhausted_separately():
    entries = {
        "a": entry("a"),
        "b": entry("b", status="failed", attempts=1),
        "c": entry("c", status="failed", attempts=hf.HATE_EXPAND_MAX_ATTEMPTS),
    }
    s = hf.summary(entries)
    assert s == {"tracked": 3, "ok": 1, "failed": 2, "exhausted": 1}


def test_candidates_without_an_id_are_ignored():
    assert hf.select_frontier([{"handle": "nope"}], {}, max_accounts=5) == []


def test_saved_file_is_valid_json_with_an_envelope(tmp_path):
    path = tmp_path / "hate_expand.json"
    hf.save({"u1": entry("alpha")}, path)
    raw = json.loads(path.read_text())
    assert "updated_at" in raw and "entries" in raw

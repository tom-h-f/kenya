from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import duckdb
import pyarrow as pa

from kenya_monitor.collectors.base import Author, FollowEdge
from kenya_monitor.follow_crawl import (
    CrawlEntry,
    crawl_follows,
    discover_from_edges,
    is_due,
    load_crawl_state,
    save_crawl_state,
)

NOW = datetime.now(timezone.utc)


def test_crawl_state_roundtrip(tmp_path):
    path = tmp_path / "follow_crawl.json"
    entries = {"u1": CrawlEntry(handle="alice", crawled_at=NOW.isoformat(), edge_count=10)}
    save_crawl_state(entries, path)
    loaded = load_crawl_state(path)
    assert loaded["u1"].handle == "alice"
    assert loaded["u1"].edge_count == 10


def test_is_due_respects_refresh_window():
    fresh = CrawlEntry(handle="a", crawled_at=NOW.isoformat())
    stale = CrawlEntry(
        handle="b",
        crawled_at=(NOW - timedelta(days=40)).isoformat(),
    )
    assert not is_due(fresh, refresh_days=30)
    assert is_due(stale, refresh_days=30)
    assert is_due(None, refresh_days=30)


def test_discover_from_edges_finds_uncrawled():
    authors = pa.table(
        {
            "platform": pa.array(["x", "x"], type=pa.string()),
            "platform_user_id": pa.array(["1", "2"], type=pa.string()),
            "handle": pa.array(["alice", "bob"], type=pa.string()),
            "collected_at": pa.array([NOW, NOW], type=pa.timestamp("us", tz="UTC")),
        }
    )
    follows = pa.table(
        {
            "platform": pa.array(["x"], type=pa.string()),
            "follower_id": pa.array(["1"], type=pa.string()),
            "followed_id": pa.array(["2"], type=pa.string()),
            "collected_at": pa.array([NOW], type=pa.timestamp("us", tz="UTC")),
        }
    )
    con = duckdb.connect()
    con.register("authors_tbl", authors)
    con.register("follows_tbl", follows)
    entries = {
        "1": CrawlEntry(handle="alice", crawled_at=NOW.isoformat()),
    }
    found = discover_from_edges(con, "follows_tbl", "authors_tbl", entries, refresh_days=30)
    assert found == [("2", "bob")]


class _StubStorage:
    """Minimal Storage surface `crawl_follows` touches, over in-memory tables."""

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self.written_edges: list[FollowEdge] = []
        self.written_authors: list[Author] = []

    def authors_view(self, platform: str = "*") -> str:
        return "authors_tbl"

    def follows_view(self, platform: str = "*") -> str:
        return "follows_tbl"

    def write_follows(self, edges, now=None) -> str | None:
        self.written_edges.extend(edges)
        return "follows/key.parquet" if edges else None

    def write_authors(self, authors, now=None) -> str | None:
        self.written_authors.extend(authors)
        return "authors/key.parquet" if authors else None


class _StubCollector:
    def __init__(self, edges_by_handle: dict[str, list[FollowEdge]]):
        self._edges = edges_by_handle
        self.api = None

    async def follows(self, handle: str, limit: int = 0):
        for edge in self._edges.get(handle.lower(), []):
            yield edge

    def collected_authors(self):
        return []


def _con_with(authors_rows: list[tuple[str, str]]) -> duckdb.DuckDBPyConnection:
    authors = pa.table(
        {
            "platform": pa.array(["x"] * len(authors_rows), type=pa.string()),
            "platform_user_id": pa.array([r[0] for r in authors_rows], type=pa.string()),
            "handle": pa.array([r[1] for r in authors_rows], type=pa.string()),
            "collected_at": pa.array(
                [NOW] * len(authors_rows), type=pa.timestamp("us", tz="UTC")
            ),
        }
    )
    follows = pa.table(
        {
            "platform": pa.array([], type=pa.string()),
            "follower_id": pa.array([], type=pa.string()),
            "followed_id": pa.array([], type=pa.string()),
            "collected_at": pa.array([], type=pa.timestamp("us", tz="UTC")),
        }
    )
    con = duckdb.connect()
    con.register("authors_tbl", authors)
    con.register("follows_tbl", follows)
    return con


def test_seed_handle_resolves_and_crawls(tmp_path):
    """The regression guard for the `params=` slip.

    Seeds are always enqueued with an empty uid, so `_resolve_uid` runs on the
    first queue item of every run, outside the per-account try block. Passing
    DuckDB's `params` positionally raised TypeError there and killed the whole
    step - visible only as one 'follow_crawl step failed' line per cycle.
    """
    con = _con_with([("1", "alice"), ("2", "bob")])
    storage = _StubStorage(con)
    edge = FollowEdge(platform="x", follower_id="1", followed_id="2")
    collector = _StubCollector({"alice": [edge]})

    counts = asyncio.run(
        crawl_follows(
            collector,
            storage,
            seed_handles=["@alice"],
            limit=10,
            max_accounts=1,
            refresh_days=30,
            from_edges=False,
            state_path=tmp_path / "follow_crawl.json",
        )
    )

    assert counts["crawled"] == 1
    assert counts["follow_edges"] == 1
    assert counts["not_found"] == 0
    assert storage.written_edges == [edge]


def test_unresolvable_seed_is_recorded_then_left_alone(tmp_path):
    """An unresolvable handle has no uid, so its only ledger key is `handle:`.
    Without that record `is_due` hands it back at the head of the queue on every
    run, which is how one dead handle used to head the queue forever.
    """
    state_path = tmp_path / "follow_crawl.json"
    con = _con_with([("1", "alice")])
    collector = _StubCollector({})

    def run() -> dict:
        return asyncio.run(
            crawl_follows(
                collector,
                _StubStorage(con),
                seed_handles=["ghost"],
                limit=10,
                max_accounts=5,
                refresh_days=30,
                from_edges=False,
                state_path=state_path,
            )
        )

    first = run()
    assert first["not_found"] == 1
    entry = load_crawl_state(state_path)["handle:ghost"]
    assert entry.status == "not_found"
    assert entry.attempts == 1

    # Immediately again: the refresh window, not the attempt cap, is what stops
    # it. Before the fix this re-queued and re-resolved on every single run.
    assert run()["not_found"] == 0

    # Aged past the window it becomes due again, and the attempt count climbs.
    state = load_crawl_state(state_path)
    state["handle:ghost"].crawled_at = (NOW - timedelta(days=40)).isoformat()
    save_crawl_state(state, state_path)
    assert run()["not_found"] == 1
    assert load_crawl_state(state_path)["handle:ghost"].attempts == 2

    # Once capped it is never queued again, however old it gets.
    state = load_crawl_state(state_path)
    state["handle:ghost"].attempts = 3
    state["handle:ghost"].crawled_at = (NOW - timedelta(days=400)).isoformat()
    save_crawl_state(state, state_path)
    assert run()["not_found"] == 0


def test_author_directory_is_scanned_once_per_run(tmp_path, monkeypatch):
    """It used to be re-scanned per crawled account - a full pass over the whole
    authors prefix, up to max_accounts times, for data already in hand."""
    import kenya_monitor.follow_crawl as fc

    calls = {"n": 0}
    real = fc._author_directory

    def counting(con, view):
        calls["n"] += 1
        return real(con, view)

    monkeypatch.setattr(fc, "_author_directory", counting)

    con = _con_with([("1", "alice"), ("2", "bob"), ("3", "carol")])
    collector = _StubCollector(
        {
            "alice": [FollowEdge(platform="x", follower_id="1", followed_id="2")],
            "bob": [FollowEdge(platform="x", follower_id="2", followed_id="3")],
        }
    )

    counts = asyncio.run(
        crawl_follows(
            collector,
            _StubStorage(con),
            seed_handles=["alice", "bob"],
            limit=10,
            max_accounts=2,
            refresh_days=30,
            from_edges=False,
            state_path=tmp_path / "follow_crawl.json",
        )
    )

    assert counts["crawled"] == 2
    assert calls["n"] == 1


def test_failed_entries_respect_the_attempt_cap():
    capped = CrawlEntry(handle="a", crawled_at=NOW.isoformat(), status="failed", attempts=3)
    under = CrawlEntry(
        handle="b",
        crawled_at=(NOW - timedelta(days=40)).isoformat(),
        status="failed",
        attempts=1,
    )
    assert not is_due(capped, refresh_days=30, max_attempts=3)
    assert is_due(under, refresh_days=30, max_attempts=3)


def test_state_write_is_staged_through_a_temp_file(tmp_path, monkeypatch):
    """The ledger must only ever be replaced by an atomic rename.

    Writing in place means a crash mid-write truncates it, and this file is
    rewritten once per crawled account. Asserting on the rename rather than on
    the end state is deliberate: a direct `write_text` also leaves the file
    intact when the *serialisation* fails, so only the staging proves the point.
    """
    import kenya_monitor.follow_crawl as fc

    path = tmp_path / "follow_crawl.json"
    save_crawl_state({"u1": CrawlEntry(handle="alice", crawled_at=NOW.isoformat())}, path)
    before = path.read_text()

    def refuse(src, dst):
        raise RuntimeError("interrupted before the rename landed")

    monkeypatch.setattr(fc.os, "replace", refuse)
    try:
        save_crawl_state({"u2": CrawlEntry(handle="bob", crawled_at=NOW.isoformat())}, path)
    except RuntimeError:
        pass

    # New content went to the staging file; the live ledger is untouched.
    assert path.read_text() == before
    assert load_crawl_state(path)["u1"].handle == "alice"
    assert (tmp_path / "follow_crawl.json.tmp").exists()


def test_successful_write_leaves_no_staging_file(tmp_path):
    path = tmp_path / "follow_crawl.json"
    save_crawl_state({"u1": CrawlEntry(handle="alice", crawled_at=NOW.isoformat())}, path)
    assert not (tmp_path / "follow_crawl.json.tmp").exists()
    assert load_crawl_state(path)["u1"].handle == "alice"

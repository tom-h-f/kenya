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


class _PagedCollector(_StubCollector):
    """Returns, with each crawled page, the author snapshots X sends alongside
    the edges - which is where neighbour handles come from."""

    def __init__(self, edges_by_handle, authors_by_handle):
        super().__init__(edges_by_handle)
        self._authors = authors_by_handle
        self._last: list[Author] = []

    async def follows(self, handle: str, limit: int = 0):
        self._last = self._authors.get(handle.lower(), [])
        async for edge in super().follows(handle, limit):
            yield edge

    def collected_authors(self):
        return self._last


def test_neighbours_are_enqueued_from_the_crawled_page_without_a_prefix_scan(tmp_path):
    """The authors prefix is empty here, so a neighbour can only be enqueued by
    the handle its own page carried. The crawl used to build a second full
    directory of the prefix for this."""
    con = _con_with([])
    collector = _PagedCollector(
        {"alice": [FollowEdge(platform="x", follower_id="1", followed_id="2")]},
        {"alice": [Author(platform="x", platform_user_id="2", handle="bob")]},
    )

    counts = asyncio.run(
        crawl_follows(
            collector,
            _StubStorage(con),
            seed_accounts=[("1", "alice")],
            limit=10,
            max_accounts=2,
            refresh_days=30,
            from_edges=False,
            state_path=tmp_path / "follow_crawl.json",
        )
    )

    assert counts["crawled"] == 2
    assert set(load_crawl_state(tmp_path / "follow_crawl.json")) == {"1", "2"}


def test_a_seed_with_an_id_needs_no_resolution_and_is_skipped_when_fresh(tmp_path):
    """Handle-only seeds were resolved with an authors-prefix scan each, and
    4-10 of every 10 then turned out fresh. With the id, freshness is a ledger
    lookup."""
    state_path = tmp_path / "follow_crawl.json"
    save_crawl_state({"1": CrawlEntry(handle="alice", crawled_at=NOW.isoformat())}, state_path)
    con = _con_with([])
    collector = _StubCollector({"bob": []})

    counts = asyncio.run(
        crawl_follows(
            collector,
            _StubStorage(con),
            seed_accounts=[("1", "alice"), ("2", "bob")],
            limit=10,
            max_accounts=5,
            refresh_days=30,
            from_edges=False,
            state_path=state_path,
        )
    )

    assert counts["crawled"] == 1
    assert counts["not_found"] == 0
    assert counts["skipped_fresh"] == 0


def test_discovery_is_bounded_and_skips_fresh_accounts():
    ids = [str(i) for i in range(20)]
    authors = pa.table(
        {
            "platform": pa.array(["x"] * 20, type=pa.string()),
            "platform_user_id": pa.array(ids, type=pa.string()),
            "handle": pa.array([f"h{i}" for i in ids], type=pa.string()),
            "collected_at": pa.array([NOW] * 20, type=pa.timestamp("us", tz="UTC")),
        }
    )
    follows = pa.table(
        {
            "platform": pa.array(["x"] * 19, type=pa.string()),
            "follower_id": pa.array(ids[:-1], type=pa.string()),
            "followed_id": pa.array(ids[1:], type=pa.string()),
            "collected_at": pa.array([NOW] * 19, type=pa.timestamp("us", tz="UTC")),
        }
    )
    con = duckdb.connect()
    con.register("authors_tbl", authors)
    con.register("follows_tbl", follows)
    fresh = {i: CrawlEntry(handle=f"h{i}", crawled_at=NOW.isoformat()) for i in ids[:15]}

    assert sorted(u for u, _ in discover_from_edges(
        con, "follows_tbl", "authors_tbl", fresh, refresh_days=30
    )) == sorted(ids[15:])
    assert len(discover_from_edges(con, "follows_tbl", "authors_tbl", {}, 30, limit=7)) == 7


def test_discovery_takes_the_newest_non_empty_handle():
    """The rule the old `QUALIFY` directory applied, kept under `arg_max`: an
    empty handle on the newest snapshot does not erase a real one."""
    older, newer = NOW - timedelta(days=2), NOW
    authors = pa.table(
        {
            "platform": pa.array(["x"] * 3, type=pa.string()),
            "platform_user_id": pa.array(["1", "1", "1"], type=pa.string()),
            "handle": pa.array(["old_name", "new_name", ""], type=pa.string()),
            "collected_at": pa.array(
                [older, newer - timedelta(hours=1), newer], type=pa.timestamp("us", tz="UTC")
            ),
        }
    )
    follows = pa.table(
        {
            "platform": pa.array(["x"], type=pa.string()),
            "follower_id": pa.array(["1"], type=pa.string()),
            "followed_id": pa.array([None], type=pa.string()),
            "collected_at": pa.array([NOW], type=pa.timestamp("us", tz="UTC")),
        }
    )
    con = duckdb.connect()
    con.register("authors_tbl", authors)
    con.register("follows_tbl", follows)

    assert discover_from_edges(con, "follows_tbl", "authors_tbl", {}, 30) == [("1", "new_name")]


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

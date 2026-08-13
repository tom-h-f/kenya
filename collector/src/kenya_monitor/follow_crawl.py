"""Recursive follower/following crawler with crawl-state tracking.

Walks the social graph BFS-style: crawl an account's followers + following,
enqueue newly discovered accounts, skip accounts crawled within the refresh
window. State lives in ``state/follow_crawl.json`` (resumed across runs).
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

from kenya_monitor.collectors.base import Collector, FollowEdge
from kenya_monitor.config import (
    FOLLOW_CRAWL_MAX_ATTEMPTS,
    FOLLOW_CRAWL_REFRESH_DAYS,
    FOLLOW_CRAWL_STATE_PATH,
)
from kenya_monitor.storage import Storage

log = logging.getLogger("kenya_monitor")


@dataclass
class CrawlEntry:
    handle: str
    crawled_at: str
    edge_count: int = 0
    status: str = "ok"  # ok | failed | not_found
    attempts: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_crawl_state(path: Path = FOLLOW_CRAWL_STATE_PATH) -> dict[str, CrawlEntry]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {
        uid: CrawlEntry(**entry)
        for uid, entry in (raw.get("entries") or {}).items()
    }


def save_crawl_state(
    entries: dict[str, CrawlEntry],
    path: Path = FOLLOW_CRAWL_STATE_PATH,
) -> None:
    """Write via a temp file + rename, so an interrupted run leaves the previous
    ledger intact rather than a truncated one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            {"updated_at": _now_iso(), "entries": {k: asdict(v) for k, v in entries.items()}},
            indent=2,
        )
    )
    os.replace(tmp, path)


def is_due(
    entry: CrawlEntry | None,
    refresh_days: int,
    now: datetime | None = None,
    max_attempts: int = FOLLOW_CRAWL_MAX_ATTEMPTS,
) -> bool:
    """Never crawled -> due. Repeatedly failing or unresolvable -> give up rather
    than re-attempt every run forever.

    A failure retries once the refresh window has passed, like any other entry,
    rather than immediately: an account that errored is not more urgent than one
    that succeeded, and treating it as due-now is what let unresolvable handles
    occupy the head of the queue on every pass.
    """
    if entry is None:
        return True
    if entry.status in ("failed", "not_found") and entry.attempts >= max_attempts:
        return False
    now = now or datetime.now(timezone.utc)
    crawled = datetime.fromisoformat(entry.crawled_at)
    return crawled < now - timedelta(days=refresh_days)


def crawl_summary(entries: dict[str, CrawlEntry]) -> dict[str, int | str | None]:
    if not entries:
        return {
            "tracked": 0,
            "ok": 0,
            "failed": 0,
            "not_found": 0,
            "latest_crawl": None,
            "earliest_crawl": None,
        }
    times = [datetime.fromisoformat(e.crawled_at) for e in entries.values()]
    statuses = [e.status for e in entries.values()]
    return {
        "tracked": len(entries),
        "ok": sum(s == "ok" for s in statuses),
        "failed": sum(s == "failed" for s in statuses),
        "not_found": sum(s == "not_found" for s in statuses),
        "latest_crawl": max(times).isoformat(),
        "earliest_crawl": min(times).isoformat(),
    }


def _author_directory(
    con: duckdb.DuckDBPyConnection, authors_view: str
) -> dict[str, str]:
    """platform_user_id -> handle (latest snapshot)."""
    try:
        rows = con.sql(
            f"""
            SELECT platform_user_id, handle
            FROM {authors_view}
            WHERE handle IS NOT NULL AND trim(handle) != ''
            QUALIFY row_number() OVER (
                PARTITION BY platform_user_id ORDER BY collected_at DESC
            ) = 1
            """
        ).fetchall()
    except duckdb.Error:
        return {}
    return {r[0]: r[1] for r in rows}


def discover_from_edges(
    con: duckdb.DuckDBPyConnection,
    follows_view: str,
    authors_view: str,
    entries: dict[str, CrawlEntry],
    refresh_days: int,
) -> list[tuple[str, str]]:
    """Accounts appearing in follows/ with a known handle and due for crawl."""
    try:
        con.sql(f"SELECT 1 FROM {follows_view} LIMIT 1").fetchall()
    except duckdb.Error:
        return []
    directory = _author_directory(con, authors_view)
    rows = con.sql(
        f"""
        SELECT uid FROM (
            SELECT follower_id AS uid FROM {follows_view}
            UNION
            SELECT followed_id AS uid FROM {follows_view}
        )
        WHERE uid IS NOT NULL
        """
    ).fetchall()
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for (uid,) in rows:
        if uid in seen or uid not in directory:
            continue
        seen.add(uid)
        if is_due(entries.get(uid), refresh_days):
            out.append((uid, directory[uid]))
    return out


async def _resolve_uid(
    collector: Collector,
    con: duckdb.DuckDBPyConnection,
    authors_view: str,
    handle: str,
) -> str | None:
    # `params` is keyword-only on DuckDB's `sql()`; passing it positionally raises
    # TypeError, which escapes the caller's try block and kills the whole crawl.
    #
    # The window dedupes snapshots per user id, not per handle, so a handle that
    # has belonged to several ids still yields one row each. Order the outer
    # select so `LIMIT 1` takes the most recently seen id rather than an
    # arbitrary one.
    row = con.sql(
        f"""
        SELECT platform_user_id FROM {authors_view}
        WHERE lower(handle) = lower(?)
        QUALIFY row_number() OVER (
            PARTITION BY platform_user_id ORDER BY collected_at DESC
        ) = 1
        ORDER BY collected_at DESC
        LIMIT 1
        """,
        params=[handle],
    ).fetchone()
    if row:
        return str(row[0])
    api = getattr(collector, "api", None)
    if api is None:
        return None
    user = await api.user_by_login(handle)
    return str(user.id) if user else None


def _queue_candidates(
    seed_handles: list[str],
    discovered: list[tuple[str, str]],
    entries: dict[str, CrawlEntry],
    refresh_days: int,
) -> deque[tuple[str, str]]:
    """Deduped BFS queue of (uid, handle), seeds first."""
    queue: deque[tuple[str, str]] = deque()
    queued_uids: set[str] = set()
    queued_handles: set[str] = set()

    for handle in seed_handles:
        h = handle.lstrip("@").strip()
        key = h.lower()
        if not h or key in queued_handles:
            continue
        # Seeds arrive as handles with no id, so their only ledger record is the
        # `handle:` key written when resolution failed. Honour it, or a handle
        # that can never resolve is re-queued first on every run.
        if not is_due(entries.get(f"handle:{key}"), refresh_days):
            continue
        queue.append(("", h))
        queued_handles.add(key)

    for uid, handle in discovered:
        key = handle.lower()
        if uid in queued_uids or key in queued_handles:
            continue
        if is_due(entries.get(uid), refresh_days):
            queue.append((uid, handle))
            queued_uids.add(uid)
            queued_handles.add(key)

    return queue


async def crawl_follows(
    collector: Collector,
    storage: Storage,
    *,
    seed_handles: list[str] | None = None,
    limit: int,
    max_accounts: int,
    refresh_days: int = FOLLOW_CRAWL_REFRESH_DAYS,
    from_edges: bool = True,
    state_path: Path = FOLLOW_CRAWL_STATE_PATH,
) -> dict[str, int]:
    """BFS crawl of follower/following graphs. Returns run counters."""
    entries = load_crawl_state(state_path)
    authors_view = storage.authors_view(platform="x")
    follows_view = storage.follows_view(platform="x")

    discovered: list[tuple[str, str]] = []
    if from_edges:
        discovered = discover_from_edges(
            storage.con, follows_view, authors_view, entries, refresh_days
        )

    queue = _queue_candidates(seed_handles or [], discovered, entries, refresh_days)
    log.info(
        "follow crawl: queue=%d seeds=%d discovered=%d tracked=%d",
        len(queue),
        len(seed_handles or []),
        len(discovered),
        len(entries),
    )

    counts = {
        "crawled": 0,
        "skipped_fresh": 0,
        "skipped_no_handle": 0,
        "failed": 0,
        "not_found": 0,
        "follow_edges": 0,
        "authors": 0,
        "enqueued": 0,
    }
    seen_this_run: set[str] = set()

    # Built once. This is a full scan of the whole `authors/` prefix, and it used
    # to sit inside the loop below, so a 50-account run re-read every author ever
    # collected 50 times for information the freshly-written snapshots already
    # carry. That scan was the dominant cost of the step.
    directory = _author_directory(storage.con, authors_view)

    while queue and counts["crawled"] < max_accounts:
        uid_hint, handle = queue.popleft()
        handle = handle.lstrip("@").strip()
        if not handle:
            counts["skipped_no_handle"] += 1
            continue

        uid = uid_hint or await _resolve_uid(collector, storage.con, authors_view, handle)
        if not uid:
            counts["not_found"] += 1
            log.warning("follow crawl: could not resolve @%s", handle)
            # Keyed by handle, because the whole problem is that no id exists for
            # it. Without this the ledger never learns, and `is_due` hands the
            # same unresolvable handle back on every run.
            prior = entries.get(f"handle:{handle.lower()}")
            entries[f"handle:{handle.lower()}"] = CrawlEntry(
                handle=handle,
                crawled_at=_now_iso(),
                status="not_found",
                attempts=(prior.attempts if prior else 0) + 1,
            )
            continue

        if uid in seen_this_run:
            continue
        seen_this_run.add(uid)

        if not is_due(entries.get(uid), refresh_days):
            counts["skipped_fresh"] += 1
            continue

        try:
            edges: list[FollowEdge] = [
                e async for e in collector.follows(handle, limit=limit)
            ]
            key = storage.write_follows(edges)
            counts["follow_edges"] += len(edges)
            if key:
                log.info("crawl @%s -> %d edges -> %s", handle, len(edges), key)

            authors = collector.collected_authors()
            akey = storage.write_authors(authors)
            counts["authors"] += len(authors)
            if akey:
                log.debug("crawl @%s -> %d author snapshots", handle, len(authors))

            entries[uid] = CrawlEntry(
                handle=handle,
                crawled_at=_now_iso(),
                edge_count=len(edges),
                status="ok",
            )
            counts["crawled"] += 1

            directory.update({a.platform_user_id: a.handle for a in authors})
            touched = {uid}
            for edge in edges:
                touched.add(edge.follower_id)
                touched.add(edge.followed_id)

            for new_uid in touched:
                if new_uid == uid or new_uid in seen_this_run:
                    continue
                new_handle = directory.get(new_uid)
                if not new_handle:
                    continue
                if is_due(entries.get(new_uid), refresh_days):
                    queue.append((new_uid, new_handle))
                    counts["enqueued"] += 1

        except Exception:
            log.exception("follow crawl failed for @%s", handle)
            prior = entries.get(uid)
            entries[uid] = CrawlEntry(
                handle=handle,
                crawled_at=_now_iso(),
                status="failed",
                attempts=(prior.attempts if prior else 0) + 1,
            )
            counts["failed"] += 1

        save_crawl_state(entries, state_path)

    # The `continue` paths above (unresolvable handle, already seen, still fresh)
    # skip the in-loop save, so a run that only ever hit those would otherwise
    # discard the not_found records it just learned.
    save_crawl_state(entries, state_path)

    counts["queue_remaining"] = len(queue)
    counts["tracked_total"] = len(entries)
    return counts


def pending_count(
    storage: Storage,
    refresh_days: int = FOLLOW_CRAWL_REFRESH_DAYS,
    state_path: Path = FOLLOW_CRAWL_STATE_PATH,
) -> int:
    entries = load_crawl_state(state_path)
    discovered = discover_from_edges(
        storage.con,
        storage.follows_view(platform="x"),
        storage.authors_view(platform="x"),
        entries,
        refresh_days,
    )
    return len(discovered)

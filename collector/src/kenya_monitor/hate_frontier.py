"""Expansion ledger for hate-network seeds: which accounts we already pulled.

Without this, `run_hate_expand_once` asks for the top-N seeds by score and gets
the *same* top-N every pass, so the whole expansion budget goes on re-pulling
incumbents and the frontier never moves outward.

Modelled on `follow_crawl.CrawlEntry` - a record of work done, keyed by
`platform_user_id` because handles change and ids don't. Deliberately NOT
`adaptive.DynamicEntry`, whose expiry means "stop believing this" and would
silently forget visited accounts; here expiry means "due for another look".

Four things `follow_crawl` gets wrong that are fixed here:

- **score order preserved** - it takes candidates already ranked by the caller,
  rather than whatever order the database returned
- **attempt cap** - `follow_crawl.is_due` retries failures forever with no backoff
- **pruning** - `follow_crawl`'s ledger grows without bound
- **atomic writes** - a crash mid-write truncates every other state file here
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kenya_monitor.config import (
    HATE_EXPAND_KEEP_DAYS,
    HATE_EXPAND_MAX_ATTEMPTS,
    HATE_EXPAND_REFRESH_DAYS,
    HATE_EXPAND_STATE_PATH,
)

log = logging.getLogger("kenya_monitor")


@dataclass
class ExpandEntry:
    handle: str
    expanded_at: str
    n_posts: int = 0
    status: str = "ok"  # ok | failed
    attempts: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path = HATE_EXPAND_STATE_PATH) -> dict[str, ExpandEntry]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning("hate frontier: unreadable state at %s; starting fresh", path)
        return {}
    return {k: ExpandEntry(**v) for k, v in (raw.get("entries") or {}).items()}


def save(entries: dict[str, ExpandEntry], path: Path = HATE_EXPAND_STATE_PATH) -> None:
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
    entry: ExpandEntry | None,
    refresh_days: int = HATE_EXPAND_REFRESH_DAYS,
    max_attempts: int = HATE_EXPAND_MAX_ATTEMPTS,
    now: datetime | None = None,
) -> bool:
    """Never expanded -> due. Repeatedly failing -> give up rather than burn the
    budget on it every pass. Otherwise due once the refresh window has passed."""
    if entry is None:
        return True
    if entry.status == "failed" and entry.attempts >= max_attempts:
        return False
    now = now or datetime.now(timezone.utc)
    return datetime.fromisoformat(entry.expanded_at) < now - timedelta(days=refresh_days)


def prune(
    entries: dict[str, ExpandEntry],
    keep_days: int = HATE_EXPAND_KEEP_DAYS,
    now: datetime | None = None,
) -> dict[str, ExpandEntry]:
    """Drop entries older than `keep_days`. An account not seen for that long is
    outside the scoring lookback anyway, so remembering it only grows the file."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=keep_days)
    return {
        uid: e
        for uid, e in entries.items()
        if datetime.fromisoformat(e.expanded_at) >= cutoff
    }


def select_frontier(
    candidates: list[dict],
    entries: dict[str, ExpandEntry],
    max_accounts: int,
    refresh_days: int = HATE_EXPAND_REFRESH_DAYS,
    max_attempts: int = HATE_EXPAND_MAX_ATTEMPTS,
    now: datetime | None = None,
) -> list[dict]:
    """The next batch to expand: candidates that are due, in the order given.

    `candidates` are seed dicts from `hate_signal.hate_accounts`, already ranked.
    Order is preserved - the caller's ranking is the priority, which is the whole
    point of a frontier over a FIFO queue."""
    out: list[dict] = []
    for c in candidates:
        uid = c.get("author_id")
        if not uid:
            continue
        if is_due(entries.get(uid), refresh_days, max_attempts, now):
            out.append(c)
        if len(out) >= max_accounts:
            break
    return out


def record(
    entries: dict[str, ExpandEntry],
    candidate: dict,
    n_posts: int = 0,
    status: str = "ok",
    now: datetime | None = None,
) -> dict[str, ExpandEntry]:
    """Mark one account expanded. Attempts accumulate across failures so
    `is_due` can eventually stop retrying a handle that never resolves."""
    uid = candidate["author_id"]
    prior = entries.get(uid)
    entries[uid] = ExpandEntry(
        handle=candidate.get("handle") or (prior.handle if prior else ""),
        expanded_at=(now or datetime.now(timezone.utc)).isoformat(),
        n_posts=n_posts,
        status=status,
        attempts=((prior.attempts if prior else 0) + 1) if status == "failed" else 0,
    )
    return entries


def summary(entries: dict[str, ExpandEntry]) -> dict[str, int]:
    return {
        "tracked": len(entries),
        "ok": sum(e.status == "ok" for e in entries.values()),
        "failed": sum(e.status == "failed" for e in entries.values()),
        "exhausted": sum(
            e.status == "failed" and e.attempts >= HATE_EXPAND_MAX_ATTEMPTS
            for e in entries.values()
        ),
    }

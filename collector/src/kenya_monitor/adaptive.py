"""Adaptive target promotion + burst detection (docs/collection/cib-collection.md).

Promoted targets live in a JSON state file merged with the curated
`targets.yaml` at run time - the static file is never written back. Entries
carry their source and expire after `DYNAMIC_EXPIRY_DAYS` without
re-confirmation. Caps keep a runaway promotion from eating the rate budget.

SQL runs against view expressions (`storage.posts_view()` etc.) so tests can
substitute local temp tables for R2 globs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

from kenya_monitor.config import (
    BURST_MIN_POSTS,
    BURST_ZSCORE,
    DYNAMIC_EXPIRY_DAYS,
    DYNAMIC_HASHTAG_MIN_COUNT,
    DYNAMIC_HASHTAG_RATIO,
    DYNAMIC_MAX_ACCOUNTS,
    DYNAMIC_MAX_KEYWORDS,
    DYNAMIC_TARGETS_PATH,
    PlatformTargets,
)

log = logging.getLogger("kenya_monitor")


@dataclass
class DynamicEntry:
    value: str
    kind: str  # "keyword" | "account"
    source: str  # "hashtag-burst" | "coordination-cluster"
    added_at: str  # ISO timestamps (JSON-friendly)
    last_confirmed: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(path: Path = DYNAMIC_TARGETS_PATH) -> list[DynamicEntry]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [DynamicEntry(**e) for e in raw.get("entries", [])]


def save_state(entries: list[DynamicEntry], path: Path = DYNAMIC_TARGETS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"updated_at": _now_iso(), "entries": [e.__dict__ for e in entries]}, indent=2)
    )


def bursting_hashtags(
    con: duckdb.DuckDBPyConnection,
    posts_view: str,
    min_count: int = DYNAMIC_HASHTAG_MIN_COUNT,
    ratio: float = DYNAMIC_HASHTAG_RATIO,
) -> list[str]:
    """Hashtags whose last-24h volume is `ratio` x their prior-7d daily average
    (new tags: any volume >= min_count). Case-folded, '#'-prefixed."""
    rows = con.sql(
        f"""
        WITH lp AS (
            SELECT * FROM {posts_view}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
            ) = 1
        ), tags AS (
            SELECT lower(unnest(hashtags)) AS tag, created_at FROM lp
            WHERE len(hashtags) > 0
        ), recent AS (
            SELECT tag, count(*) AS n24 FROM tags
            WHERE created_at > now() - INTERVAL 1 DAY GROUP BY 1
        ), baseline AS (
            SELECT tag, count(*) / 7.0 AS daily FROM tags
            WHERE created_at <= now() - INTERVAL 1 DAY
              AND created_at > now() - INTERVAL 8 DAY
            GROUP BY 1
        )
        SELECT r.tag
        FROM recent r LEFT JOIN baseline b USING (tag)
        WHERE r.n24 >= {int(min_count)}
          AND (b.daily IS NULL OR r.n24 >= {float(ratio)} * b.daily)
        ORDER BY r.n24 DESC
        """
    ).fetchall()
    return [f"#{r[0]}" for r in rows]


def cluster_accounts(
    con: duckdb.DuckDBPyConnection, clusters_view: str, authors_view: str
) -> list[str]:
    """Handles of members of the most recent persisted coordination clusters."""
    try:
        rows = con.sql(
            f"""
            WITH latest_run AS (
                SELECT * FROM {clusters_view}
                QUALIFY dense_rank() OVER (ORDER BY computed_at DESC) = 1
            ), la AS (
                SELECT * FROM {authors_view}
                QUALIFY row_number() OVER (
                    PARTITION BY platform, platform_user_id ORDER BY collected_at DESC
                ) = 1
            )
            SELECT DISTINCT la.handle
            FROM latest_run c JOIN la ON c.author_id = la.platform_user_id
            WHERE la.handle IS NOT NULL
            """
        ).fetchall()
    except duckdb.Error:
        return []  # no clusters persisted yet
    return [r[0] for r in rows]


def refresh_entries(
    existing: list[DynamicEntry],
    keywords: list[str],
    accounts: list[str],
    max_keywords: int = DYNAMIC_MAX_KEYWORDS,
    max_accounts: int = DYNAMIC_MAX_ACCOUNTS,
    expiry_days: int = DYNAMIC_EXPIRY_DAYS,
    now: datetime | None = None,
    sources: dict[str, str] | None = None,
) -> list[DynamicEntry]:
    """Merge fresh promotions into the state: confirm still-active entries,
    add new ones (newest first within caps), drop expired ones."""
    now = now or datetime.now(timezone.utc)
    now_iso = now.isoformat()
    cutoff = now - timedelta(days=expiry_days)
    sources = sources or {}
    by_key = {(e.kind, e.value.lower()): e for e in existing}

    for kind, values in (("keyword", keywords), ("account", accounts)):
        for v in values:
            key = (kind, v.lower())
            if key in by_key:
                by_key[key].last_confirmed = now_iso
            else:
                by_key[key] = DynamicEntry(
                    value=v,
                    kind=kind,
                    source=sources.get(v, "hashtag-burst" if kind == "keyword" else "coordination-cluster"),
                    added_at=now_iso,
                    last_confirmed=now_iso,
                )

    alive = [
        e for e in by_key.values() if datetime.fromisoformat(e.last_confirmed) > cutoff
    ]
    out: list[DynamicEntry] = []
    for kind, cap in (("keyword", max_keywords), ("account", max_accounts)):
        pool = sorted(
            (e for e in alive if e.kind == kind),
            key=lambda e: e.last_confirmed,
            reverse=True,
        )
        out.extend(pool[:cap])
    return out


def merge_targets(static: PlatformTargets, entries: list[DynamicEntry]) -> PlatformTargets:
    """Static targets + live dynamic entries, deduped case-insensitively."""
    keywords = list(static.keywords)
    accounts = list(static.accounts)
    seen_kw = {k.lower() for k in keywords}
    seen_acc = {a.lower() for a in accounts}
    for e in entries:
        if e.kind == "keyword" and e.value.lower() not in seen_kw:
            keywords.append(e.value)
        elif e.kind == "account" and e.value.lower() not in seen_acc:
            accounts.append(e.value)
    return PlatformTargets(accounts=accounts, keywords=keywords)


def detect_burst(
    con: duckdb.DuckDBPyConnection,
    posts_view: str,
    zscore: float = BURST_ZSCORE,
    min_posts: int = BURST_MIN_POSTS,
) -> tuple[bool, float, int]:
    """Is the last complete hour's post volume a burst vs the prior 48h?
    Returns (bursting, z, posts_last_hour)."""
    rows = con.sql(
        f"""
        WITH lp AS (
            SELECT * FROM {posts_view}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
            ) = 1
        )
        SELECT count(*) AS n
        FROM lp
        WHERE created_at > now() - INTERVAL 49 HOUR
          AND created_at < date_trunc('hour', now())
        GROUP BY date_trunc('hour', created_at)
        ORDER BY date_trunc('hour', created_at)
        """
    ).fetchall()
    if len(rows) < 12:  # not enough signal for a baseline
        return False, 0.0, 0
    counts = [n for (n,) in rows]
    latest = counts[-1]
    base = counts[:-1]
    mean = sum(base) / len(base)
    var = sum((c - mean) ** 2 for c in base) / max(len(base) - 1, 1)
    std = var**0.5
    if std > 0:
        z = (latest - mean) / std
    else:  # flat baseline: any excess is infinitely surprising, none is not
        z = float("inf") if latest > mean else 0.0
    return (z >= zscore and latest >= min_posts), z, latest


def promote(
    con: duckdb.DuckDBPyConnection,
    posts_view: str,
    clusters_view: str,
    authors_view: str,
    state_path: Path = DYNAMIC_TARGETS_PATH,
    dry_run: bool = False,
) -> list[DynamicEntry]:
    """One promotion pass: compute candidates, refresh the state file, return
    the live entries. `dry_run` computes without saving."""
    keywords = bursting_hashtags(con, posts_view)
    accounts = cluster_accounts(con, clusters_view, authors_view)
    entries = refresh_entries(load_state(state_path), keywords, accounts)
    if not dry_run:
        save_state(entries, state_path)
    for e in entries:
        log.info("dynamic target: %s %r (%s, confirmed %s)", e.kind, e.value, e.source, e.last_confirmed)
    return entries

from __future__ import annotations

import asyncio
import logging
import os
import random
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger

from kenya_monitor import adaptive
from kenya_monitor.accounts import (
    active_count,
    metrics_cap,
    posts_gap_hours,
    sync_accounts,
)
from kenya_monitor.collectors.x import MAX_AGE_DAYS, backfill_windows, build_api, recent_windows
from kenya_monitor.config import (
    ACCOUNT_SYNC_HOURS,
    FOLLOW_FETCH_LIMIT,
    FOLLOW_MAX_ACCOUNTS,
    METRICS_MAX_POSTS_FLOOR,
    METRICS_MAX_POSTS_PER_ACCOUNT,
    POSTS_MAX_GAP_HOURS,
    POSTS_MIN_GAP_HOURS,
    SEARCH_BACKFILL_WINDOW_DAYS,
    SEARCH_MIN_FAVES,
    SEARCH_RECENT_DAYS,
    SEARCH_WINDOW_LIMIT,
    PlatformTargets,
    R2Config,
    load_accounts,
    load_targets,
)
from kenya_monitor.runner import (
    build_x_collector,
    collect_backfill,
    collect_follows,
    collect_metrics,
    collect_snowball,
    collect_x,
)
from kenya_monitor.storage import Storage

log = logging.getLogger("kenya_monitor")

# Burst check cadence: cheap R2 aggregate, so every 30min is fine. One trigger
# per hour bucket at most - a burst should cause one extra sweep, not a loop.
BURST_CHECK_MINUTES = 30

# Metrics pass defaults (cap computed from pool size at runtime).
METRICS_SINCE_DAYS = 5
METRICS_TOP_PCT = 0.05


def _adaptive_targets(storage: Storage, dry_run: bool = False) -> PlatformTargets:
    """Static targets merged with the capped dynamic promotions (hashtag bursts
    + coordination-cluster members). Never edits targets.yaml."""
    static = load_targets().get("x", PlatformTargets())
    try:
        entries = adaptive.promote(
            storage.con,
            storage.posts_view(platform="x"),
            storage.clusters_view(platform="x"),
            storage.authors_view(platform="x"),
            dry_run=dry_run,
        )
    except Exception:
        log.exception("adaptive promotion failed; using static targets")
        return static
    return adaptive.merge_targets(static, entries)


async def run_once(
    limit: int, include_backfill: bool = False, keywords: bool = True, accounts: bool = True
) -> dict[str, int]:
    """One post-collection pass: recent day-windows, plus the older backfill windows when
    `include_backfill` is set (scheduled once per day). Targets include the live
    adaptive promotions."""
    storage = Storage(R2Config.from_env())
    collector = await build_x_collector(load_accounts())
    targets = _adaptive_targets(storage)
    windows = recent_windows(SEARCH_RECENT_DAYS)
    if include_backfill:
        windows += backfill_windows(SEARCH_RECENT_DAYS, SEARCH_BACKFILL_WINDOW_DAYS)
    counts = await collect_x(
        collector,
        storage,
        targets,
        search_windows=windows,
        min_faves=SEARCH_MIN_FAVES,
        window_limit=SEARCH_WINDOW_LIMIT,
        timeline_limit=limit,
        keywords=keywords,
        accounts=accounts,
    )
    log.info("posts run complete (backfill=%s, %d windows): %s", include_backfill, len(windows), counts)
    return counts


async def run_snowball_once(**overrides) -> dict[str, int]:
    """One snowball pass over hot objects (see runner.collect_snowball)."""
    storage = Storage(R2Config.from_env())
    collector = await build_x_collector(load_accounts())
    counts = await collect_snowball(collector, storage, **overrides)
    log.info("snowball run complete: %s", counts)
    return counts


async def run_follows_once(
    handles: list[str] | None = None,
    limit: int = FOLLOW_FETCH_LIMIT,
    max_accounts: int = FOLLOW_MAX_ACCOUNTS,
) -> dict[str, int]:
    """Follower/following edges for flagged-cluster members (or explicit handles)."""
    storage = Storage(R2Config.from_env())
    if handles is None:
        handles = adaptive.cluster_accounts(
            storage.con, storage.clusters_view(platform="x"), storage.authors_view(platform="x")
        )
    handles = handles[:max_accounts]
    if not handles:
        log.info("follows: no flagged accounts to fetch")
        return {"follow_edges": 0}
    collector = await build_x_collector(load_accounts())
    counts = await collect_follows(collector, storage, handles, limit)
    log.info("follows run complete: %s", counts)
    return counts


async def run_metrics_once(
    since_days: int = METRICS_SINCE_DAYS,
    top_pct: float = METRICS_TOP_PCT,
    max_posts: int | None = None,
) -> dict[str, int]:
    """One metrics-refresh pass over the top-engagement posts of the recent window."""
    storage = Storage(R2Config.from_env())
    api = build_api()
    await sync_accounts(api, load_accounts())
    n_active = await active_count(api.pool)
    cap = max_posts or metrics_cap(n_active, METRICS_MAX_POSTS_PER_ACCOUNT, METRICS_MAX_POSTS_FLOOR)
    collector = await build_x_collector(load_accounts(), api=api)
    counts = await collect_metrics(
        collector, storage, since_days=since_days, top_pct=top_pct, max_posts=cap
    )
    log.info("metrics run complete (cap=%d, active=%d): %s", cap, n_active, counts)
    return counts


async def run_backfill_once(
    days: int = MAX_AGE_DAYS, window_limit: int = SEARCH_WINDOW_LIMIT
) -> dict[str, int]:
    """One-time deep backfill: daily windows across the last `days` days, to even out the
    historical (pre-fix) temporal coverage."""
    storage = Storage(R2Config.from_env())
    collector = await build_x_collector(load_accounts())
    targets = load_targets().get("x", PlatformTargets())
    windows = recent_windows(days)  # daily granularity across the whole window
    counts = await collect_backfill(
        collector, storage, targets, windows, min_faves=SEARCH_MIN_FAVES, window_limit=window_limit
    )
    log.info("backfill complete: %s", counts)
    return counts


def _gap_seconds(lo: float, hi: float) -> float:
    return random.uniform(lo * 3600, hi * 3600)


async def run_scheduler(limit: int) -> None:
    """Continual collection: posts every 1.5-5h (scales with pool size), metrics near
    each gap midpoint, pool maintenance on a timer. twscrape rotates accounts per
    request; only one posts pass runs at a time."""
    posts_lock = asyncio.Lock()
    last_backfill: dict[str, object] = {"date": None}
    scheduler = AsyncIOScheduler(timezone=timezone.utc)
    scheduler.start()

    async def _pool_size() -> int:
        api = build_api()
        return await active_count(api.pool)

    async def maintain_accounts() -> None:
        try:
            api = build_api()
            await sync_accounts(api, load_accounts(), relogin_failed=True)
            await api.pool.reset_locks()
        except Exception:
            log.exception("account maintenance failed")
        finally:
            scheduler.add_job(
                maintain_accounts,
                DateTrigger(run_date=datetime.now(timezone.utc) + timedelta(hours=ACCOUNT_SYNC_HOURS)),
                id="accounts", replace_existing=True,
            )

    async def metrics_job() -> None:
        try:
            await run_metrics_once()
        except Exception:
            log.exception("metrics run failed")

    async def posts_job() -> None:
        try:
            async with posts_lock:
                today = datetime.now(timezone.utc).date()
                do_backfill = last_backfill["date"] != today
                await run_once(limit, include_backfill=do_backfill)
                if do_backfill:
                    last_backfill["date"] = today
                await run_snowball_once()
        except Exception:
            log.exception("posts run failed")
        finally:
            now = datetime.now(timezone.utc)
            n_active = await _pool_size()
            lo, hi = posts_gap_hours(n_active, POSTS_MIN_GAP_HOURS, POSTS_MAX_GAP_HOURS)
            gap = _gap_seconds(lo, hi)
            midpoint = gap * random.uniform(0.4, 0.6)
            scheduler.add_job(
                posts_job, DateTrigger(run_date=now + timedelta(seconds=gap)),
                id="posts", replace_existing=True,
            )
            scheduler.add_job(
                metrics_job, DateTrigger(run_date=now + timedelta(seconds=midpoint)),
                id="metrics", replace_existing=True,
            )
            log.info(
                "next posts in %.1fh, metrics in %.1fh (%d active accounts)",
                gap / 3600, midpoint / 3600, n_active,
            )

    last_burst: dict[str, object] = {"hour": None}

    async def burst_job() -> None:
        try:
            if posts_lock.locked():
                return
            storage = Storage(R2Config.from_env())
            bursting, z, n = adaptive.detect_burst(storage.con, storage.posts_view(platform="x"))
            hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            if bursting and last_burst["hour"] != hour:
                last_burst["hour"] = hour
                log.info("burst detected (z=%.1f, %d posts/h): immediate sweep + snowball", z, n)
                async with posts_lock:
                    await run_once(limit, include_backfill=False)
                    await run_snowball_once()
        except Exception:
            log.exception("burst check failed")
        finally:
            scheduler.add_job(
                burst_job,
                DateTrigger(run_date=datetime.now(timezone.utc) + timedelta(minutes=BURST_CHECK_MINUTES)),
                id="burst", replace_existing=True,
            )

    scheduler.add_job(posts_job, DateTrigger(run_date=datetime.now(timezone.utc)), id="posts")
    scheduler.add_job(
        burst_job,
        DateTrigger(run_date=datetime.now(timezone.utc) + timedelta(minutes=BURST_CHECK_MINUTES)),
        id="burst",
    )
    scheduler.add_job(
        maintain_accounts,
        DateTrigger(run_date=datetime.now(timezone.utc) + timedelta(minutes=5)),
        id="accounts",
    )
    await asyncio.Event().wait()

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger

from kenya_monitor.collectors.x import MAX_AGE_DAYS, backfill_windows, recent_windows
from kenya_monitor.config import (
    SEARCH_BACKFILL_WINDOW_DAYS,
    SEARCH_MIN_FAVES,
    SEARCH_RECENT_DAYS,
    SEARCH_WINDOW_LIMIT,
    PlatformTargets,
    R2Config,
    load_accounts,
    load_targets,
)
from kenya_monitor.runner import build_x_collector, collect_backfill, collect_metrics, collect_x
from kenya_monitor.storage import Storage

log = logging.getLogger("kenya_monitor")

# Posts: ~6 full collections/day (randomized 3-5h gaps).
POSTS_MIN_GAP_HOURS = 3
POSTS_MAX_GAP_HOURS = 5
# Metrics: one pass per posts-gap (so also ~6x/day), placed near the midpoint of the gap
# to stay clear of post collection. Refreshes the top 5% (by likes+quotes+reposts) of the
# last 5 days; max_posts caps 1-account rate-limit load.
METRICS_SINCE_DAYS = 5
METRICS_TOP_PCT = 0.05
METRICS_MAX_POSTS = 200


async def run_once(
    limit: int, include_backfill: bool = False, keywords: bool = True, accounts: bool = True
) -> dict[str, int]:
    """One post-collection pass: recent day-windows, plus the older backfill windows when
    `include_backfill` is set (scheduled once per day)."""
    storage = Storage(R2Config.from_env())
    collector = await build_x_collector(load_accounts())
    targets = load_targets().get("x", PlatformTargets())
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


async def run_metrics_once(
    since_days: int = METRICS_SINCE_DAYS,
    top_pct: float = METRICS_TOP_PCT,
    max_posts: int = METRICS_MAX_POSTS,
) -> dict[str, int]:
    """One metrics-refresh pass over the top-engagement posts of the recent window."""
    storage = Storage(R2Config.from_env())
    collector = await build_x_collector(load_accounts())
    counts = await collect_metrics(
        collector, storage, since_days=since_days, top_pct=top_pct, max_posts=max_posts
    )
    log.info("metrics run complete: %s", counts)
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
    """Post collection every 3-5h; one metrics pass placed near the midpoint of each gap
    (so also ~6x/day, kept clear of post collection). A shared lock is the backstop so the
    single account is never scraped concurrently.
    """
    account_lock = asyncio.Lock()
    last_backfill: dict[str, object] = {"date": None}  # once-per-UTC-day backfill marker
    scheduler = AsyncIOScheduler(timezone=timezone.utc)
    scheduler.start()

    async def metrics_job() -> None:
        try:
            async with account_lock:
                await run_metrics_once()
        except Exception:
            log.exception("metrics run failed")

    async def posts_job() -> None:
        try:
            async with account_lock:
                today = datetime.now(timezone.utc).date()
                do_backfill = last_backfill["date"] != today
                await run_once(limit, include_backfill=do_backfill)
                if do_backfill:
                    last_backfill["date"] = today
        except Exception:
            log.exception("posts run failed")
        finally:
            now = datetime.now(timezone.utc)
            gap = _gap_seconds(POSTS_MIN_GAP_HOURS, POSTS_MAX_GAP_HOURS)
            midpoint = gap * random.uniform(0.4, 0.6)  # near the middle of the gap
            scheduler.add_job(
                posts_job, DateTrigger(run_date=now + timedelta(seconds=gap)),
                id="posts", replace_existing=True,
            )
            scheduler.add_job(
                metrics_job, DateTrigger(run_date=now + timedelta(seconds=midpoint)),
                id="metrics", replace_existing=True,
            )
            log.info("next posts in %.1fh, metrics in %.1fh", gap / 3600, midpoint / 3600)

    scheduler.add_job(posts_job, DateTrigger(run_date=datetime.now(timezone.utc)), id="posts")
    await asyncio.Event().wait()

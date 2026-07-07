from __future__ import annotations

import logging

import os
from kenya_monitor.collectors.base import Collector, Post
from kenya_monitor.collectors.x import Window, XCollector, build_api, sync_accounts
from kenya_monitor.config import PlatformTargets, XAccount
from kenya_monitor.pacing import human_pause
from kenya_monitor.storage import Storage

log = logging.getLogger("kenya_monitor")


async def build_x_collector(accounts: list[XAccount]) -> Collector:
    backend = os.getenv("X_BACKEND", "twscrape").lower()
    if backend == "apify":
        from kenya_monitor.collectors.apify import ApifyXCollector
        token = os.getenv("APIFY_TOKEN") or os.getenv("APIFY_API_TOKEN")
        if not token:
            raise RuntimeError("APIFY_TOKEN environment variable is required for Apify backend")
        actor_id = os.getenv("APIFY_ACTOR_ID", "xquik/x-tweet-scraper")
        return ApifyXCollector(token, actor_id=actor_id)
    else:
        api = build_api()
        added = await sync_accounts(api, accounts)
        if added:
            log.info("added %d new account(s) to pool", added)
        return XCollector(api)


async def collect_x(
    collector: Collector,
    storage: Storage,
    targets: PlatformTargets,
    search_windows: list[Window],
    min_faves: int,
    window_limit: int,
    timeline_limit: int,
    keywords: bool = True,
    accounts: bool = True,
) -> dict[str, int]:
    counts: dict[str, int] = {}

    if keywords and targets.keywords and search_windows:
        search_posts: list[Post] = []
        req = 0
        for kw in targets.keywords:
            kw_posts: list[Post] = []
            for since, until in search_windows:
                if req:
                    await human_pause()
                req += 1
                kw_posts.extend(
                    [
                        p
                        async for p in collector.search(
                            kw, limit=window_limit, since=since, until=until, min_faves=min_faves
                        )
                    ]
                )
            log.info("search %r (%d windows) -> %d posts", kw, len(search_windows), len(kw_posts))
            search_posts.extend(kw_posts)
        key = storage.write_posts(search_posts, target_type="search")
        counts["search"] = len(search_posts)
        if key:
            log.info("wrote %d search posts -> %s", len(search_posts), key)

    if accounts and targets.accounts:
        timeline_posts: list[Post] = []
        for i, handle in enumerate(targets.accounts):
            if i:
                await human_pause()
            got = [p async for p in collector.timeline(handle, limit=timeline_limit)]
            log.info("timeline @%s -> %d posts", handle, len(got))
            timeline_posts.extend(got)
        key = storage.write_posts(timeline_posts, target_type="timeline")
        counts["timeline"] = len(timeline_posts)
        if key:
            log.info("wrote %d timeline posts -> %s", len(timeline_posts), key)

    authors = collector.collected_authors()
    key = storage.write_authors(authors)
    counts["authors"] = len(authors)
    if key:
        log.info("wrote %d authors -> %s", len(authors), key)

    return counts


async def collect_backfill(
    collector: Collector,
    storage: Storage,
    targets: PlatformTargets,
    windows: list[Window],
    min_faves: int,
    window_limit: int,
) -> dict[str, int]:
    """One-time deep sweep to even out historical coverage: for each keyword, pull engaged
    posts from every window, writing per keyword (resumable if interrupted)."""
    total = 0
    req = 0
    for kw in targets.keywords:
        kw_posts: list[Post] = []
        for since, until in windows:
            if req:
                await human_pause()
            req += 1
            kw_posts.extend(
                [
                    p
                    async for p in collector.search(
                        kw, limit=window_limit, since=since, until=until, min_faves=min_faves
                    )
                ]
            )
        if kw_posts:
            storage.write_posts(kw_posts, target_type="search")
        authors = collector.collected_authors()
        if authors:
            storage.write_authors(authors)
        total += len(kw_posts)
        log.info("backfill %r -> %d posts (running total %d)", kw, len(kw_posts), total)
    return {"backfill": total}


async def collect_metrics(
    collector: Collector,
    storage: Storage,
    since_days: int = 5,
    top_pct: float = 0.05,
    max_posts: int = 200,
) -> dict[str, int]:
    """Re-fetch engagement for the top `top_pct` of posts (by likes+quotes+reposts) from
    the last `since_days` days. Writes lightweight count snapshots to the metrics/ prefix.

    `max_posts` is a safety cap: refreshing is 1 request/post, so a single account can only
    sustain so many per pass before rate limiting (see README notes).
    """
    source = storage.posts_view(platform=collector.platform)
    threshold = 1.0 - top_pct
    rows = storage.query(
        f"""
        WITH latest AS (
            SELECT platform_post_id,
                   (like_count + quote_count + repost_count) AS engagement
            FROM {source}
            WHERE collected_at > now() - INTERVAL {since_days} DAY
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY collected_at DESC
            ) = 1
        )
        SELECT platform_post_id FROM latest
        WHERE engagement >= (SELECT quantile_cont(engagement, {threshold}) FROM latest)
        ORDER BY engagement DESC
        LIMIT {max_posts}
        """
    ).fetchall()
    ids = [r[0] for r in rows]
    if not ids:
        log.info("metrics: no candidates in last %dd", since_days)
        return {"metrics": 0}

    snapshots = [m async for m in collector.refresh_metrics(ids)]
    key = storage.write_metrics(snapshots)
    if key:
        log.info(
            "metrics: refreshed top %.0f%% (%d posts) of last %dd -> %s",
            top_pct * 100,
            len(snapshots),
            since_days,
            key,
        )
    return {"metrics": len(snapshots)}

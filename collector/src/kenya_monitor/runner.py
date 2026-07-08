from __future__ import annotations

import asyncio
import json
import logging

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

from twscrape import API

from kenya_monitor.collectors.base import Collector, Engagement, FollowEdge, Post
from kenya_monitor.collectors.x import Window, XCollector, build_api, sync_accounts
from kenya_monitor.config import (
    COLLECT_CONCURRENCY,
    SEARCH_INCLUDE_RETWEETS,
    SEARCH_PRODUCT,
    SNOWBALL_HYDRATE_LIMIT,
    SNOWBALL_LOOKBACK_DAYS,
    SNOWBALL_REFRESH_HOURS,
    SNOWBALL_REPLIES_LIMIT,
    SNOWBALL_RETWEETERS_LIMIT,
    SNOWBALL_STATE_PATH,
    SNOWBALL_TOP_CONVERSATIONS,
    SNOWBALL_TOP_RETWEETED,
    PlatformTargets,
    XAccount,
)
from kenya_monitor.storage import Storage

log = logging.getLogger("kenya_monitor")


async def build_x_collector(accounts: list[XAccount], api: API | None = None) -> Collector:
    backend = os.getenv("X_BACKEND", "twscrape").lower()
    if backend == "apify":
        from kenya_monitor.collectors.apify import ApifyXCollector
        token = os.getenv("APIFY_TOKEN") or os.getenv("APIFY_API_TOKEN")
        if not token:
            raise RuntimeError("APIFY_TOKEN environment variable is required for Apify backend")
        actor_id = os.getenv("APIFY_ACTOR_ID", "xquik/x-tweet-scraper")
        return ApifyXCollector(token, actor_id=actor_id)
    else:
        api = api or build_api()
        result = await sync_accounts(api, accounts)
        if result.added or result.updated:
            log.info(
                "account pool ready: %d active (%d added, %d updated)",
                result.active,
                result.added,
                result.updated,
            )
        return XCollector(api)


async def _search_keyword(
    collector: Collector,
    kw: str,
    search_windows: list[Window],
    min_faves: int,
    window_limit: int,
) -> list[Post]:
    kw_posts: list[Post] = []
    for since, until in search_windows:
        kw_posts.extend(
            [
                p
                async for p in collector.search(
                    kw,
                    limit=window_limit,
                    since=since,
                    until=until,
                    min_faves=min_faves,
                    product=SEARCH_PRODUCT,
                    include_retweets=SEARCH_INCLUDE_RETWEETS,
                )
            ]
        )
    log.info("search %r (%d windows) -> %d posts", kw, len(search_windows), len(kw_posts))
    return kw_posts


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
        concurrency = max(1, COLLECT_CONCURRENCY)
        if concurrency > 1 and len(targets.keywords) > 1:
            sem = asyncio.Semaphore(concurrency)

            async def bounded(kw: str) -> list[Post]:
                async with sem:
                    return await _search_keyword(
                        collector, kw, search_windows, min_faves, window_limit
                    )

            chunks = await asyncio.gather(*[bounded(kw) for kw in targets.keywords])
            for chunk in chunks:
                search_posts.extend(chunk)
        else:
            for kw in targets.keywords:
                search_posts.extend(
                    await _search_keyword(
                        collector, kw, search_windows, min_faves, window_limit
                    )
                )
        key = storage.write_posts(search_posts, target_type="search")
        counts["search"] = len(search_posts)
        if key:
            log.info("wrote %d search posts -> %s", len(search_posts), key)

    if accounts and targets.accounts:
        timeline_posts: list[Post] = []
        for handle in targets.accounts:
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
    for kw in targets.keywords:
        kw_posts: list[Post] = []
        for since, until in windows:
            kw_posts.extend(
                [
                    p
                    async for p in collector.search(
                        kw,
                        limit=window_limit,
                        since=since,
                        until=until,
                        min_faves=min_faves,
                        product=SEARCH_PRODUCT,
                        include_retweets=SEARCH_INCLUDE_RETWEETS,
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


def hot_objects(
    con: duckdb.DuckDBPyConnection,
    posts_view: str,
    lookback_days: int = SNOWBALL_LOOKBACK_DAYS,
    top_retweeted: int = SNOWBALL_TOP_RETWEETED,
    top_conversations: int = SNOWBALL_TOP_CONVERSATIONS,
    hydrate_limit: int = SNOWBALL_HYDRATE_LIMIT,
) -> tuple[list[str], list[str], list[str]]:
    """Snowball targets from already-collected posts: the most-amplified
    objects, the busiest conversations, and referenced ids we never fetched.
    Returns (retweeted_ids, conversation_ids, missing_ids)."""
    base = f"""
        WITH lp AS (
            SELECT * FROM {posts_view}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
            ) = 1
        ), recent AS (
            SELECT * FROM lp WHERE created_at > now() - INTERVAL {int(lookback_days)} DAY
        )
    """
    retweeted = [
        r[0]
        for r in con.sql(
            base
            + f"""
            SELECT repost_of_id FROM recent
            WHERE repost_of_id IS NOT NULL
            GROUP BY 1
            ORDER BY max(repost_count) DESC, count(*) DESC
            LIMIT {int(top_retweeted)}
            """
        ).fetchall()
    ]
    conversations = [
        r[0]
        for r in con.sql(
            base
            + f"""
            SELECT conversation_id FROM recent
            WHERE conversation_id IS NOT NULL
            GROUP BY 1
            ORDER BY max(reply_count) DESC, count(*) DESC
            LIMIT {int(top_conversations)}
            """
        ).fetchall()
    ]
    missing = [
        r[0]
        for r in con.sql(
            base
            + f"""
            , refs AS (
                SELECT repost_of_id AS ref, max(repost_count) AS eng FROM recent
                WHERE repost_of_id IS NOT NULL GROUP BY 1
                UNION ALL
                SELECT quoted_post_id, max(quote_count) FROM recent
                WHERE quoted_post_id IS NOT NULL GROUP BY 1
                UNION ALL
                SELECT in_reply_to_id, max(reply_count) FROM recent
                WHERE in_reply_to_id IS NOT NULL GROUP BY 1
            )
            SELECT ref FROM refs
            WHERE ref NOT IN (SELECT platform_post_id FROM lp)
            GROUP BY ref
            ORDER BY max(eng) DESC
            LIMIT {int(hydrate_limit)}
            """
        ).fetchall()
    ]
    return retweeted, conversations, missing


def _load_snowball_state(path: Path = SNOWBALL_STATE_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_snowball_state(state: dict[str, str], path: Path = SNOWBALL_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def _due(object_ids: list[str], state: dict[str, str], refresh_hours: int) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=refresh_hours)
    return [
        oid
        for oid in object_ids
        if oid not in state or datetime.fromisoformat(state[oid]) < cutoff
    ]


async def collect_snowball(
    collector: Collector,
    storage: Storage,
    lookback_days: int = SNOWBALL_LOOKBACK_DAYS,
    top_retweeted: int = SNOWBALL_TOP_RETWEETED,
    top_conversations: int = SNOWBALL_TOP_CONVERSATIONS,
    retweeters_limit: int = SNOWBALL_RETWEETERS_LIMIT,
    replies_limit: int = SNOWBALL_REPLIES_LIMIT,
    hydrate_limit: int = SNOWBALL_HYDRATE_LIMIT,
    refresh_hours: int = SNOWBALL_REFRESH_HOURS,
    state_path: Path = SNOWBALL_STATE_PATH,
) -> dict[str, int]:
    """Census pass over hot objects: full retweeter lists (-> engagements/),
    full reply threads (-> posts/type=replies), and hydration of referenced
    originals (-> posts/type=hydrated). Per-object TTL in `state_path` stops
    hot objects being re-fetched every pass."""
    counts: dict[str, int] = {}
    retweeted, conversations, missing = hot_objects(
        storage.con,
        storage.posts_view(platform=collector.platform),
        lookback_days,
        top_retweeted,
        top_conversations,
        hydrate_limit,
    )
    state = _load_snowball_state(state_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    due_rt = _due(retweeted, state, refresh_hours)
    engagements: list[Engagement] = []
    for oid in due_rt:
        got = [e async for e in collector.retweeters(oid, limit=retweeters_limit)]
        log.info("retweeters of %s -> %d", oid, len(got))
        engagements.extend(got)
        state[oid] = now_iso
    key = storage.write_engagements(engagements)
    counts["retweeters"] = len(engagements)
    if key:
        log.info("wrote %d engagement rows -> %s", len(engagements), key)

    due_conv = _due(conversations, state, refresh_hours)
    reply_posts: list[Post] = []
    for cid in due_conv:
        got = [p async for p in collector.replies(cid, limit=replies_limit)]
        log.info("replies under %s -> %d", cid, len(got))
        reply_posts.extend(got)
        state[f"conv:{cid}"] = now_iso
    key = storage.write_posts(reply_posts, target_type="replies")
    counts["replies"] = len(reply_posts)
    if key:
        log.info("wrote %d reply posts -> %s", len(reply_posts), key)

    hydrated = [p async for p in collector.hydrate(missing)]
    key = storage.write_posts(hydrated, target_type="hydrated")
    counts["hydrated"] = len(hydrated)
    if key:
        log.info("hydrated %d referenced posts -> %s", len(hydrated), key)

    authors = collector.collected_authors()
    key = storage.write_authors(authors)
    counts["authors"] = len(authors)
    if key:
        log.info("wrote %d authors -> %s", len(authors), key)

    # prune state entries older than 2x the TTL so the file stays small
    prune = datetime.now(timezone.utc) - timedelta(hours=2 * refresh_hours)
    state = {k: v for k, v in state.items() if datetime.fromisoformat(v) >= prune}
    _save_snowball_state(state, state_path)
    return counts


async def collect_follows(
    collector: Collector,
    storage: Storage,
    handles: list[str],
    limit: int,
) -> dict[str, int]:
    """Follower/following edges for flagged accounts only (never the general
    population) -> follows/ prefix."""
    edges: list[FollowEdge] = []
    for handle in handles:
        got = [e async for e in collector.follows(handle, limit=limit)]
        log.info("follows @%s -> %d edges", handle, len(got))
        edges.extend(got)
    key = storage.write_follows(edges)
    counts = {"follow_edges": len(edges)}
    if key:
        log.info("wrote %d follow edges -> %s", len(edges), key)
    authors = collector.collected_authors()
    key = storage.write_authors(authors)
    counts["authors"] = len(authors)
    if key:
        log.info("wrote %d authors -> %s", len(authors), key)
    return counts


async def collect_metrics(
    collector: Collector,
    storage: Storage,
    since_days: int = 5,
    top_pct: float = 0.05,
    max_posts: int = 200,
) -> dict[str, int]:
    """Re-fetch engagement for the top `top_pct` of posts (by likes+quotes+reposts) from
    the last `since_days` days. Writes lightweight count snapshots to the metrics/ prefix.

    `max_posts` is a safety cap: defaults scale with the active account pool.
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

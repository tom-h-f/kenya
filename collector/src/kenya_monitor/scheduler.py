from __future__ import annotations

import asyncio
import logging
import os
import random
from datetime import datetime, timedelta, timezone

from kenya_monitor import adaptive
from kenya_monitor.accounts import active_count, metrics_cap, sync_accounts
from kenya_monitor.collectors.x import MAX_AGE_DAYS, backfill_windows, build_api, recent_windows
from kenya_monitor.config import (
    ACCOUNT_SYNC_HOURS,
    CYCLE_COOLDOWN_MAX_S,
    CYCLE_COOLDOWN_MIN_S,
    FOLLOW_CRAWL_MAX_PER_RUN,
    FOLLOW_CRAWL_REFRESH_DAYS,
    FOLLOW_CRAWL_TOP_HATE,
    FOLLOW_CRAWL_TOP_SUSPICIOUS,
    FOLLOW_FETCH_LIMIT,
    FOLLOW_MAX_ACCOUNTS,
    HATE_MINE_AUTOPROMOTE,
    HATE_SEED_MAX_ACCOUNTS,
    HATE_SEEK_CONCURRENCY,
    HATE_SEEK_ENABLED,
    HATE_SEEK_EVERY_N_CYCLES,
    HATE_SEEK_MAX_TERMS,
    HATE_SEEK_RECENT_DAYS,
    HATE_SEEK_WINDOW_LIMIT,
    HATE_TERMS_PATH,
    METRICS_MAX_POSTS_FLOOR,
    METRICS_MAX_POSTS_PER_ACCOUNT,
    SEARCH_BACKFILL_WINDOW_DAYS,
    SEARCH_MIN_FAVES,
    SEARCH_RECENT_DAYS,
    SEARCH_WINDOW_LIMIT,
    PlatformTargets,
    R2Config,
    load_accounts,
    load_hate_terms,
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

# Metrics pass defaults (cap computed from pool size at runtime).
METRICS_SINCE_DAYS = 5
METRICS_TOP_PCT = 0.05


def _adaptive_targets(storage: Storage, dry_run: bool = False) -> PlatformTargets:
    """Static targets merged with the capped dynamic promotions (hashtag bursts,
    coordination-cluster members, flagged-story keywords). Never edits targets.yaml."""
    static = load_targets().get("x", PlatformTargets())
    try:
        entries = adaptive.promote(
            storage.con,
            storage.posts_view(platform="x"),
            storage.clusters_view(platform="x"),
            storage.authors_view(platform="x"),
            stories_view=storage.stories_view(platform="x"),
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


async def run_hate_seek_once(
    terms: list[str] | None = None,
    limit: int = HATE_SEEK_WINDOW_LIMIT,
    max_terms: int = HATE_SEEK_MAX_TERMS,
    recent_days: int = HATE_SEEK_RECENT_DAYS,
) -> dict[str, int]:
    """One hate-seeking search pass over the rotated coded-term register."""
    from kenya_monitor import hate_seek as hs
    from kenya_monitor.runner import collect_hate_seek

    register = load_hate_terms(HATE_TERMS_PATH)
    state = hs.load_state()
    if terms:
        wanted = {t.lower() for t in terms}
        queries = [
            q
            for q in hs.select_terms(register, max_terms=len(register.terms) + 99, max_targets=99)
            if q.term.lower() in wanted
        ]
        if not queries:
            log.warning("hate-seek: no register term matched %s", terms)
            return {}
    else:
        from kenya_monitor import mined_terms as mt

        mined = mt.active(mt.load(), autopromote=HATE_MINE_AUTOPROMOTE)
        queries = hs.select_terms(register, mined=mined, state=state, max_terms=max_terms)
    if not queries:
        return {}
    storage = Storage(R2Config.from_env())
    collector = await build_x_collector(load_accounts())
    counts = await collect_hate_seek(
        collector,
        storage,
        queries,
        register.anchors,
        search_windows=recent_windows(recent_days),
        min_faves=SEARCH_MIN_FAVES,
        window_limit=limit,
        concurrency=HATE_SEEK_CONCURRENCY,
    )
    hs.save_state(hs.mark_searched(queries, state))
    log.info("hate-seek run complete (%d terms): %s", len(queries), counts)
    return counts


def _hate_seed_handles(storage: Storage, n: int) -> list[str]:
    """Top-N hate-network seeds, falling back to the generic suspicion ranking
    when the enrich worker has stopped writing scores - selecting on stale
    scores would silently freeze the seed set."""
    from kenya_monitor import hate_signal as hsig
    from kenya_monitor.suspicion import top_suspicious_handles

    hate_view = storage.hatespeech_view(platform="x")
    posts_view = storage.posts_view(platform="x")
    authors_view = storage.authors_view(platform="x")
    if hsig.scores_are_stale(storage.con, hate_view):
        log.warning("hate scores are stale or missing; falling back to suspicion ranking")
        return top_suspicious_handles(storage.con, authors_view, posts_view, n=n)
    handles = hsig.top_hate_seeds(
        storage.con, hate_view, posts_view, authors_view,
        storage.engagements_view(platform="x"), n=n,
    )
    if not handles:
        log.info("no accounts cleared the hate-seed floors; falling back to suspicion ranking")
        return top_suspicious_handles(storage.con, authors_view, posts_view, n=n)
    return handles


async def run_hate_expand_once(
    top_hate: int = HATE_SEED_MAX_ACCOUNTS, timeline_limit: int = 40
) -> dict[str, int]:
    """Expand around the hate seeds: their full timelines plus a toxic-object
    snowball. Timelines are the highest-value pass here - they need no search
    operator and no model, so they recover the coded register that the
    classifier structurally cannot see.

    Everything lands in quarantined partitions. Seeds are never added to
    `targets.accounts`: promoting them into baseline collection would inflate
    their post counts and co-action degree, making them look more coordinated
    next run purely because we watched them harder."""
    storage = Storage(R2Config.from_env())
    handles = _hate_seed_handles(storage, top_hate)
    if not handles:
        log.info("hate expand: no seed accounts")
        return {}
    collector = await build_x_collector(load_accounts())
    counts = await collect_x(
        collector,
        storage,
        PlatformTargets(accounts=handles),
        search_windows=[],
        min_faves=0,
        window_limit=0,
        timeline_limit=timeline_limit,
        keywords=False,
        accounts=True,
        timeline_type="hate_timeline",
    )
    from kenya_monitor import hate_signal as hsig

    objects = hsig.hot_toxic_objects(
        storage.con, storage.hatespeech_view(platform="x"), storage.posts_view(platform="x")
    )
    if any(objects):
        counts.update(
            await collect_snowball(
                collector, storage, objects=objects, type_prefix="hate_"
            )
        )
    counts["seed_accounts"] = len(handles)
    log.info("hate expand complete (%d seeds): %s", len(handles), counts)
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
    top_suspicious: int | None = None,
    top_hate: int | None = None,
) -> dict[str, int]:
    """Follower/following edges for explicit handles, flagged-cluster members,
    the top-N hate-network seeds, or the top-N most suspicious accounts."""
    from kenya_monitor.suspicion import top_suspicious_handles

    storage = Storage(R2Config.from_env())
    if top_hate:
        handles = _hate_seed_handles(storage, top_hate)
        log.info("follows: selected %d hate-seed accounts (requested top %d)", len(handles), top_hate)
    elif top_suspicious:
        handles = top_suspicious_handles(
            storage.con,
            storage.authors_view(platform="x"),
            storage.posts_view(platform="x"),
            n=top_suspicious,
        )
        log.info("follows: selected %d suspicious accounts (requested top %d)", len(handles), top_suspicious)
    elif handles is None:
        handles = adaptive.cluster_accounts(
            storage.con, storage.clusters_view(platform="x"), storage.authors_view(platform="x")
        )
    handles = handles[:max_accounts]
    if not handles:
        log.info("follows: no accounts to fetch")
        return {"follow_edges": 0}
    collector = await build_x_collector(load_accounts())
    counts = await collect_follows(collector, storage, handles, limit)
    counts["accounts"] = len(handles)
    log.info("follows run complete: %s", counts)
    return counts


async def run_follow_crawl_once(
    seed_handles: list[str] | None = None,
    limit: int = FOLLOW_FETCH_LIMIT,
    max_accounts: int = FOLLOW_CRAWL_MAX_PER_RUN,
    refresh_days: int = FOLLOW_CRAWL_REFRESH_DAYS,
    from_edges: bool = True,
    top_suspicious: int | None = None,
    top_hate: int | None = None,
) -> dict[str, int]:
    """Recursive BFS follow-graph crawl with persisted per-account state."""
    from kenya_monitor.follow_crawl import crawl_follows, crawl_summary, load_crawl_state
    from kenya_monitor.suspicion import top_suspicious_handles

    storage = Storage(R2Config.from_env())
    seeds = list(seed_handles or [])
    if top_hate:
        seeds.extend(_hate_seed_handles(storage, top_hate))
    if top_suspicious:
        seeds.extend(
            top_suspicious_handles(
                storage.con,
                storage.authors_view(platform="x"),
                storage.posts_view(platform="x"),
                n=top_suspicious,
            )
        )
    collector = await build_x_collector(load_accounts())
    counts = await crawl_follows(
        collector,
        storage,
        seed_handles=seeds,
        limit=limit,
        max_accounts=max_accounts,
        refresh_days=refresh_days,
        from_edges=from_edges,
    )
    summary = crawl_summary(load_crawl_state())
    counts.update({f"tracked_{k}": v for k, v in summary.items() if isinstance(v, int)})
    log.info("follow crawl run complete: %s", counts)
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


async def _maintain_accounts_loop() -> None:
    while True:
        try:
            api = build_api()
            await sync_accounts(api, load_accounts(), relogin_failed=True)
            await api.pool.reset_locks()
        except Exception:
            log.exception("account maintenance failed")
        await asyncio.sleep(ACCOUNT_SYNC_HOURS * 3600)


async def run_scheduler(limit: int) -> None:
    """Always-on collection: cycles of posts -> snowball -> metrics -> follow
    crawl run back to back, forever. The throttle is the per-account pacing +
    twscrape's rate-limit rotation (it waits when the whole pool is limited),
    not wall-clock gaps. A short randomized cooldown between cycles keeps the
    cadence organic; a detected volume burst skips it. Backfill windows join
    the first cycle of each UTC day; pool maintenance runs on its own timer."""
    maintenance = asyncio.create_task(_maintain_accounts_loop())
    last_backfill: dict[str, object] = {"date": None}
    cycle = 0
    try:
        while True:
            cycle += 1
            started = datetime.now(timezone.utc)

            async def _posts() -> None:
                today = datetime.now(timezone.utc).date()
                do_backfill = last_backfill["date"] != today
                await run_once(limit, include_backfill=do_backfill)
                if do_backfill:
                    last_backfill["date"] = today

            # Hate-seeking runs after baseline posts/snowball on purpose: when
            # the account pool hits a rate-limit wall it must hit discretionary
            # work first, never baseline coverage.
            hate_due = HATE_SEEK_ENABLED and cycle % max(1, HATE_SEEK_EVERY_N_CYCLES) == 0
            steps = [
                ("posts", _posts),
                ("snowball", run_snowball_once),
                ("metrics", run_metrics_once),
                (
                    "follow_crawl",
                    lambda: run_follow_crawl_once(
                        top_suspicious=FOLLOW_CRAWL_TOP_SUSPICIOUS,
                        top_hate=FOLLOW_CRAWL_TOP_HATE if HATE_SEEK_ENABLED else None,
                    ),
                ),
            ]
            if hate_due:
                steps.insert(2, ("hate_seek", run_hate_seek_once))
                steps.insert(3, ("hate_expand", run_hate_expand_once))
            for name, step in steps:
                try:
                    await step()
                except Exception:
                    log.exception("cycle %d: %s step failed", cycle, name)

            bursting, z = False, 0.0
            try:
                storage = Storage(R2Config.from_env())
                bursting, z, _n = adaptive.detect_burst(
                    storage.con, storage.posts_view(platform="x")
                )
            except Exception:
                log.exception("burst check failed")
            cooldown = 0.0 if bursting else random.uniform(CYCLE_COOLDOWN_MIN_S, CYCLE_COOLDOWN_MAX_S)
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            log.info(
                "cycle %d done in %.1fmin; %s next cycle in %.0fs (volume z=%.1f)",
                cycle, elapsed / 60,
                "burst -" if bursting else "cooldown,", cooldown, z,
            )
            await asyncio.sleep(cooldown)
    finally:
        maintenance.cancel()

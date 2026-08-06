from __future__ import annotations

import asyncio
import json
import logging

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

from twscrape import API

from kenya_monitor import census
from kenya_monitor.collectors.base import Collector, Engagement, FollowEdge, Post
from kenya_monitor.collectors.x import Window, XCollector, build_api, sync_accounts
from kenya_monitor.config import (
    CENSUS_OVER_BAND_WARN,
    CENSUS_TIMELINE_ACCOUNTS,
    CENSUS_TIMELINE_POOL_HOURS,
    CENSUS_TIMELINE_POOL_SIZE,
    CENSUS_TIMELINE_RETRY_DAYS,
    COLLECT_CONCURRENCY,
    SEARCH_INCLUDE_RETWEETS,
    SEARCH_PRODUCT,
    SNOWBALL_BAND_MAX,
    SNOWBALL_BAND_MIN,
    SNOWBALL_FLUSH_EVERY,
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
    search_type: str = "search",
    timeline_type: str = "timeline",
    include_replies: bool = False,
) -> dict[str, int]:
    """`search_type` / `timeline_type` pick the R2 `posts/type=` partition.
    Targeted passes override them so their oversampled posts stay out of the
    baseline prevalence denominator (see kma.db.BASELINE_TYPES)."""
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
        key = storage.write_posts(search_posts, target_type=search_type)
        counts[search_type] = len(search_posts)
        if key:
            log.info("wrote %d search posts -> %s", len(search_posts), key)

    if accounts and targets.accounts:
        timeline_posts: list[Post] = []
        for handle in targets.accounts:
            got = [
                p
                async for p in collector.timeline(
                    handle, limit=timeline_limit, include_replies=include_replies
                )
            ]
            log.info("timeline @%s -> %d posts", handle, len(got))
            timeline_posts.extend(got)
        key = storage.write_posts(timeline_posts, target_type=timeline_type)
        counts[timeline_type] = len(timeline_posts)
        if key:
            log.info("wrote %d timeline posts -> %s", len(timeline_posts), key)

    authors = collector.collected_authors()
    key = storage.write_authors(authors)
    counts["authors"] = len(authors)
    if key:
        log.info("wrote %d authors -> %s", len(authors), key)

    return counts


async def collect_hate_seek(
    collector: Collector,
    storage: Storage,
    queries: list,
    anchors: dict[str, list[str]],
    search_windows: list[Window],
    min_faves: int = 0,
    window_limit: int = 15,
    concurrency: int = 1,
) -> dict[str, int]:
    """Hate-seeking search pass. Writes to its own `hate_search` /
    `hate_target_search` partitions, never `search`: these queries oversample
    toxic speech by construction, so mixing them into the baseline partition
    would silently bias every prevalence measurement computed downstream
    (see kma.db.BASELINE_TYPES). Records the rendered queries to
    `collection_runs/` as the audit trail.

    Its own semaphore, not COLLECT_CONCURRENCY: this is discretionary work and
    must never contend with baseline keyword coverage."""
    from kenya_monitor import hate_seek as hs

    sem = asyncio.Semaphore(max(1, concurrency))
    manifest: list[dict] = []

    async def one(q) -> tuple[str, list[Post]]:
        got: list[Post] = []
        async with sem:
            for since, until in search_windows:
                rendered = hs.render(
                    q, anchors, since=since, until=until,
                    min_faves=min_faves, include_retweets=SEARCH_INCLUDE_RETWEETS,
                )
                window_posts = [
                    p
                    async for p in collector.search(
                        q.query,
                        limit=window_limit,
                        since=since,
                        until=until,
                        min_faves=min_faves,
                        product=SEARCH_PRODUCT,
                        include_retweets=SEARCH_INCLUDE_RETWEETS,
                        anchors=hs.anchors_for(q, anchors),
                    )
                ]
                for p in window_posts:
                    p.source_query = f"hate:{q.term}"
                got.extend(window_posts)
                manifest.append(
                    {
                        "target_type": q.target_type,
                        "term": q.term,
                        "source": q.source,
                        "fp_risk": q.fp_risk,
                        "query": rendered,
                        "since": since,
                        "until": until,
                        "n_posts": len(window_posts),
                    }
                )
        log.info("hate-seek %r (%s) -> %d posts", q.term, q.source, len(got))
        return q.target_type, got

    results = await asyncio.gather(*[one(q) for q in queries])

    counts: dict[str, int] = {}
    by_partition: dict[str, list[Post]] = {}
    for target_type, got in results:
        by_partition.setdefault(target_type, []).extend(got)
    for target_type, got in by_partition.items():
        key = storage.write_posts(got, target_type=target_type)
        counts[target_type] = len(got)
        if key:
            log.info("wrote %d %s posts -> %s", len(got), target_type, key)

    authors = collector.collected_authors()
    if authors:
        storage.write_authors(authors)
        counts["authors"] = len(authors)
    manifest_key = storage.write_collection_run(manifest)
    if manifest_key:
        log.info("wrote %d manifest rows -> %s", len(manifest), manifest_key)
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
    band_min: int = SNOWBALL_BAND_MIN,
    band_max: int = SNOWBALL_BAND_MAX,
    engagements_view: str | None = None,
    replies_view: str | None = None,
    refresh_hours: int = SNOWBALL_REFRESH_HOURS,
    *,
    stats: dict | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Snowball targets from already-collected posts: the most-amplified
    objects, the busiest conversations, and referenced ids we never fetched.
    Returns (retweeted_ids, conversation_ids, missing_ids).

    `stats`, when given, is filled with the selection counters - in particular
    `candidates_in_band` and `candidates_uncensused`, the supply and the
    backlog. Those are what make coverage saturation observable: once the
    backlog trends to zero, `SNOWBALL_REFRESH_HOURS` and not
    `SNOWBALL_TOP_RETWEETED` is the parameter that governs discovery."""
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
    # Exclude objects censused inside the TTL *during selection*, not after.
    # Selection ranks by repost_count and is deterministic, so without this the
    # same densest-in-band objects are re-picked every pass and then discarded
    # by `_due` - observed: 495 selected, 10-35 actually fetched, while ~2,900
    # uncensused in-band objects went untouched.
    censused = census.censused_expr(
        con, engagements_view, "recent.repost_of_id", refresh_hours
    )

    # The backlog counts come off the SAME query as the selection. The
    # GROUP BY ... HAVING result has to be fully materialised before
    # ORDER BY ... LIMIT anyway, so the windows are free; a second query over
    # the same CTE would not be. Both windows sit BEFORE the TTL filter, so the
    # supply figure covers the whole band and not just the part still unworked.
    rows = con.sql(
        base
        + f"""
        , cand AS (
            SELECT recent.repost_of_id AS oid,
                   max(repost_count) AS deg,
                   count(*) AS n_rt_rows,
                   {censused} AS censused
            FROM recent
            WHERE repost_of_id IS NOT NULL
            GROUP BY 1
            HAVING max(repost_count) BETWEEN {int(band_min)} AND {int(band_max)}
        ), ranked AS (
            SELECT oid, deg, n_rt_rows, censused,
                   count(*) OVER () AS n_in_band,
                   sum(CASE WHEN censused THEN 0 ELSE 1 END) OVER () AS n_uncensused
            FROM cand
        )
        SELECT oid, deg, n_in_band, n_uncensused, censused FROM ranked
        -- Uncensused first, then densest-first WITHIN the band: nearer the hub
        -- cap means more co-amplification pairs per request, without crossing
        -- it. Censused rows are ordered last and dropped below rather than
        -- filtered out here, so that when everything in the band is already
        -- censused the query still returns a row carrying the window counts.
        -- That is the saturated state - the one the backlog series exists to
        -- detect - and it must not report an empty band.
        ORDER BY censused, deg DESC, n_rt_rows DESC, oid
        LIMIT {int(top_retweeted)}
        """
    ).fetchall()
    picked = [r for r in rows if not r[4]]
    retweeted = [r[0] for r in picked]
    if stats is not None:
        stats["candidates_in_band"] = int(rows[0][2]) if rows else 0
        stats["candidates_uncensused"] = int(rows[0][3]) if rows else 0
        stats["selected_retweeted"] = len(retweeted)
        stats["selected_deg_min"] = int(picked[-1][1]) if picked else 0
        stats["selected_deg_max"] = int(picked[0][1]) if picked else 0
        stats["band_min"] = int(band_min)
        stats["band_max"] = int(band_max)
        stats["lookback_days"] = int(lookback_days)
        stats["refresh_hours"] = int(refresh_hours)
        stats["top_retweeted"] = int(top_retweeted)
    # The conversation arm is banded and TTL-aware for the same reasons as the
    # retweeted arm above, and it was neither until 2026-08-06.
    #
    # Unbanded `ORDER BY max(reply_count) DESC` selects the BIGGEST threads,
    # which is precisely the population `HUB_CAP_MAX` deletes on the analysis
    # side. Measured on collected posts/type=replies over the 14-day window: 82
    # parents above the cap carried 31% of all collected reply rows and 54% of
    # every account pair those rows offered, fetched and then discarded. The
    # same window held 14,608 conversations INSIDE the band, untouched.
    #
    # This is what starved co_reply. Per pairable account the channel tested
    # 0.57 pairs against co_retweet's 24, and of the 72 accounts in the
    # validated co_reply layer exactly 1 had any co_retweet trace and 0 were
    # co_retweet-pairable - so corroboration had a ceiling of zero and the 0/1
    # it printed was one bridge account flickering. See docs/analysis/
    # census-tuning.md §11.
    #
    # `max(reply_count)` is the thread root's own reply count, a proxy for the
    # distinct-replier degree the hub cap actually measures. It is the quantity
    # the previous ranking already used, so banding on it changes selection
    # without introducing a new notion of conversation size.
    threaded = census.threaded_expr(
        con, replies_view, "recent.conversation_id", refresh_hours
    )
    conv_rows = con.sql(
        base
        + f"""
        , conv AS (
            SELECT recent.conversation_id AS cid,
                   max(reply_count) AS deg,
                   count(*) AS n_reply_rows,
                   {threaded} AS threaded
            FROM recent
            WHERE conversation_id IS NOT NULL
            GROUP BY 1
            HAVING max(reply_count) BETWEEN {int(band_min)} AND {int(band_max)}
        ), ranked AS (
            SELECT cid, deg, n_reply_rows, threaded,
                   count(*) OVER () AS n_in_band,
                   sum(CASE WHEN threaded THEN 0 ELSE 1 END) OVER () AS n_unthreaded
            FROM conv
        )
        SELECT cid, deg, n_in_band, n_unthreaded, threaded FROM ranked
        -- Unworked first, then densest-first WITHIN the band, mirroring the
        -- retweeted arm. Threaded rows are ordered last and dropped below
        -- rather than filtered out here, so a fully-worked band still returns
        -- a row carrying the window counts instead of reporting empty supply.
        ORDER BY threaded, deg DESC, n_reply_rows DESC, cid
        LIMIT {int(top_conversations)}
        """
    ).fetchall()
    conv_picked = [r for r in conv_rows if not r[4]]
    conversations = [r[0] for r in conv_picked]
    if stats is not None:
        stats["conversations_in_band"] = int(conv_rows[0][2]) if conv_rows else 0
        stats["conversations_unthreaded"] = int(conv_rows[0][3]) if conv_rows else 0
        stats["selected_conversations"] = len(conversations)
        stats["conv_deg_min"] = int(conv_picked[-1][1]) if conv_picked else 0
        stats["conv_deg_max"] = int(conv_picked[0][1]) if conv_picked else 0
        stats["top_conversations"] = int(top_conversations)
    # Censused-but-unhydrated objects come FIRST, ahead of the most-engaged.
    #
    # A censused object with no post row has no `created_at`, so the
    # coordination window cannot place it and every one of its retweeter traces
    # is unusable. Measured 2026-08-02: 2,805 of 4,679 censused objects (60%)
    # were in that state, carrying 97,621 traces - 30% of the whole engagement
    # arm. Ordering by engagement instead sent these slots to the most-amplified
    # refs, i.e. exactly the hubs the census no longer collects.
    #
    # `hydrate` costs one API call per id, so the budget is small and spending
    # it on the right ids matters more than raising it.
    censused_ever = census.censused_ever_expr(con, engagements_view, "refs.ref")
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
            ), unresolved AS (
                SELECT refs.ref AS ref, max(refs.eng) AS eng,
                       bool_or({censused_ever}) AS censused
                FROM refs
                WHERE NOT EXISTS (
                    SELECT 1 FROM lp WHERE lp.platform_post_id = refs.ref
                )
                GROUP BY 1
            )
            SELECT ref FROM unresolved
            ORDER BY censused DESC, eng DESC
            LIMIT {int(hydrate_limit)}
            """
        ).fetchall()
    ]
    if stats is not None:
        stats["selected_conversations"] = len(conversations)
        stats["selected_missing"] = len(missing)
    return retweeted, conversations, missing


def select_census_objects(
    con: duckdb.DuckDBPyConnection,
    posts_view: str,
    engagements_view: str | None = None,
    replies_view: str | None = None,
    hatespeech_view: str | None = None,
    *,
    stats: dict | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Engagement-ranked census targets with toxic-ranked ones appended.

    One place knows about both selectors, so the arguments they must share
    cannot be passed to one and forgotten on the other - which is how the
    censused-TTL filter came to be applied on the baseline path only.

    Toxic ids are APPENDED on purpose: `collect_snowball` walks the list in
    order and flushes every `SNOWBALL_FLUSH_EVERY`, so a rate-limit abort
    truncates the discretionary tail rather than the baseline."""
    from kenya_monitor import hate_signal as hsig

    retweeted, conversations, missing = hot_objects(
        con,
        posts_view,
        engagements_view=engagements_view,
        replies_view=replies_view,
        stats=stats,
    )
    if hatespeech_view is None:
        return retweeted, conversations, missing

    tox_rt, _tox_conv, _ = hsig.hot_toxic_objects(
        con, hatespeech_view, posts_view, engagements_view=engagements_view
    )
    seen = set(retweeted)
    extra = [o for o in tox_rt if o not in seen]
    if extra:
        log.info(
            "snowball: +%d toxic-ranked objects appended to %d baseline",
            len(extra), len(retweeted),
        )
    if stats is not None:
        stats["pass_kind"] = "merged"
        stats["selected_retweeted"] = len(retweeted) + len(extra)
    return retweeted + extra, conversations, missing


def census_discovered_handles(
    con: duckdb.DuckDBPyConnection,
    engagements_view: str,
    authors_view: str,
    posts_view: str,
    limit: int = CENSUS_TIMELINE_ACCOUNTS,
    state: dict | None = None,
    retry_days: int = CENSUS_TIMELINE_RETRY_DAYS,
    pool_size: int = CENSUS_TIMELINE_POOL_SIZE,
    pool_hours: int = CENSUS_TIMELINE_POOL_HOURS,
) -> list[str]:
    """Handles of accounts the census discovered but whose posts we never
    collected, sampled at random.

    These accounts exist only as retweeter ids, so they carry `co_retweet`
    traces and can never carry `co_reply` ones - which is why corroboration
    reads 0 and cannot currently be evaluated at all.

    `ORDER BY random()` is load bearing. Selecting by cluster membership or
    suspicion would condition collection on the outcome being measured: the
    promoted accounts would gain posts, traces and edges, and look more
    coordinated on the next run. Random selection is exogenous, so it closes
    the population gap without manufacturing the answer.

    The candidate set is self-draining - once an account has any post row it
    stops qualifying. `state` covers the accounts that never will (no tweets,
    suspended, protected), which would otherwise be re-picked forever."""
    state = {} if state is None else state
    cutoff = datetime.now(timezone.utc) - timedelta(days=retry_days)
    attempted: dict = state.setdefault("attempted", {})
    pool: list = state.get("pool") or []

    if _pool_is_stale(state, pool_hours):
        # Announce before the query: it scans the parquet globs over the network
        # and takes minutes, during which a silent container is indistinguishable
        # from a hung one.
        log.info("census timelines: refreshing candidate pool (%d)", pool_size)
        pool = _refresh_census_pool(
            con, engagements_view, authors_view, posts_view, size=pool_size
        )
        if pool:
            state["pool"] = pool
            state["pool_refreshed_at"] = datetime.now(timezone.utc).isoformat()

    picked: list[tuple[str, str]] = []
    remaining: list = []
    for uid, handle in pool:
        seen = attempted.get(uid)
        if seen and datetime.fromisoformat(seen) >= cutoff:
            continue
        if len(picked) < limit:
            picked.append((uid, handle))
        else:
            remaining.append([uid, handle])
    state["pool"] = remaining

    now_iso = datetime.now(timezone.utc).isoformat()
    for uid, _ in picked:
        attempted[uid] = now_iso
    # Attempted entries older than the retry window are dead weight.
    state["attempted"] = {
        u: t for u, t in attempted.items() if datetime.fromisoformat(t) >= cutoff
    }
    return [h for _, h in picked]


def _pool_is_stale(state: dict, pool_hours: int) -> bool:
    if not state.get("pool"):
        return True
    ts = state.get("pool_refreshed_at")
    if not ts:
        return True
    age = datetime.now(timezone.utc) - datetime.fromisoformat(ts)
    return age >= timedelta(hours=pool_hours)


def _refresh_census_pool(
    con: duckdb.DuckDBPyConnection,
    engagements_view: str,
    authors_view: str,
    posts_view: str,
    size: int,
) -> list[list[str]]:
    """Draw a batch of candidate (user_id, handle) pairs. Expensive; cached.

    Measured against live R2 at 800MB/2 threads: 464s with the posts anti-join,
    215s without. Both are far too slow to run once per cycle on a Raspberry Pi,
    and neither cost depends much on how many rows come back - it is dominated
    by scanning the parquet globs over the network. So it is paid rarely and
    amortised over many cycles.

    The posts anti-join is dropped deliberately. It exists to skip accounts that
    already have posts, but 179,106 of ~180,000 censused accounts have none, so
    it removes almost nothing for half the runtime. The `attempted` map prevents
    repeats, which is what actually matters."""
    try:
        return [
            [r[0], r[1]]
            for r in con.sql(
                f"""
                WITH censused AS (
                    SELECT DISTINCT platform_user_id AS uid
                    FROM {engagements_view} WHERE kind = 'retweet'
                ), sampled AS (
                    SELECT uid FROM censused ORDER BY random() LIMIT {int(size) * 2}
                ), la AS (
                    SELECT * FROM {authors_view}
                    WHERE platform_user_id IN (SELECT uid FROM sampled)
                    QUALIFY row_number() OVER (
                        PARTITION BY platform, platform_user_id
                        ORDER BY collected_at DESC
                    ) = 1
                )
                SELECT la.platform_user_id, la.handle
                FROM sampled s
                JOIN la ON la.platform_user_id = s.uid
                WHERE la.handle IS NOT NULL
                ORDER BY random()
                LIMIT {int(size)}
                """
            ).fetchall()
        ]
    except duckdb.Error:
        log.exception("census timelines: pool refresh failed")
        return []


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
    objects: tuple[list[str], list[str], list[str]] | None = None,
    type_prefix: str = "",
    flush_every: int = SNOWBALL_FLUSH_EVERY,
    band_max: int = SNOWBALL_BAND_MAX,
    pass_kind: str = "baseline",
    stats: dict | None = None,
) -> dict[str, int]:
    """Census pass over hot objects: full retweeter lists (-> engagements/),
    full reply threads (-> posts/type=replies), and hydration of referenced
    originals (-> posts/type=hydrated). Per-object TTL in `state_path` stops
    hot objects being re-fetched every pass.

    `objects` supplies a pre-computed (retweeted, conversations, missing) triple
    instead of the engagement-ranked default - the toxic snowball passes
    `hate_signal.hot_toxic_objects`. `type_prefix` redirects the written posts
    to a quarantined partition (`hate_replies` / `hate_hydrated`); engagement
    edges have no `type` partition and need none, since they never enter a
    prevalence denominator."""
    counts: dict[str, int] = {}
    stats = {} if stats is None else stats
    retweeted, conversations, missing = objects or hot_objects(
        storage.con,
        storage.posts_view(platform=collector.platform),
        lookback_days,
        top_retweeted,
        top_conversations,
        hydrate_limit,
        stats=stats,
    )
    state = _load_snowball_state(state_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    # Flush in batches rather than once at the end. At the original 15 objects a
    # pass took a couple of minutes; at 250 it is ~40 minutes, and a single write
    # at the end means a restart or a rate-limit abort discards every API call
    # made in that window - the expensive resource here is pool budget, not CPU.
    #
    # Order within a batch matters: write FIRST, then mark the objects done, then
    # checkpoint. A crash before the write leaves them unmarked and they retry;
    # a crash between write and checkpoint re-fetches them next pass, which is
    # harmless because reads dedup on (platform, platform_post_id).
    due_rt = _due(retweeted, state, refresh_hours)
    counts["retweeters"] = 0
    batch: list[Engagement] = []
    pending: list[str] = []

    def _flush_engagements() -> None:
        nonlocal batch, pending
        if not batch:
            return
        key = storage.write_engagements(batch)
        counts["retweeters"] += len(batch)
        if key:
            log.info("wrote %d engagement rows -> %s", len(batch), key)
        for done_id in pending:
            state[done_id] = now_iso
        _save_snowball_state(state, state_path)
        batch, pending = [], []

    degrees: list[int] = []
    for i, oid in enumerate(due_rt, 1):
        got = [e async for e in collector.retweeters(oid, limit=retweeters_limit)]
        log.info("retweeters of %s -> %d", oid, len(got))
        degrees.append(len(got))
        batch.extend(got)
        pending.append(oid)
        if i % max(1, flush_every) == 0:
            _flush_engagements()
    _flush_engagements()

    due_conv = _due(conversations, state, refresh_hours)
    counts["replies"] = 0
    reply_batch: list[Post] = []
    reply_pending: list[str] = []

    def _flush_replies() -> None:
        nonlocal reply_batch, reply_pending
        if not reply_batch:
            return
        key = storage.write_posts(reply_batch, target_type=f"{type_prefix}replies")
        counts["replies"] += len(reply_batch)
        if key:
            log.info("wrote %d reply posts -> %s", len(reply_batch), key)
        for done_id in reply_pending:
            state[f"conv:{done_id}"] = now_iso
        _save_snowball_state(state, state_path)
        reply_batch, reply_pending = [], []

    for i, cid in enumerate(due_conv, 1):
        got = [p async for p in collector.replies(cid, limit=replies_limit)]
        log.info("replies under %s -> %d", cid, len(got))
        reply_batch.extend(got)
        reply_pending.append(cid)
        if i % max(1, flush_every) == 0:
            _flush_replies()
    _flush_replies()

    hydrated = [p async for p in collector.hydrate(missing)]
    key = storage.write_posts(hydrated, target_type=f"{type_prefix}hydrated")
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

    stats.update(
        {
            "pass_kind": pass_kind,
            "retweeters_limit": int(retweeters_limit),
            "refresh_hours": int(refresh_hours),
            "band_max": int(band_max),
            "selected_retweeted": stats.get("selected_retweeted", len(retweeted)),
            "due_retweeted": len(due_rt),
            "fetched_retweeted": len(degrees),
            "skipped_ttl_retweeted": len(retweeted) - len(due_rt),
            "engagement_rows": counts["retweeters"],
            "degree_p50": _quantile(degrees, 0.5),
            "degree_p90": _quantile(degrees, 0.9),
            "degree_max": max(degrees, default=0),
            "n_over_band_max": sum(1 for d in degrees if d > band_max),
            "selected_conversations": stats.get(
                "selected_conversations", len(conversations)
            ),
            "due_conversations": len(due_conv),
            "reply_rows": counts["replies"],
            "hydrated_rows": counts["hydrated"],
            "authors_written": counts["authors"],
        }
    )
    _warn_if_band_is_drifting(stats)
    storage.write_census_run(stats, platform=collector.platform)
    return counts


def _quantile(xs: list[int], q: float) -> int:
    """Nearest-rank quantile. Deliberately not numpy - the collector's
    dependency set is model-free and numpy is not a direct dependency."""
    if not xs:
        return 0
    ordered = sorted(xs)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return int(ordered[idx])


def _warn_if_band_is_drifting(stats: dict) -> None:
    """Standing guard for the census-tuning Q2 regression.

    Objects fetched above the band max are objects the analysis hub cap will
    discard, so every request spent on them buys zero usable pairs. Before
    banding, 99.5% of censused objects were hubs; after, the measured rate is
    6.4%. The threshold sits well clear of the healthy state and trips
    immediately on a return to the old behaviour.

    Note this measures the FETCH, not the truth: `SNOWBALL_RETWEETERS_LIMIT`
    truncates, so a 5,000-retweet object registers as 300. That is fine for a
    guard (300 > band_max trips it) but do not read the degree fields as a
    distribution, and do not respond by lowering the limit - truncating would
    disguise a hub as a mid-degree object and inject it into the projection."""
    fetched = stats.get("fetched_retweeted", 0)
    if not fetched:
        return
    share = stats["n_over_band_max"] / fetched
    if share > CENSUS_OVER_BAND_WARN:
        log.warning(
            "census: %.0f%% of fetched objects came back above the band max (%d) "
            "- selection is drifting back above the analysis hub cap, where "
            "objects contribute zero usable pairs",
            100 * share,
            stats.get("band_max", SNOWBALL_BAND_MAX),
        )


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

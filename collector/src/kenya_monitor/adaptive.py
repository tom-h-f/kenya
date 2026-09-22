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
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

from kenya_monitor.config import (
    BURST_MIN_POSTS,
    BURST_ZSCORE,
    CLUSTER_MIN_CHANNELS,
    CLUSTER_PROMOTION_ENABLED,
    CLUSTER_MIN_KENYA_SHARE,
    CLUSTER_RELEVANCE_THRESHOLD,
    KEYWORD_MIN_KENYA_SHARE,
    DYNAMIC_EXPIRY_DAYS,
    DYNAMIC_HASHTAG_MIN_COUNT,
    DYNAMIC_HASHTAG_RATIO,
    DYNAMIC_MAX_ACCOUNTS,
    DYNAMIC_MAX_KEYWORDS,
    DYNAMIC_TARGETS_PATH,
    STORY_FLAG_MIN_INDEX,
    V2_PROMOTION_ENABLED,
    V2_PROMOTION_MAX_AGE_HOURS,
    V2_PROMOTION_MAX_NEW_PER_PASS,
    V2_PROMOTION_STATE_PATH,
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
    # Burst magnitude (24h count) at the last confirmation, for keywords. The
    # cap sorted on `last_confirmed` alone, and every entry confirmed in the
    # same pass shares that timestamp - so ties broke on dict insertion order
    # and an incumbent tag beat a hotter new burst. Defaulted so state files
    # written before this field loads unchanged.
    strength: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(path: Path = DYNAMIC_TARGETS_PATH) -> list[DynamicEntry]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [DynamicEntry(**e) for e in raw.get("entries", [])]


def save_state(entries: list[DynamicEntry], path: Path = DYNAMIC_TARGETS_PATH) -> None:
    """Temp file + rename, so a crash mid-write cannot truncate the target set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"updated_at": _now_iso(), "entries": [e.__dict__ for e in entries]}, indent=2)
    )
    os.replace(tmp, path)


def bursting_hashtags(
    con: duckdb.DuckDBPyConnection,
    posts_view: str,
    min_count: int = DYNAMIC_HASHTAG_MIN_COUNT,
    ratio: float = DYNAMIC_HASHTAG_RATIO,
    hatespeech_view: str | None = None,
    min_kenya_share: float = KEYWORD_MIN_KENYA_SHARE,
) -> list[tuple[str, int]]:
    """Hashtags whose last-24h volume is `ratio` x their prior-7d daily average
    (new tags: any volume >= min_count), as (tag, n24) pairs.

    Volume and acceleration alone are NOT a relevance test. The implicit
    assumption was that a tag reaching 20/day inside a Kenya-targeted corpus is
    Kenyan, and a globally trending tag clears that easily: `#bbnaija`,
    `#bbnaijaxpepsi`, `#thirstyformore`, `#momsonn` and `#citizenweekend` were
    all promoted, and all 709 of their rows landed in `type=search` - a BASELINE
    partition, i.e. directly into the prevalence denominator.

    When `hatespeech_view` is given, a tag must also carry a minimum share of
    Kenya-referencing posts, read from the persisted `domain` column. A tag with
    no scored posts is allowed through and counted, for the same reason as in
    `cluster_accounts`: failing closed would stop promotion entirely whenever
    enrichment falls behind.

    `n24` travels with the tag because the caller's cap needs it. Sorting the
    cap by `last_confirmed` alone made every tag confirmed this pass tie, so
    ties broke on insertion order and an already-known tag beat a hotter new
    burst - the burst magnitude was discarded before the cap was applied.
    """
    if hatespeech_view is not None:
        kenya_cte = f"""
            , lh AS (
                SELECT platform_post_id, domain FROM {hatespeech_view}
                QUALIFY row_number() OVER (
                    PARTITION BY platform_post_id ORDER BY scored_at DESC
                ) = 1
            ), tag_domain AS (
                SELECT t.tag,
                       count(*) AS n_scored,
                       avg(CASE WHEN lh.domain = 'kenya' THEN 1.0 ELSE 0.0 END)
                           AS kenya_share
                FROM tags t
                JOIN lh ON lh.platform_post_id = t.platform_post_id
                WHERE lh.domain IS NOT NULL
                GROUP BY 1
            )"""
        kenya_join = "LEFT JOIN tag_domain td USING (tag)"
        kenya_filter = (
            f"AND (td.n_scored IS NULL OR td.n_scored = 0 "
            f"OR td.kenya_share >= {float(min_kenya_share)})"
        )
    else:
        log.warning("keyword promotion: no hatespeech view; Kenya gate disabled")
        kenya_cte = kenya_join = kenya_filter = ""

    rows = con.sql(
        f"""
        WITH lp AS (
            SELECT * FROM {posts_view}
            -- Partition pruning: the widest window below reaches back 8 days by
            -- created_at, and nothing can be collected before it was posted, so
            -- no in-window row lives in an older dt partition. Without this the
            -- whole corpus is scanned once per posts step to look at 8 days.
            WHERE dt >= current_date - INTERVAL 9 DAY
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
            ) = 1
        ), tags AS (
            SELECT platform_post_id, lower(unnest(hashtags)) AS tag, created_at
            FROM lp WHERE len(hashtags) > 0
        ), recent AS (
            SELECT tag, count(*) AS n24 FROM tags
            WHERE created_at > now() - INTERVAL 1 DAY GROUP BY 1
        ), baseline AS (
            SELECT tag, count(*) / 7.0 AS daily FROM tags
            WHERE created_at <= now() - INTERVAL 1 DAY
              AND created_at > now() - INTERVAL 8 DAY
            GROUP BY 1
        ){kenya_cte}
        SELECT r.tag, r.n24
        FROM recent r
        LEFT JOIN baseline b USING (tag)
        {kenya_join}
        WHERE r.n24 >= {int(min_count)}
          AND (b.daily IS NULL OR r.n24 >= {float(ratio)} * b.daily)
          {kenya_filter}
        ORDER BY r.n24 DESC
        """
    ).fetchall()
    return [(f"#{r[0]}", int(r[1])) for r in rows]


def cluster_accounts(
    con: duckdb.DuckDBPyConnection,
    clusters_view: str,
    authors_view: str,
    min_channels: int = CLUSTER_MIN_CHANNELS,
    posts_view: str | None = None,
    hatespeech_view: str | None = None,
    min_kenya_share: float = CLUSTER_MIN_KENYA_SHARE,
    enabled: bool = CLUSTER_PROMOTION_ENABLED,
    relevance_view: str | None = None,
    relevance_threshold: float = CLUSTER_RELEVANCE_THRESHOLD,
) -> list[str]:
    """Handles from the most recent persisted coordination run, gated on
    cross-channel corroboration AND on the cluster being about Kenya.

    Two gates, because either alone is wrong.

    Corroboration is the strongest evidence available short of ground truth,
    and without it single-channel clusters cover most of the active author base
    - promoting them all would target nearly everyone.

    But corroboration alone selects the WRONG population. Measured 2026-08-12,
    corroborated clusters are 8.3% Kenya-referencing against 51.5% for
    single-channel ones: reciprocal engagement pods are the most abundant
    coordination on the platform and co_reply detects them best. Raising the
    channel floor without a relevance gate spends the expansion budget on
    Ugandan, Nigerian and US follow-trains.

    The Kenya share is read from the `domain` column `kma.measure` persists onto
    `hatespeech/`, so this costs a join rather than re-deriving the regex. A
    cluster with no scored posts CANNOT be evaluated and is allowed through
    rather than rejected: failing closed would silently stop all targeting
    whenever the enrichment worker falls behind. It is counted separately so
    that stays visible.

    Ordered strongest first so the caller's cap keeps the best candidates.

    Returns nothing when `enabled` is false, before touching R2: the query is
    minutes of network scan on pi0, so a disabled path must not pay for it.
    """
    if not enabled:
        log.info("cluster promotion disabled; no accounts promoted")
        return []
    gate_available = posts_view is not None and hatespeech_view is not None
    if gate_available:
        # The learned gate where a post has been scored, the keyword `domain`
        # column where it has not. The keyword one is precise (0.971) and leaky
        # (recall 0.655), and its misses are plain Kenyan politics that avoids
        # the anchor vocabulary - exactly the clusters worth promoting.
        relevance_cte = (
            f""", lr AS (
                SELECT platform_post_id, p_kenya FROM {relevance_view}
                QUALIFY row_number() OVER (
                    PARTITION BY platform_post_id ORDER BY scored_at DESC
                ) = 1
            )"""
            if relevance_view else ""
        )
        relevance_join = (
            "LEFT JOIN lr ON lr.platform_post_id = lp.platform_post_id" if relevance_view else ""
        )
        kenya_call = (
            f"CASE WHEN lr.p_kenya IS NOT NULL THEN lr.p_kenya >= {float(relevance_threshold)}"
            " ELSE lh.domain = 'kenya' END"
            if relevance_view else "lh.domain = 'kenya'"
        )
        kenya_cte = f"""
            , lp AS (
                SELECT author_id, platform_post_id FROM {posts_view}
                QUALIFY row_number() OVER (
                    PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
                ) = 1
            ), lh AS (
                SELECT platform_post_id, domain FROM {hatespeech_view}
                QUALIFY row_number() OVER (
                    PARTITION BY platform_post_id ORDER BY scored_at DESC
                ) = 1
            ){relevance_cte}, cluster_domain AS (
                SELECT c.cluster_id,
                       count(*) AS n_scored,
                       avg(CASE WHEN {kenya_call} THEN 1.0 ELSE 0.0 END)
                           AS kenya_share
                FROM latest_run c
                JOIN lp ON lp.author_id = c.author_id
                JOIN lh ON lh.platform_post_id = lp.platform_post_id
                {relevance_join}
                WHERE lh.domain IS NOT NULL
                GROUP BY c.cluster_id
            )"""
        kenya_join = "LEFT JOIN cluster_domain d ON d.cluster_id = c.cluster_id"
        kenya_share_expr, n_scored_expr = "d.kenya_share", "d.n_scored"
    else:
        log.warning(
            "cluster targeting: no hatespeech view supplied; Kenya gate disabled"
        )
        kenya_cte, kenya_join = "", ""
        kenya_share_expr = n_scored_expr = "NULL"

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
            ){kenya_cte}
            SELECT la.handle,
                   max(COALESCE(c.n_channels, 1)) AS n_channels,
                   max(c.internal_edge_share) AS density,
                   max({kenya_share_expr}) AS kenya_share,
                   max({n_scored_expr}) AS n_scored
            FROM latest_run c
            JOIN la ON c.author_id = la.platform_user_id
            {kenya_join}
            WHERE la.handle IS NOT NULL
            GROUP BY la.handle
            ORDER BY n_channels DESC, kenya_share DESC NULLS LAST,
                     density DESC NULLS LAST
            """
        ).fetchall()
    except duckdb.Error:
        log.warning("cluster targeting: clusters unreadable; promoting nobody")
        return []  # no clusters persisted yet, or an older schema without n_channels

    kept: list[str] = []
    counts = {"candidates": len(rows), "channels": 0, "kenya": 0, "unmeasured": 0}
    for handle, n_channels, _density, kenya_share, n_scored in rows:
        if (n_channels or 1) < int(min_channels):
            counts["channels"] += 1
            continue
        if gate_available and (n_scored is None or not n_scored):
            counts["unmeasured"] += 1
        elif gate_available and float(kenya_share) < float(min_kenya_share):
            counts["kenya"] += 1
            continue
        kept.append(handle)

    log.info(
        "cluster targeting: %d candidates -> %d promoted "
        "(rejected: %d on channels, %d on Kenya share; %d unmeasured, allowed)",
        counts["candidates"], len(kept),
        counts["channels"], counts["kenya"], counts["unmeasured"],
    )
    if not kept:
        # Distinguish "nothing cleared the bar" from the error path above. They
        # look identical from outside and mean opposite things.
        log.warning(
            "cluster targeting: nothing cleared n_channels >= %d and Kenya share "
            ">= %.2f; promoting nobody",
            int(min_channels), float(min_kenya_share),
        )
    return kept


def v2_accounts(
    con: duckdb.DuckDBPyConnection,
    scores_view: str,
    authors_view: str,
    min_kenya_share: float = CLUSTER_MIN_KENYA_SHARE,
    max_age_hours: float = V2_PROMOTION_MAX_AGE_HOURS,
    state_path: Path = V2_PROMOTION_STATE_PATH,
    now: datetime | None = None,
) -> list[str]:
    """Handles from the latest v2 scores run, strongest centrality first.

    The re-point docs/plans/2026-09-05-v2-methodology.md section 11 calls for:
    v2 ranks accounts, so this needs no clusters. Latest run only, for the
    reason `deep_timelines._score_targets` gives - each run is a complete
    ranking and a union of two belongs to neither.

    Refuses a run older than `max_age_hours`. The latest run on 2026-09-22 was
    eight days old; promoting from it would densify collection around last
    week's ranking and call it current.

    Scores carry ids, not handles, and a handle comes from the authors prefix -
    a full scan (resolving ONE handle measured 1,580s against live R2 from the
    mac on 2026-09-22). So handles are resolved once per scores run and cached
    under its `computed_at`; the posts pass calls this several times a day, and
    a new ranking arrives at most daily.

    `predicted` is not a gate: it was true for 500 of 500 in every 2026-09-14
    run. The Kenya share is kept as one, at the cluster path's threshold, and
    passes 499 of 500 on the latest run - so neither bounds the rate. The cap
    in `promote` does."""
    now = now or datetime.now(timezone.utc)
    try:
        latest = con.sql(
            f"SELECT CAST(max(computed_at) AS VARCHAR) FROM {scores_view}"
        ).fetchone()[0]
    except duckdb.IOException:
        log.info("v2 promotion: no persisted v2 scores; promoting nobody")
        return []
    if latest is None:
        return []
    age = now - datetime.fromisoformat(latest)
    if age > timedelta(hours=max_age_hours):
        log.warning(
            "v2 promotion: latest scores run %s is %.0fh old (limit %.0fh); promoting nobody",
            latest, age.total_seconds() / 3600, max_age_hours,
        )
        return []

    try:
        cached = json.loads(state_path.read_text())
    except (OSError, ValueError):
        cached = {}
    if cached.get("computed_at") == latest and cached.get("min_kenya_share") == min_kenya_share:
        return list(cached.get("handles", []))

    rows = con.sql(
        f"""
        WITH ranked AS (
            SELECT CAST(user_id AS VARCHAR) AS user_id, max(centrality) AS centrality
            FROM {scores_view}
            WHERE computed_at = (SELECT max(computed_at) FROM {scores_view})
              AND user_id IS NOT NULL
              AND kenya_share >= {float(min_kenya_share)}
            GROUP BY 1
        ), handles AS (
            SELECT platform_user_id, arg_max(handle, collected_at) AS handle
            FROM {authors_view}
            WHERE handle IS NOT NULL AND trim(handle) != ''
              AND platform_user_id IN (SELECT user_id FROM ranked)
            GROUP BY platform_user_id
        )
        SELECT h.handle
        FROM ranked r JOIN handles h ON h.platform_user_id = r.user_id
        ORDER BY r.centrality DESC
        """
    ).fetchall()
    handles = [r[0] for r in rows]
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "computed_at": latest,
            "min_kenya_share": min_kenya_share,
            "resolved_at": now.isoformat(),
            "handles": handles,
        }, indent=2))
        os.replace(tmp, state_path)
    except OSError:
        log.warning("v2 promotion: could not cache handles at %s", state_path)
    log.info("v2 promotion: %d handles from scores run %s", len(handles), latest)
    return handles


def cap_new_accounts(
    candidates: list[str], existing: list[DynamicEntry], max_new: int
) -> list[str]:
    """Keep every candidate already live, and at most `max_new` others, in
    candidate order.

    The rate bound for v2 promotion. Without it the first pass after enabling
    adds up to DYNAMIC_MAX_ACCOUNTS accounts at once, and a ranking that moves
    between runs churns the whole set daily - the 6.6x targeting multiplication
    that froze the Leiden resolution, arriving by another route."""
    live = {e.value.lower() for e in existing if e.kind == "account"}
    kept, added = [], 0
    for handle in candidates:
        if handle.lower() in live:
            kept.append(handle)
        elif added < max_new:
            kept.append(handle)
            added += 1
    return kept


def flagged_story_keywords(
    con: duckdb.DuckDBPyConnection,
    stories_view: str,
    min_index: float = STORY_FLAG_MIN_INDEX,
) -> list[str]:
    """Keywords + hashtags of the latest flagged-stories run whose
    story_suspicion_index clears `min_index`. These are promoted as targeted
    search terms so the collector chases a suspicious story (Phase 4 handoff)."""
    try:
        rows = con.sql(
            f"""
            WITH latest_run AS (
                SELECT * FROM {stories_view}
                QUALIFY dense_rank() OVER (ORDER BY computed_at DESC) = 1
            ), flagged AS (
                SELECT * FROM latest_run WHERE story_suspicion_index >= {float(min_index)}
            )
            SELECT DISTINCT term FROM (
                SELECT unnest(hashtags) AS term FROM flagged
                UNION ALL
                SELECT unnest(keywords) AS term FROM flagged
            )
            WHERE term IS NOT NULL AND length(trim(term)) > 0
            """
        ).fetchall()
    except duckdb.Error:
        return []  # no stories persisted yet
    return [r[0] for r in rows]


def refresh_entries(
    existing: list[DynamicEntry],
    keywords: list[str] | list[tuple[str, int]],
    accounts: list[str],
    max_keywords: int = DYNAMIC_MAX_KEYWORDS,
    max_accounts: int = DYNAMIC_MAX_ACCOUNTS,
    expiry_days: int = DYNAMIC_EXPIRY_DAYS,
    now: datetime | None = None,
    sources: dict[str, str] | None = None,
) -> list[DynamicEntry]:
    """Merge fresh promotions into the state: confirm still-active entries,
    add new ones, drop expired ones, then cap each kind.

    `keywords` may be plain strings or (value, strength) pairs; strength is the
    24h burst count and breaks the cap's ties. Sorting on `last_confirmed`
    alone left every entry confirmed this pass tied, so the cap resolved on
    insertion order - incumbents first - and burst magnitude was discarded
    before it could decide anything."""
    now = now or datetime.now(timezone.utc)
    now_iso = now.isoformat()
    cutoff = now - timedelta(days=expiry_days)
    sources = sources or {}
    by_key = {(e.kind, e.value.lower()): e for e in existing}

    keyword_pairs = [(k, 0) if isinstance(k, str) else k for k in keywords]
    account_pairs = [(a, 0) for a in accounts]

    for kind, values in (("keyword", keyword_pairs), ("account", account_pairs)):
        for v, strength in values:
            key = (kind, v.lower())
            if key in by_key:
                by_key[key].last_confirmed = now_iso
                by_key[key].strength = strength
            else:
                by_key[key] = DynamicEntry(
                    value=v,
                    kind=kind,
                    source=sources.get(v, "hashtag-burst" if kind == "keyword" else "coordination-cluster"),
                    added_at=now_iso,
                    last_confirmed=now_iso,
                    strength=strength,
                )

    alive = [
        e for e in by_key.values() if datetime.fromisoformat(e.last_confirmed) > cutoff
    ]
    out: list[DynamicEntry] = []
    for kind, cap in (("keyword", max_keywords), ("account", max_accounts)):
        pool = sorted(
            (e for e in alive if e.kind == kind),
            key=lambda e: (e.last_confirmed, e.strength),
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
    Returns (bursting, z, posts_last_hour).

    Every hour is counted at the SAME AGE as the latest one: only posts first
    collected by the end of that hour plus however long the latest hour has
    been over. Counted on everything held, the latest hour was always the least
    collected - search, snowball and hydration keep adding posts to an hour for
    12 hours and more after it ends - so z sat negative whatever X was doing.
    Measured 2026-09-22 against live R2: raw z -0.84 (latest 392 against a
    prior-48h mean of 797); restricted to posts seen within an hour of
    creation, +1.04; hours aged 12h compared on posts seen within 12h, -0.24.
    One 09-21 hour went from 88 posts at one hour old to 523.

    Age-matching removes the bias, not the dependence on collection cadence:
    an hour that no posts pass reached while it was young counts low here too.

    A hash aggregate on the post id rather than `QUALIFY row_number()`. A
    post's `created_at` does not change, so its earliest snapshot and its
    latest agree, and the first-seen time is the minimum by definition."""
    rows = con.sql(
        f"""
        WITH lp AS (
            SELECT min(created_at) AS created_at, min(collected_at) AS first_seen
            FROM {posts_view}
            -- Partition pruning; see `bursting_hashtags`. This one runs once per
            -- cycle purely to decide whether to skip a <=300s cooldown, so an
            -- unpruned full-corpus scan here cost more than the sleep it saved.
            WHERE dt >= current_date - INTERVAL 3 DAY
            GROUP BY platform, platform_post_id
        )
        SELECT count(*) FILTER (
            WHERE first_seen <= date_trunc('hour', created_at) + INTERVAL 1 HOUR
                                + (now() - date_trunc('hour', now()))
        ) AS n
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
    stories_view: str | None = None,
    hatespeech_view: str | None = None,
    state_path: Path = DYNAMIC_TARGETS_PATH,
    dry_run: bool = False,
    relevance_view: str | None = None,
    scores_view: str | None = None,
    v2_enabled: bool = V2_PROMOTION_ENABLED,
    v2_max_new: int = V2_PROMOTION_MAX_NEW_PER_PASS,
) -> list[DynamicEntry]:
    """One promotion pass: compute candidates, refresh the state file, return
    the live entries. `dry_run` computes without saving. When `stories_view` is
    given, flagged-story keywords/hashtags join the hashtag bursts as promoted
    keywords, tagged source="story-flag". `hatespeech_view` enables the
    Kenya-relevance gates on both account and keyword promotion."""
    keywords = bursting_hashtags(con, posts_view, hatespeech_view=hatespeech_view)
    accounts = cluster_accounts(
        con, clusters_view, authors_view,
        posts_view=posts_view, hatespeech_view=hatespeech_view,
        relevance_view=relevance_view,
    )
    sources: dict[str, str] = {}
    existing = load_state(state_path)
    if v2_enabled and scores_view is not None:
        ranked = v2_accounts(con, scores_view, authors_view)
        v2 = cap_new_accounts(ranked, existing, v2_max_new)
        log.info(
            "v2 promotion: %d ranked -> %d kept (at most %d new per pass)",
            len(ranked), len(v2), v2_max_new,
        )
        for handle in v2:
            sources[handle] = "v2-centrality"
        seen = {a.lower() for a in accounts}
        accounts = accounts + [h for h in v2 if h.lower() not in seen]
    if stories_view is not None:
        story_kw = flagged_story_keywords(con, stories_view)
        seen = {k for k, _ in keywords}
        for kw in story_kw:
            sources[kw] = "story-flag"
        # Flagged-story terms carry no burst count. They rank above an ordinary
        # burst of the same age on purpose: a story cleared STORY_FLAG_MIN_INDEX,
        # which is a stronger claim than volume.
        keywords = keywords + [(kw, DYNAMIC_HASHTAG_MIN_COUNT) for kw in story_kw if kw not in seen]
    entries = refresh_entries(existing, keywords, accounts, sources=sources)
    if not dry_run:
        save_state(entries, state_path)
    for e in entries:
        log.info("dynamic target: %s %r (%s, confirmed %s)", e.kind, e.value, e.source, e.last_confirmed)
    return entries

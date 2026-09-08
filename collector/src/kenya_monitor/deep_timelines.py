"""Resumable, prioritised deep-timeline collection for the accounts v2 surfaced.

Measured on snapshot `2026-09-05-promotion-off`: 331,138 authors, of which
**251,171 (75.9%) have exactly one post** and 287,244 (86.7%) have two or fewer.
Mean 3.2 posts per author. Only **79,967 clear v2's 2-entity activity floor**
and only 8,047 have 20 or more posts.

That one fact explains most of v2's behaviour on this corpus: 54.8% of
co_retweet users have exactly one entity, the activity floor discards 58% of
accounts, 238,197 users fell below the text-similarity floor, and the fused
graph fragments. v2 needs per-account behavioural history and we hold a wide,
one-post-deep sample. Keyword search cannot fix it - the timeline endpoint
reaches ~3,200 posts per account, far past the 14-day search horizon - which is
what makes this the one thing collection can do that search structurally
cannot.

## THE FEEDBACK ARTEFACT - read this before reading any v2 output

**Deep-timelining accounts BECAUSE v2 surfaced them inflates their own future
centrality.** More posts means more entities, more chances to match, more edges,
higher centrality - whether or not the account is coordinated. v1 hit exactly
this and mitigated it twice over: quarantined partitions, and ranking
count-based score components WITHIN observation-volume strata, so pulling more
of an account's timeline moves it to a different bucket rather than up the
ranking (`hate_signal.VOLUME_BUCKETS`).

`type=deep_timeline` is the first mitigation and it only handles provenance:
targeted types leave every prevalence denominator, but coordination reads every
type, which is the entire point of collecting this. The targeting strata below
are the second, applied to selection rather than to scoring - a deepened
account leaves the stratum that made it a candidate.

Neither is a substitute for the check. **Any v2 re-run after a bulk pass MUST
report whether deep-timelined accounts rose in rank, because that would be the
artefact and not a finding.** To make that a join rather than archaeology, every
deepened account is recorded to `deep_timelines/` with its pre-treatment state -
`user_id`, `rank_value`, `stratum`, `held_posts_before`, `held_entities_before`,
`posts_written`, `depth` - keyed on `user_id` so it joins straight onto
`coord2/platform=x/kind=scores`. See `storage.DEEP_TIMELINE_RUN_SCHEMA`.

## Targeting: strata by held volume, centrality within stratum

Cost per account is near-constant: ~10 requests at depth 200, 20 posts a page,
regardless of what we already hold. Benefit is not. An account holding one post
has a degenerate one-hot vector - two such users acting on the same object have
cosine 1.0 whatever the IDF weight, which is precisely why the floor exists -
so 200 fetched posts turn it from unscorable into scorable. An account already
holding 200 posts can at best double its history, and its vector barely moves.

So selection is stratified on held post count and ranked by centrality inside
each stratum, strata ascending:

| stratum | held posts | why it sits here |
|---|---|---|
| 0 | `< MIN_ENTITIES` (2) | below v2's floor: discarded from every bipartite trace today, and 75.9% of the corpus |
| 1 | 2 .. `thin_posts - 1` (19) | clears the floor, no real history |
| 2 | `thin_posts` .. `depth - 1` | has history, not saturated |
| 3 | `>= depth` | saturated for this depth |

Strata rather than a blended score - `centrality / log(1 + n_posts)` and its
relatives weight two incommensurable quantities by a constant nobody can
defend. Every boundary comes from something measured: 2 is
`kma.coord2.MIN_ENTITIES_PER_USER`, the floor in the code; 20 is where this
corpus's depth distribution breaks (8,047 of 331,138 authors hold 20 or more
posts, the 97.6th percentile); `depth` is the saturation point of this pass by
construction. Inside a stratum the ranking is pure centrality, which is the
only ordering here we actually believe.

**Stratum 0 is empty for the primary target set, and that is why stratum 1
exists.** v2's activity floor is applied BEFORE centrality
(`similarity_network` calls `min_activity`, then `detect` scores the fused
graph), so an account holding one post has fewer than 2 entities in every trace
and cannot appear in a persisted `kind=scores` run at all. Two consequences,
both easy to get wrong:

1. A pass targeting persisted v2 scores **cannot move the corpus-wide
   floor-clearing count of 79,967**, because every one of its targets already
   clears it. Measured on a synthetic corpus reproducing the real depth
   distribution: 500 of 500 targets in stratum 1 or above, 0 below the floor.
   What it buys those accounts is vector DENSITY - the ability to adjudicate a
   ranking built on 2-to-19 observations - not floor crossings.
2. Without the 20-post boundary the strata would collapse to a plain centrality
   ranking on the real target set, which is exactly what they exist to prevent.
   The same synthetic run put a 39-post account above a 2-post one under a
   three-stratum rule.

Moving the 79,967 figure needs the accounts v2 DISCARDED, which is a different
target set with no centrality to rank it by. `runner.census_discovered_handles`
is the precedent and the argument: selection there is `ORDER BY random()`
precisely so that collection is exogenous to the outcome being measured. That is
a separate pass, not a flag on this one.

The two objectives agree, which is the argument for this order rather than the
reverse. Coverage: only stratum 0 can move the floor-clearing count at all.
Epistemics: a high-centrality account with one post is ranked high BECAUSE of
the single entity the floor distrusts, so it is simultaneously the least
trustworthy row in the ranking and the cheapest one to settle. Deepening it
either corroborates or refutes; deepening a 500-post account does neither.

**Known imprecision, stated rather than hidden.** Strata use held POSTS while
v2's floor counts distinct ENTITIES per trace. Five retweets of one object are
five posts and one co_retweet entity, so such an account sits in stratum 1
while being below the floor in the trace that carries 47.4% of our edges. Posts
are used anyway because they are the quantity `depth` directly increases, they
are trace-agnostic, and they are what the plan's 79,967 figure counts
(331,138 - 251,171 = 79,967, i.e. authors with two or more posts). Distinct
co-retweet entities are measured and reported beside it in `--dry-run` and
`--status` so the gap is visible, and both are recorded per account in
`deep_timelines/`.

## What the pass reports

Accounts fetched is not the quantity anyone cares about. Two coverage numbers
are, and they differ by three orders of magnitude:

- **Within the target set.** A persisted v2 run is the top 500 by centrality
  (`coord2_run.run(top=500)`), so a bounded pass can move a large share of the
  accounts v2 actually ranked over the floor.
- **Corpus-wide.** The same pass moves 79,967 by at most `--limit`. 50 accounts
  is +0.06%. Anyone quoting the corpus figure as the outcome of this command has
  misunderstood what it is for.

Both are printed, target-set first.

## Provenance

Rows land in `posts/type=deep_timeline`, a TARGETED type, never the BASELINE
`timeline` one. Writing per-account histories into `type=timeline` would shift
baseline composition far harder than the 2026-08-06 conversation widening did,
and that widening is what made the raw toxicity series unpublishable: it moved
the mix from 70.7% search / 12.1% replies to 14.3% / 72.6% while replies carry
3.2x the hate rate. A single depth-200 account contributes more rows than the
median baseline day contributes for its whole author population.

## Resumability

Two mechanisms, because neither is sufficient alone:

1. **The R2 partition**, via `deepened_expr`. Posts in
   `type=deep_timeline` collected inside the TTL are evidence the account was
   deepened, so this self-heals if the local ledger is lost - the reason
   `census.censused_expr` works the same way. Applied in the candidate SQL,
   before the LIMIT: `census.threaded_expr` documents what happens otherwise,
   a deterministic ranking re-picking completed work every pass (495 selected,
   10-35 fetched).
2. **The ledger** at `state/deep_timeline.json`, for the outcomes R2 can never
   learn about. An account that returned nothing writes no post row, so no
   anti-join can ever exclude it, and selection is deterministic: without the
   ledger it occupies the same slot at the head of the ranking on every pass
   forever. That is how the snowball hydrate arm starved.

Unlike the parent backfill there is NO self-draining anti-join here. A parent
stops qualifying once its post row exists; an account always has posts, so
depth is a TTL question rather than a done/not-done one, and the ledger is
load bearing rather than insurance.

`no_posts` is terminal, `failed` is not. A by-id fetch cannot tell suspended
from protected from genuinely empty - all three yield nothing - and none of them
become fetchable on this project's timescale. A raised request is about our side
(rate limit, proxy, transport), so it retries, bounded.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pyarrow as pa

from kenya_monitor.collectors.base import Collector, Post
from kenya_monitor.config import (
    DEEP_TIMELINE_DEPTH,
    DEEP_TIMELINE_FALLBACK_TARGETS,
    DEEP_TIMELINE_FLUSH_EVERY,
    DEEP_TIMELINE_LIMIT,
    DEEP_TIMELINE_LOOKBACK_DAYS,
    DEEP_TIMELINE_MAX_ATTEMPTS,
    DEEP_TIMELINE_REFRESH_DAYS,
    DEEP_TIMELINE_STATE_PATH,
    DEEP_TIMELINE_THIN_POSTS,
)
from kenya_monitor.storage import Storage

log = logging.getLogger("kenya_monitor")

TARGET_TYPE = "deep_timeline"

# `kma.coord2.MIN_ENTITIES_PER_USER`, duplicated rather than imported: the
# collector takes no dependency on the analysis package. It is the boundary of
# targeting stratum 0, so if the analysis side moves its floor this has to move
# with it or the pass prioritises against a threshold that no longer exists.
MIN_ENTITIES = 2

SOURCE_SCORES = "coord2_scores"
SOURCE_SUSPICION = "suspicion"

STATUS_OK = "ok"
STATUS_NO_POSTS = "no_posts"
STATUS_FAILED = "failed"


@dataclass(slots=True)
class DeepTarget:
    """One selected account, with the pre-treatment state that justified it.

    Carried through the pass rather than re-derived, because it is the covariate
    set the feedback-artefact check needs: "did deepened accounts rise in rank"
    is only answerable against what they held before being deepened."""

    user_id: str
    rank_value: float = 0.0
    rank_metric: str = "centrality"
    source: str = SOURCE_SCORES
    source_run: str = ""
    held_posts: int = 0
    held_entities: int = 0
    stratum: int = 0


@dataclass(slots=True)
class DeepEntry:
    """One account's outcome.

    `held_posts_before` and `posts` are both kept so the ledger can answer the
    only question that matters about a finished pass - how many accounts it
    moved over the floor - without re-reading R2.

    Written indented, unlike `parent_backfill`'s ledger. That one is compact
    because it tracks 167,220 ids (19 MB, 260 MB peak RSS); this one tracks the
    target set, which is 500 accounts for a persisted v2 run, so readability
    costs nothing worth having."""

    fetched_at: str
    status: str = STATUS_OK
    attempts: int = 0
    posts: int = 0
    held_posts_before: int = 0
    held_entities_before: int = 0
    rank_value: float = 0.0
    stratum: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(path: Path = DEEP_TIMELINE_STATE_PATH) -> dict[str, DeepEntry]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {
        uid: DeepEntry(**entry) for uid, entry in (raw.get("entries") or {}).items()
    }


def save_state(
    entries: dict[str, DeepEntry], path: Path = DEEP_TIMELINE_STATE_PATH
) -> None:
    """Write via a temp file + rename, so an interrupted run leaves the previous
    ledger intact rather than a truncated one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "updated_at": _now_iso(),
                "entries": {k: asdict(v) for k, v in entries.items()},
            },
            indent=2,
        )
    )
    os.replace(tmp, path)


def is_due(
    entry: DeepEntry | None,
    refresh_days: int = DEEP_TIMELINE_REFRESH_DAYS,
    now: datetime | None = None,
    max_attempts: int = DEEP_TIMELINE_MAX_ATTEMPTS,
) -> bool:
    """Never deepened -> due. Otherwise it depends on WHY, and the reasons are
    not interchangeable.

    - `no_posts`: the id resolved to nothing fetchable. Suspended, protected or
      empty, indistinguishable through a by-id fetch and none of them change on
      this project's timescale. TERMINAL - retrying it forever occupies the same
      slot at the head of a deterministic ranking, which is how the snowball
      hydrate arm starved before `hydrate:` keys were added.
    - `failed`: the request raised - rate limit, proxy, transport. About our
      side rather than the account, so it retries, bounded by `max_attempts`,
      and only once the TTL has lapsed. Not immediately: `follow_crawl.is_due`
      records what treating a failure as due-now costs, which is unresolvable
      accounts occupying the head of the queue on every pass.
    - `ok`: due again after `refresh_days`. This is the whole point of the TTL -
      a second pass extends coverage to accounts never deepened rather than
      re-paying ~10 requests for a history it already holds. At a fixed depth a
      refresh only adds what the account posted since last time.
    """
    if entry is None:
        return True
    if entry.status == STATUS_NO_POSTS:
        return False
    if entry.status == STATUS_FAILED and entry.attempts >= max_attempts:
        return False
    now = now or datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(entry.fetched_at) < now - timedelta(days=refresh_days)
    except ValueError:
        return True


def not_due_ids(
    entries: dict[str, DeepEntry],
    refresh_days: int = DEEP_TIMELINE_REFRESH_DAYS,
    max_attempts: int = DEEP_TIMELINE_MAX_ATTEMPTS,
) -> list[str]:
    """Accounts the ledger says must not be selected on this pass."""
    return [
        uid
        for uid, e in entries.items()
        if not is_due(e, refresh_days, max_attempts=max_attempts)
    ]


def timeline_summary(
    entries: dict[str, DeepEntry],
    refresh_days: int = DEEP_TIMELINE_REFRESH_DAYS,
    max_attempts: int = DEEP_TIMELINE_MAX_ATTEMPTS,
) -> dict[str, int | float | str | None]:
    """What the ledger has bought so far, in coverage terms.

    `floor_cleared` is the quantity of record: accounts that held fewer than
    MIN_ENTITIES posts and now hold at least that many. `posts_per_account` is
    the derivative worth watching - the strata put the thinnest accounts first,
    so a pass that starts returning few posts per account is a pass whose
    targets are dormant rather than one that is going well."""
    if not entries:
        return {
            "tracked": 0,
            "ok": 0,
            "no_posts": 0,
            "failed": 0,
            "due_now": 0,
            "posts": 0,
            "posts_per_account": 0.0,
            "floor_cleared": 0,
            "latest_fetch": None,
        }
    ok = [e for e in entries.values() if e.status == STATUS_OK]
    posts = sum(e.posts for e in ok)
    return {
        "tracked": len(entries),
        "ok": len(ok),
        "no_posts": sum(e.status == STATUS_NO_POSTS for e in entries.values()),
        "failed": sum(e.status == STATUS_FAILED for e in entries.values()),
        "due_now": sum(
            is_due(e, refresh_days, max_attempts=max_attempts) for e in entries.values()
        ),
        "posts": posts,
        "posts_per_account": round(posts / len(ok), 1) if ok else 0.0,
        "floor_cleared": sum(
            1
            for e in ok
            if e.held_posts_before < MIN_ENTITIES
            and e.held_posts_before + e.posts >= MIN_ENTITIES
        ),
        "latest_fetch": max(e.fetched_at for e in entries.values()),
    }


def deepened_expr(
    con: duckdb.DuckDBPyConnection,
    deep_posts_view: str | None,
    column: str,
    refresh_days: int = DEEP_TIMELINE_REFRESH_DAYS,
) -> str:
    """A boolean SQL expression: was `column` deep-timelined inside the TTL?

    R2 is the source of truth rather than the local JSON state, so this
    self-heals if that state is lost - `census.censused_expr`'s reasoning, and
    the same shape.

    `column` MUST be table-qualified, for the same reason: the correlated
    subquery reads the posts view, and a bare outer column that also exists on
    the inner side binds to the inner table, making the predicate trivially
    true for every row so that every candidate is excluded and the pass stalls
    completely.

    EXISTS, not IN: under three-valued logic one NULL `author_id` on the inner
    side makes `NOT IN` evaluate to NULL for every candidate, and the selector
    silently returns nothing - the same total stall reached another way.

    "FALSE" when the partition does not exist yet, so a first run selects
    normally instead of selecting nothing.
    """
    if column and "." not in column:
        raise ValueError(
            f"column must be table-qualified to correlate correctly, got {column!r}"
        )
    if not deep_posts_view:
        return "FALSE"
    try:
        con.sql(f"SELECT 1 FROM {deep_posts_view} LIMIT 1").fetchall()
    except duckdb.Error as exc:
        # Logged rather than swallowed. An absent partition is expected on a
        # first run, but any other fault here disables the self-healing half of
        # resumability and the pass would look healthy while re-picking work.
        # The ledger still bounds the damage, which is why this degrades rather
        # than raising - unlike `_score_targets`, where the fallback would
        # silently substitute a different target set.
        log.info("deep timelines: no deep_timeline partition to exclude (%s)", exc)
        return "FALSE"
    return f"""EXISTS (
        SELECT 1 FROM {deep_posts_view} d
        WHERE d.author_id = {column}
          AND d.collected_at > now() - INTERVAL {int(refresh_days)} DAY
    )"""


def _score_targets(
    con: duckdb.DuckDBPyConnection,
    scores_view: str,
    *,
    min_kenya_share: float | None,
    stats: dict | None,
) -> bool:
    """Stage the LATEST persisted v2 run as `_dt_targets`. False if there is none.

    Latest run only, never the union. Each object under `kind=scores` is a
    complete ranking of its own - `coord2_run.run` sorts by centrality and takes
    the top `top` - so unioning two runs mixes two rankings and produces an
    order that belongs to neither. `computed_at` is the discriminator rather
    than the `dt=` partition, because two runs can share a day.

    Tiny relation (500 rows per run), so a MAX over it costs nothing and does
    not need the staging discipline the corpus scans do.

    `IOException` only, never `duckdb.Error`. The broad catch was here first and
    it silently substituted the suspicion fallback for any fault at all - a
    schema change, a corrupt object, a permissions failure - which is the worst
    possible outcome, because the pass then reports a healthy run against a
    target set nobody chose. An absent prefix is the one recoverable case.

    `max(computed_at)` is cast to VARCHAR rather than fetched as a timestamp:
    DuckDB converts a tz-aware timestamp through `pytz`, which is not a
    dependency of this project, so pulling one into Python raises
    `InvalidInputException` on a container that does not happen to have it.
    """
    try:
        n_scores, latest = con.sql(
            f"SELECT count(*), CAST(max(computed_at) AS VARCHAR) FROM {scores_view}"
        ).fetchone()
    except duckdb.IOException:
        log.info("deep timelines: no persisted v2 scores; falling back to suspicion")
        return False
    if not n_scores or latest is None:
        return False

    kenya_filter = (
        f"AND kenya_share >= {float(min_kenya_share)}" if min_kenya_share is not None else ""
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _dt_targets AS
        SELECT CAST(user_id AS VARCHAR) AS user_id,
               max(centrality) AS rank_value,
               max(filename) AS source_run
        FROM {scores_view}
        WHERE computed_at = (SELECT max(computed_at) FROM {scores_view})
          AND user_id IS NOT NULL {kenya_filter}
        GROUP BY 1
        """
    )
    if stats is not None:
        stats["scores_computed_at"] = str(latest)
        stats["min_kenya_share"] = min_kenya_share
    return bool(con.sql("SELECT count(*) FROM _dt_targets").fetchone()[0])


def _suspicion_targets(
    con: duckdb.DuckDBPyConnection,
    authors_view: str,
    posts_view: str,
    *,
    pool: int,
    lookback_days: int | None,
) -> None:
    """Stage a suspicion-ranked fallback as `_dt_targets`.

    So the command works before a v2 pass has ever run. It is a WEAKER target
    set and not a substitute: suspicion is a bot-likeness heuristic over profile
    and duplicate-text features (`kma.authenticity`'s SQL mirror) and says
    nothing about acting together, which is the signal depth is being collected
    for. It is also windowed to SUSPICION_LOOKBACK_DAYS and its materialisation
    measured 1,444s on pi0, so a pass on this path is dominated by its own
    selection query.

    `pool` matches the size of a persisted v2 run (top 500) so the fallback
    offers the same triage budget rather than an unbounded one.
    """
    from kenya_monitor import suspicion

    table = (
        suspicion.materialise(con, authors_view, posts_view, lookback_days)
        if lookback_days
        else suspicion.materialise(con, authors_view, posts_view)
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _dt_targets AS
        SELECT CAST(user_id AS VARCHAR) AS user_id,
               max(suspicion) AS rank_value,
               '' AS source_run
        FROM {table}
        WHERE user_id IS NOT NULL AND suspicion IS NOT NULL
        GROUP BY 1
        ORDER BY rank_value DESC
        LIMIT {int(pool)}
        """
    )


def candidate_accounts(
    con: duckdb.DuckDBPyConnection,
    posts_view: str,
    *,
    limit: int,
    depth: int = DEEP_TIMELINE_DEPTH,
    thin_posts: int = DEEP_TIMELINE_THIN_POSTS,
    scores_view: str | None = None,
    authors_view: str | None = None,
    deep_posts_view: str | None = None,
    blocked: Sequence[str] = (),
    refresh_days: int = DEEP_TIMELINE_REFRESH_DAYS,
    lookback_days: int | None = DEEP_TIMELINE_LOOKBACK_DAYS,
    min_kenya_share: float | None = None,
    fallback_pool: int = DEEP_TIMELINE_FALLBACK_TARGETS,
    stats: dict | None = None,
) -> list[DeepTarget]:
    """The accounts to deepen, thinnest stratum first and centrality within it.

    See the module docstring for the strata and the argument for their order.

    This runs on pi0 inside a 600 MB DuckDB limit in a 1 GB container, so it
    obeys the four collector-query rules (docs/OBJECTIVES.md C8):

    - **windowed**: `lookback_days` prunes on `dt` when set. Default is the
      whole corpus, because the held-post count that decides an account's
      stratum is a claim about its whole history here - a windowed count calls
      a long-held account thin and deepens it again.
    - **projected**: two columns per scan, never `SELECT *`. Carrying `text`
      through a dedup measured 569s against 16.7s.
    - **spillable**: hash aggregates and hash joins only. No window function
      and no `count(DISTINCT ...)`: an inner DISTINCT collapses re-collected
      snapshots and the outer aggregate counts them, which is
      `suspicion._beh_sql`'s shape. `platform_post_id` ->
      (`author_id`, `repost_of_id`) is immutable for a given post id, which is
      what makes that dedup equivalent to a latest-snapshot window without the
      sort. Nothing post-level is materialised, because a DuckDB in-memory temp
      table counts against `memory_limit` and CANNOT spill - see the measured
      table beside the two activity stages below.
    - **staged**: each aggregate is materialised before the next reads it. A
      query whose every stage fits can still OOM when DuckDB runs the stages
      concurrently and their peaks add - measured on pi0 2026-09-02, three
      stages of 569s, 261s and 1,248s each fitting alone and the fused form
      still OOMing.

    No materialised relation is larger than distinct authors (331,138 on the
    2026-09-05 snapshot); nothing scales with corpus rows (39,183,192) or with
    distinct posts (1,045,318). Measured end to end on a synthetic glob of
    39,387,988 rows reproducing that shape, threads=2: peak RSS 274 MB at
    `memory_limit=600MB` and 280 MB at 300MB, and it still completes at 100MB
    by spilling. (Ignore RSS in the spilling runs - `ru_maxrss` counts
    file-backed spill pages, so it rises to 418 MB while the DuckDB budget is
    150MB.)

    That covers memory only. pi0's real cost is dominated by R2 reads, which
    cannot be measured from a worktree with no `.env`, so a bounded pass on pi0
    is still outstanding.
    """
    source = SOURCE_SCORES
    if not (scores_view and _score_targets(
        con, scores_view, min_kenya_share=min_kenya_share, stats=stats
    )):
        if not authors_view:
            raise ValueError(
                "no v2 scores available and no authors_view for the suspicion fallback"
            )
        source = SOURCE_SUSPICION
        _suspicion_targets(
            con, authors_view, posts_view, pool=fallback_pool, lookback_days=lookback_days
        )

    dt_prune = (
        f"AND dt >= current_date - INTERVAL {int(lookback_days)} DAY" if lookback_days else ""
    )
    # Two scans of the glob, each collapsing its own DISTINCT straight into its
    # aggregate, rather than one scan materialising the post-level relation and
    # both aggregates reading it. Measured on a synthetic 39,387,988-row glob
    # reproducing the real 331,138 / 251,171 / 1,036,526 shape, threads=2:
    #
    #   shape      600MB      300MB      150MB      80MB
    #   one scan   299 MB     279 MB     OOM        OOM
    #   two scan   235 MB     238 MB     ok (spill) OOM
    #
    # The one-scan form is the cheaper R2 read and the worse failure mode. A
    # DuckDB in-memory temp table counts against `memory_limit` and CANNOT
    # spill, so materialising 1,036,526 rows of three strings is a hard floor on
    # the pass's peak - and pi0 gives DuckDB 600 MB inside a 1 GB container that
    # also runs the collector. Pipelining the DISTINCT into the aggregate leaves
    # only the 331,138-row and 151,493-row outputs resident.
    #
    # The second scan is the same trade `parent_backfill._pb_held` takes, for
    # the same reason: within one connection the httpfs buffer usually still
    # holds what the first stage read.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _dt_activity AS
        SELECT author_id, count(*)::BIGINT AS held_posts FROM (
            SELECT DISTINCT platform_post_id, author_id
            FROM {posts_view}
            WHERE author_id IS NOT NULL {dt_prune}
        ) GROUP BY author_id
        """
    )
    # Distinct co-retweet entities, which is what v2's floor actually counts.
    # Reported beside the post count rather than used for the strata - see the
    # module docstring's note on the gap between the two.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _dt_entities AS
        SELECT author_id, count(*)::BIGINT AS held_entities FROM (
            SELECT DISTINCT author_id, repost_of_id
            FROM {posts_view}
            WHERE author_id IS NOT NULL AND repost_of_id IS NOT NULL {dt_prune}
        ) GROUP BY author_id
        """
    )

    # LEFT JOIN, not inner: a target may hold nothing at all. Census-discovered
    # accounts are known only as retweeter ids (168,356 such accounts against
    # 73,876 with posts, measured 2026-08-02), and an account surfaced from a
    # wider snapshot than the current window can also land here. Those are the
    # accounts depth buys the most for, so an inner join would drop exactly the
    # head of the ranking.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _dt_candidates AS
        SELECT t.user_id,
               t.rank_value,
               t.source_run,
               coalesce(a.held_posts, 0) AS held_posts,
               coalesce(e.held_entities, 0) AS held_entities,
               CASE WHEN coalesce(a.held_posts, 0) < {int(MIN_ENTITIES)} THEN 0
                    WHEN coalesce(a.held_posts, 0) < {int(thin_posts)} THEN 1
                    WHEN coalesce(a.held_posts, 0) < {int(depth)} THEN 2
                    ELSE 3 END AS stratum
        FROM _dt_targets t
        LEFT JOIN _dt_activity a ON a.author_id = t.user_id
        LEFT JOIN _dt_entities e ON e.author_id = t.user_id
        """
    )

    if stats is not None:
        _collect_stats(con, stats, depth=depth, source=source)

    # The ledger cannot be expressed as a predicate on collected posts: an
    # account that returned nothing never gains a post row, so no anti-join can
    # learn about it. Registered as a table rather than inlined - Arrow carries
    # the ids for a few MB and an IN-list of thousands is not a query.
    con.register(
        "_dt_blocked",
        pa.table({"user_id": pa.array([str(b) for b in blocked], type=pa.string())}),
    )
    fresh = deepened_expr(con, deep_posts_view, "c.user_id", refresh_days)
    try:
        rows = con.sql(
            f"""
            SELECT c.user_id, c.rank_value, c.source_run,
                   c.held_posts, c.held_entities, c.stratum
            FROM _dt_candidates c
            WHERE NOT EXISTS (
                SELECT 1 FROM _dt_blocked b WHERE b.user_id = c.user_id
            ) AND NOT {fresh}
            -- Thinnest stratum first, centrality within it. `user_id` breaks
            -- ties so the ranking is deterministic and a resumed pass continues
            -- where the ledger left off rather than reshuffling.
            ORDER BY c.stratum, c.rank_value DESC NULLS LAST, c.user_id
            LIMIT {int(limit)}
            """
        ).fetchall()
    finally:
        con.unregister("_dt_blocked")

    metric = "centrality" if source == SOURCE_SCORES else "suspicion"
    out = [
        DeepTarget(
            user_id=str(uid),
            rank_value=float(rank or 0.0),
            rank_metric=metric,
            source=source,
            source_run=str(run or ""),
            held_posts=int(held),
            held_entities=int(ents),
            stratum=int(stratum),
        )
        for uid, rank, run, held, ents, stratum in rows
    ]
    if stats is not None:
        stats["selected"] = len(out)
        stats["selected_below_floor"] = sum(1 for t in out if t.stratum == 0)
        stats["selected_thin"] = sum(1 for t in out if t.stratum == 1)
        stats["selected_saturated"] = sum(1 for t in out if t.stratum == 3)
        stats["selected_posts_held"] = sum(t.held_posts for t in out)
        stats["blocked_by_ledger"] = len(blocked)
        stats["depth"] = int(depth)
        stats["requests_estimate"] = len(out) * max(1, -(-int(depth) // 20))
    return out


def _collect_stats(
    con: duckdb.DuckDBPyConnection, stats: dict, *, depth: int, source: str
) -> None:
    """Corpus-wide and target-set coverage, in the terms v2 cares about.

    Both denominators, because they differ by three orders of magnitude and
    quoting the wrong one misdescribes the whole command: 79,967 of 331,138
    authors clear the floor corpus-wide, and one bounded pass moves that by at
    most `--limit`. What a pass can move is the DEPTH of the accounts v2 ranked,
    and `targets_thin` is the figure to read - `targets_below_floor` is expected
    to be 0 whenever the target set came from persisted v2 scores, because the
    activity floor is applied before centrality.
    """
    authors, clearing = con.sql(
        f"""SELECT count(*), count(*) FILTER (held_posts >= {int(MIN_ENTITIES)})
            FROM _dt_activity"""
    ).fetchone()
    entity_clearing = con.sql(
        f"SELECT count(*) FROM _dt_entities WHERE held_entities >= {int(MIN_ENTITIES)}"
    ).fetchone()[0]
    targets, below, thin, saturated, entity_below = con.sql(
        f"""
        SELECT count(*),
               count(*) FILTER (stratum = 0),
               count(*) FILTER (stratum = 1),
               count(*) FILTER (stratum = 3),
               count(*) FILTER (held_entities < {int(MIN_ENTITIES)})
        FROM _dt_candidates
        """
    ).fetchone()
    stats.update(
        {
            "source": source,
            "depth": int(depth),
            "authors": int(authors),
            "authors_clearing_floor": int(clearing),
            "authors_clearing_entity_floor": int(entity_clearing),
            "targets": int(targets),
            "targets_below_floor": int(below),
            "targets_clearing_floor": int(targets) - int(below),
            "targets_thin": int(thin),
            "targets_saturated": int(saturated),
            "targets_below_entity_floor": int(entity_below),
        }
    )


def _log_coverage(targets: Sequence[DeepTarget], stats: dict) -> None:
    """Announce the pass in coverage, not in accounts.

    Coverage here is a depth claim, not a floor-crossing one: what a pass over
    v2's own ranking buys is history for accounts currently ranked on very few
    observations, and the median held-post count of the selection is the honest
    summary of that."""
    if not targets:
        return
    held = sorted(t.held_posts for t in targets)
    below = sum(1 for t in targets if t.stratum == 0)
    log.info(
        "deep timelines: %d account(s) holding %d..%d posts (median %d), "
        "%d below the %d-post floor, strata %s, ~%d request(s)",
        len(targets),
        held[0],
        held[-1],
        held[len(held) // 2],
        below,
        MIN_ENTITIES,
        sorted({t.stratum for t in targets}),
        stats.get("requests_estimate", 0),
    )


async def collect_deep_timelines(
    collector: Collector,
    storage: Storage,
    *,
    limit: int = DEEP_TIMELINE_LIMIT,
    depth: int = DEEP_TIMELINE_DEPTH,
    state_path: Path = DEEP_TIMELINE_STATE_PATH,
    refresh_days: int = DEEP_TIMELINE_REFRESH_DAYS,
    lookback_days: int | None = DEEP_TIMELINE_LOOKBACK_DAYS,
    max_attempts: int = DEEP_TIMELINE_MAX_ATTEMPTS,
    flush_every: int = DEEP_TIMELINE_FLUSH_EVERY,
    min_kenya_share: float | None = None,
    targets: list[DeepTarget] | None = None,
    stats: dict | None = None,
) -> dict[str, int]:
    """One bounded pass: deepen up to `limit` accounts to `depth` posts each.

    Bounded per invocation and never an unbounded drain. `targets` supplies a
    pre-ranked list instead of running the selection query - used by the tests
    and by `--dry-run`, which must reach the same ranking without issuing a
    request.
    """
    stats = {} if stats is None else stats
    entries = load_state(state_path)
    if targets is None:
        targets = candidate_accounts(
            storage.con,
            storage.posts_view(platform=collector.platform),
            limit=limit,
            depth=depth,
            scores_view=storage.coord2_scores_view(platform=collector.platform),
            authors_view=storage.authors_view(platform=collector.platform),
            deep_posts_view=storage.posts_view(
                platform=collector.platform, target_type=TARGET_TYPE
            ),
            blocked=not_due_ids(entries, refresh_days, max_attempts),
            refresh_days=refresh_days,
            lookback_days=lookback_days,
            min_kenya_share=min_kenya_share,
            stats=stats,
        )
    targets = targets[: max(0, int(limit))]
    _log_coverage(targets, stats)

    counts = {
        "selected": len(targets),
        "deepened": 0,
        "no_posts": 0,
        "failed": 0,
        "posts": 0,
        "authors": 0,
        # The quantity of record. Counted on what came BACK: an account that
        # returned nothing clears no floor, and `held_posts` is what we held
        # before this pass, so the sum is the accounts this pass made scorable.
        "floor_cleared": 0,
    }
    if not targets:
        log.info("deep timelines: no candidates")
        save_state(entries, state_path)
        return counts

    batch: list[Post] = []
    pending: list[dict] = []

    def _flush() -> None:
        """Write posts, then provenance, then checkpoint the ledger - the
        ordering `collect_snowball` documents. A crash before the write leaves
        accounts unmarked and they retry; a crash between write and checkpoint
        refetches them next pass, which is harmless because reads dedup on
        (platform, platform_post_id).

        Provenance last of the two writes because it is the recoverable one: the
        deepened set is re-derivable from `posts/type=deep_timeline` itself, just
        without the pre-treatment covariates."""
        nonlocal batch, pending
        if batch:
            key = storage.write_posts(batch, target_type=TARGET_TYPE)
            if key:
                log.info("deep timelines: wrote %d post(s) -> %s", len(batch), key)
        if pending:
            key = storage.write_deep_timeline_run(pending, platform=collector.platform)
            if key:
                log.info("deep timelines: recorded %d account(s) -> %s", len(pending), key)
        save_state(entries, state_path)
        batch, pending = [], []

    for i, target in enumerate(targets, 1):
        prior = entries.get(target.user_id)
        try:
            got = [
                p
                async for p in collector.deep_timeline(
                    target.user_id,
                    limit=depth,
                    # The reason the pass exists. Plain `user_tweets` omits
                    # replies entirely, and a history with no replies carries no
                    # reply behaviour to trace.
                    include_replies=True,
                )
            ]
        except Exception:
            log.exception("deep timelines: request failed for %s", target.user_id)
            entries[target.user_id] = DeepEntry(
                fetched_at=_now_iso(),
                status=STATUS_FAILED,
                attempts=(prior.attempts if prior else 0) + 1,
                held_posts_before=target.held_posts,
                held_entities_before=target.held_entities,
                rank_value=target.rank_value,
                stratum=target.stratum,
            )
            counts["failed"] += 1
        else:
            status = STATUS_OK if got else STATUS_NO_POSTS
            batch.extend(got)
            entries[target.user_id] = DeepEntry(
                fetched_at=_now_iso(),
                status=status,
                attempts=(prior.attempts if prior else 0) + 1,
                posts=len(got),
                held_posts_before=target.held_posts,
                held_entities_before=target.held_entities,
                rank_value=target.rank_value,
                stratum=target.stratum,
            )
            pending.append(
                {
                    "user_id": target.user_id,
                    "source": target.source,
                    "source_run": target.source_run,
                    "rank_metric": target.rank_metric,
                    "rank_value": target.rank_value,
                    "stratum": target.stratum,
                    "held_posts_before": target.held_posts,
                    "held_entities_before": target.held_entities,
                    "depth": int(depth),
                    "posts_written": len(got),
                    "status": status,
                }
            )
            if got:
                counts["deepened"] += 1
                counts["posts"] += len(got)
                if (
                    target.held_posts < MIN_ENTITIES
                    and target.held_posts + len(got) >= MIN_ENTITIES
                ):
                    counts["floor_cleared"] += 1
            else:
                counts["no_posts"] += 1
        if i % max(1, flush_every) == 0:
            _flush()
    _flush()

    # `_to_post` records each post's author snapshot as a side effect. Deepened
    # accounts are exactly the ones a reader will want a dossier for, and a
    # profile row is what makes `suspicion` and the dossier work at all.
    authors = collector.collected_authors()
    key = storage.write_authors(authors)
    counts["authors"] = len(authors)
    if key:
        log.info("deep timelines: wrote %d author(s) -> %s", len(authors), key)

    stats.update(counts)
    log.info("deep timelines: %s", counts)
    return counts

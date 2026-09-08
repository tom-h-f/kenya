"""Resumable, prioritised backfill of missing retweet parents.

Measured on snapshot `2026-09-05-promotion-off`: 228,272 distinct retweeted
objects, of which 61,052 are held and **167,220 are missing**. `fast_retweet`
needs the original's author and timestamp to place a retweet relative to it, so
with five sixths of the parents absent that trace produced 222 edges against
co_retweet's 371,259 and is effectively dead. Co-retweet entities are also
missing their originals, so nothing downstream can read what was amplified.

Rows land in `posts/type=parent_backfill`, a TARGETED type, NOT the BASELINE
`hydrated` one. 167,220 rows into a baseline partition would shift the corpus
composition further than the 2026-08-06 conversation widening did, and that
widening is what made the raw toxicity series unpublishable: it moved the mix
from 70.7% search / 12.1% replies to 14.3% / 72.6% while replies carry 3.2x the
hate rate. Targeted types are excluded from every prevalence denominator by
`kma.db.latest_posts(scope="baseline")`; coordination reads every type, so v2
still sees this data, which is the entire point of collecting it.

Prioritised, never drained. One `tweet_details` request per id - there is no
batch lookup on the GraphQL path - so the whole backlog is days of pool budget
and the order it is spent in decides whether any of it produces an edge.
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
    PARENT_BACKFILL_FLUSH_EVERY,
    PARENT_BACKFILL_LIMIT,
    PARENT_BACKFILL_LOOKBACK_DAYS,
    PARENT_BACKFILL_MAX_ATTEMPTS,
    PARENT_BACKFILL_OK_RETAIN_HOURS,
    PARENT_BACKFILL_STATE_PATH,
    SNOWBALL_BAND_MAX,
    SNOWBALL_BAND_MIN,
)
from kenya_monitor.storage import Storage

log = logging.getLogger("kenya_monitor")

TARGET_TYPE = "parent_backfill"

STATUS_OK = "ok"
STATUS_NOT_FOUND = "not_found"
STATUS_FAILED = "failed"


@dataclass(slots=True)
class BackfillEntry:
    """One id's outcome. `amplifiers` is kept so the ledger records what the
    request was spent on, not just that it happened - without it a completed
    backfill cannot be checked against the band it claimed to prioritise."""

    fetched_at: str
    status: str = STATUS_OK
    attempts: int = 0
    amplifiers: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(path: Path = PARENT_BACKFILL_STATE_PATH) -> dict[str, BackfillEntry]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {
        oid: BackfillEntry(**entry)
        for oid, entry in (raw.get("entries") or {}).items()
    }


def save_state(
    entries: dict[str, BackfillEntry],
    path: Path = PARENT_BACKFILL_STATE_PATH,
) -> None:
    """Write via a temp file + rename, so an interrupted run leaves the previous
    ledger intact rather than a truncated one.

    Written compact rather than indented, unlike the other state files. Measured
    at 167,220 entries: 19.06 MB compact against 26.25 MB at indent=2, and the
    in-memory dict alone costs 260 MB peak RSS - inside a 1 GB container that
    already gives DuckDB 600 MB, so the file has to be as small as it can be.
    Retention (see `prune_state`) is what actually keeps it off that ceiling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "updated_at": _now_iso(),
                "entries": {k: asdict(v) for k, v in entries.items()},
            },
            separators=(",", ":"),
        )
    )
    os.replace(tmp, path)


def is_retryable(
    entry: BackfillEntry | None,
    max_attempts: int = PARENT_BACKFILL_MAX_ATTEMPTS,
) -> bool:
    """Never attempted -> yes. Otherwise it depends on WHY it is not held, and
    the three reasons are not interchangeable.

    - `ok`: the post is in R2. The candidate query's anti-join against collected
      posts excludes it permanently, so the ledger entry is only insurance for
      the window before an R2 LIST reflects the write.
    - `not_found`: `tweet_details` returned nothing. Deleted, suspended or
      protected, and none of those become fetchable later. Permanent, because
      selection is deterministic: a retried absence occupies the same slot at
      the head of the ranking on every pass forever, which is precisely how the
      snowball hydrate arm starved before `hydrate:` keys were added.
    - `failed`: the request itself raised - rate limit, proxy, transport. That
      is about our side rather than the object, so it retries, bounded."""
    if entry is None:
        return True
    if entry.status == STATUS_OK:
        return False
    if entry.status == STATUS_NOT_FOUND:
        return False
    return entry.attempts < max_attempts


def blocked_ids(
    entries: dict[str, BackfillEntry],
    max_attempts: int = PARENT_BACKFILL_MAX_ATTEMPTS,
) -> list[str]:
    """Ids the ledger says must not be selected again."""
    return [oid for oid, e in entries.items() if not is_retryable(e, max_attempts)]


def prune_state(
    entries: dict[str, BackfillEntry],
    retain_hours: int = PARENT_BACKFILL_OK_RETAIN_HOURS,
) -> dict[str, BackfillEntry]:
    """Drop `ok` entries past the retention window; keep every terminal failure.

    An `ok` id has a post row, so the anti-join owns it from then on and the
    ledger entry is redundant once R2's listing has caught up. A `not_found` id
    never gains a post row, so its ledger entry is the ONLY record that it was
    tried - pruning those re-opens the starvation this module exists to avoid.
    Retention is therefore asymmetric on purpose, and it is what stops the file
    growing to the 19 MB / 260 MB figures in `save_state`."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=retain_hours)
    kept: dict[str, BackfillEntry] = {}
    for oid, e in entries.items():
        if e.status != STATUS_OK:
            kept[oid] = e
            continue
        try:
            if datetime.fromisoformat(e.fetched_at) >= cutoff:
                kept[oid] = e
        except ValueError:
            kept[oid] = e
    return kept


def backfill_summary(entries: dict[str, BackfillEntry]) -> dict[str, int | str | None]:
    if not entries:
        return {
            "tracked": 0,
            "ok": 0,
            "not_found": 0,
            "failed": 0,
            "latest_fetch": None,
            "band_share": 0.0,
        }
    statuses = [e.status for e in entries.values()]
    ok = [e for e in entries.values() if e.status == STATUS_OK]
    in_band = [
        e for e in ok if SNOWBALL_BAND_MIN <= e.amplifiers <= SNOWBALL_BAND_MAX
    ]
    return {
        "tracked": len(entries),
        "ok": len(ok),
        "not_found": sum(s == STATUS_NOT_FOUND for s in statuses),
        "failed": sum(s == STATUS_FAILED for s in statuses),
        "latest_fetch": max(e.fetched_at for e in entries.values()),
        # The check on the whole point of the prioritisation: if this is low,
        # the budget went on objects that cannot produce a coordination edge.
        "band_share": round(len(in_band) / len(ok), 3) if ok else 0.0,
    }


def candidate_parents(
    con: duckdb.DuckDBPyConnection,
    posts_view: str,
    *,
    limit: int,
    blocked: Sequence[str] = (),
    lookback_days: int | None = PARENT_BACKFILL_LOOKBACK_DAYS,
    band_min: int = SNOWBALL_BAND_MIN,
    band_max: int = SNOWBALL_BAND_MAX,
    stats: dict | None = None,
) -> list[tuple[str, int]]:
    """Missing retweet parents with their distinct-amplifier counts, ranked.

    Returns `[(object_id, amplifiers), ...]`, in-band objects first.

    Amplifiers are counted as DISTINCT retweeter accounts in our own corpus, not
    from `repost_count`. The census bands on `repost_count` because it is
    selecting objects to go and fetch retweeters FOR; here the quantity that
    decides whether hydrating an object can ever produce a co-retweet pair is
    how many amplifiers we already hold, and the two diverge badly - a platform
    counter of 40 on an object we saw twice buys nothing.

    This runs on pi0 inside a 600 MB DuckDB limit in a 1 GB container, so it
    obeys the four collector-query rules (docs/OBJECTIVES.md C8):

    - **windowed**: the amplifier scan prunes on `dt` first when
      `lookback_days` is set. `dt` is the COLLECTION date and a post cannot be
      collected before it exists, so the predicate can only prune partitions.
      The held-ids stage is deliberately NOT pruned, for the same reason
      `hot_objects.known` is not: a windowed answer calls every older post
      missing and spends the whole budget refetching R2.
    - **projected**: two columns, never `SELECT *`. Carrying `text` through a
      dedup measured 569s against 16.7s.
    - **spillable**: hash aggregates and hash joins only, no window functions.
      There is no latest-snapshot dedup here and none is needed -
      `repost_of_id` and `author_id` are immutable for a given post id, so
      re-collected snapshots of the same retweet collapse under the DISTINCT.
      That removes the `QUALIFY row_number() OVER (...)` sort that would
      otherwise dominate the pass.
    - **staged**: each aggregate is materialised before the next reads it. A
      query whose every stage fits can still OOM when DuckDB runs the stages
      concurrently and their peaks add.

    The staging also bounds the largest materialised relation by the number of
    distinct retweeted objects (228,272) rather than by corpus rows
    (39,183,192): `_pb_held` is a semi-join against the already-materialised
    parent ids, so it holds at most one row per parent instead of one per
    distinct post in the corpus (1,045,318).
    """
    dt_prune = (
        f"AND dt >= current_date - INTERVAL {int(lookback_days)} DAY"
        if lookback_days
        else ""
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _pb_pairs AS
        SELECT DISTINCT repost_of_id AS oid, author_id
        FROM {posts_view}
        WHERE repost_of_id IS NOT NULL AND author_id IS NOT NULL {dt_prune}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _pb_amps AS
        SELECT oid, count(*)::BIGINT AS amplifiers FROM _pb_pairs GROUP BY oid
        """
    )
    # The pair set is the widest thing this query builds and nothing reads it
    # again. Holding it while the joins below run is what makes the stage peaks
    # add rather than alternate.
    con.execute("DROP TABLE _pb_pairs")
    # A second scan of the glob, and unavoidable without a window function or a
    # materialised 3-column copy of the corpus: the held-ids answer has to cover
    # every partition while the amplifier answer may be windowed. It is a
    # SEMI-join against the already-materialised parent ids, so it holds at most
    # one row per parent (228,272) rather than one per distinct post
    # (1,045,318), and within one connection the httpfs buffer usually still
    # holds what the first stage read.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _pb_held AS
        SELECT DISTINCT p.platform_post_id
        FROM {posts_view} p
        WHERE EXISTS (SELECT 1 FROM _pb_amps a WHERE a.oid = p.platform_post_id)
        """
    )
    # NOT EXISTS, not NOT IN. Under three-valued logic one NULL id on the inner
    # side makes `NOT IN` evaluate to NULL for every candidate and the selector
    # silently returns nothing - the same stall `census.censused_expr` documents.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _pb_missing AS
        SELECT a.oid, a.amplifiers
        FROM _pb_amps a
        WHERE NOT EXISTS (
            SELECT 1 FROM _pb_held h WHERE h.platform_post_id = a.oid
        )
        """
    )

    if stats is not None:
        stats["retweeted_objects"] = con.sql(
            "SELECT count(*) FROM _pb_amps"
        ).fetchone()[0]
        stats["held_parents"] = con.sql(
            "SELECT count(*) FROM _pb_held"
        ).fetchone()[0]
        stats["missing_parents"] = con.sql(
            "SELECT count(*) FROM _pb_missing"
        ).fetchone()[0]
        stats["missing_in_band"] = con.sql(
            f"SELECT count(*) FROM _pb_missing "
            f"WHERE amplifiers BETWEEN {int(band_min)} AND {int(band_max)}"
        ).fetchone()[0]
        stats["band_min"] = int(band_min)
        stats["band_max"] = int(band_max)

    # The ledger cannot be expressed as a predicate on collected posts: an
    # absent object never gains a post row, so the anti-join above can never
    # learn about it. Registered as a table rather than inlined - at 167,220
    # ids an IN-list is not a query, and Arrow carries them for a few MB.
    con.register(
        "_pb_blocked",
        pa.table({"oid": pa.array([str(b) for b in blocked], type=pa.string())}),
    )
    try:
        rows = con.sql(
            f"""
            WITH ranked AS (
                SELECT m.oid, m.amplifiers,
                       CASE
                           WHEN m.amplifiers BETWEEN {int(band_min)} AND {int(band_max)} THEN 0
                           WHEN m.amplifiers < {int(band_min)} THEN 1
                           ELSE 2
                       END AS band_rank
                FROM _pb_missing m
                WHERE NOT EXISTS (
                    SELECT 1 FROM _pb_blocked b WHERE b.oid = m.oid
                )
            )
            SELECT oid, amplifiers FROM ranked
            -- In-band first, then densest-first within the band. An object with
            -- one amplifier yields no pair at all, and a hub is deleted by
            -- `coordination.validated_edges` - 420 of 422 objects censused by
            -- raw popularity were hubs, which is the measured reason the census
            -- is banded at all. Hydrating in id order would spend the whole
            -- budget on objects that cannot produce an edge.
            --
            -- Below the band ranks ahead of above it, and the tail orders
            -- towards the band from both sides: a 2-amplifier object is one
            -- retweeter short of being usable, whereas an object above the cap
            -- is discarded however much is spent on it.
            ORDER BY band_rank,
                     CASE WHEN band_rank = 2 THEN amplifiers ELSE -amplifiers END,
                     oid
            LIMIT {int(limit)}
            """
        ).fetchall()
    finally:
        con.unregister("_pb_blocked")

    if stats is not None:
        stats["selected"] = len(rows)
        stats["selected_in_band"] = sum(
            1 for _, amp in rows if band_min <= amp <= band_max
        )
        stats["blocked_by_ledger"] = len(blocked)
    return [(str(oid), int(amp)) for oid, amp in rows]


def _warn_if_out_of_band(
    candidates: Sequence[tuple[str, int]], band_min: int, band_max: int
) -> None:
    """Say so when the pass has run past the in-band supply.

    The band is a RANK here, not a filter, so a pass larger than the in-band
    population silently continues into objects with one or two amplifiers.
    Those cannot produce a co-retweet pair, and the arithmetic says that state
    arrives early: 228,272 objects share 364,287 retweet rows, so if x objects
    hold 3 or more amplifiers then 3x + (228,272 - x) <= 364,287 and x <= 68,007
    - at most 30% of the population, and far less under any realistic skew.
    An operator who does not see this will read a long green pass as progress."""
    if not candidates:
        return
    out = [amp for _, amp in candidates if not band_min <= amp <= band_max]
    if not out:
        return
    log.warning(
        "parent backfill: %d of %d selected ids are outside the [%d..%d] band "
        "(%d below, %d above) - in-band supply is exhausted, and below-band "
        "objects cannot produce a co-retweet pair",
        len(out), len(candidates), band_min, band_max,
        sum(1 for a in out if a < band_min),
        sum(1 for a in out if a > band_max),
    )


async def backfill_parents(
    collector: Collector,
    storage: Storage,
    *,
    limit: int = PARENT_BACKFILL_LIMIT,
    state_path: Path = PARENT_BACKFILL_STATE_PATH,
    lookback_days: int | None = PARENT_BACKFILL_LOOKBACK_DAYS,
    band_min: int = SNOWBALL_BAND_MIN,
    band_max: int = SNOWBALL_BAND_MAX,
    flush_every: int = PARENT_BACKFILL_FLUSH_EVERY,
    max_attempts: int = PARENT_BACKFILL_MAX_ATTEMPTS,
    candidates: list[tuple[str, int]] | None = None,
    stats: dict | None = None,
) -> dict[str, int]:
    """One bounded pass: hydrate up to `limit` missing parents.

    `candidates` supplies a pre-ranked list instead of running the selection
    query - used by the tests and by `--dry-run`, which must reach the same
    ranking without issuing a request."""
    stats = {} if stats is None else stats
    entries = load_state(state_path)
    if candidates is None:
        candidates = candidate_parents(
            storage.con,
            storage.posts_view(platform=collector.platform),
            limit=limit,
            blocked=blocked_ids(entries, max_attempts),
            lookback_days=lookback_days,
            band_min=band_min,
            band_max=band_max,
            stats=stats,
        )
    candidates = candidates[: max(0, int(limit))]
    _warn_if_out_of_band(candidates, band_min, band_max)

    counts = {
        "selected": len(candidates),
        "hydrated": 0,
        "not_found": 0,
        "failed": 0,
        "authors": 0,
    }
    if not candidates:
        log.info("parent backfill: no candidates")
        save_state(prune_state(entries), state_path)
        return counts

    amps = dict(candidates)
    batch: list[Post] = []
    pending: list[str] = []

    def _flush() -> None:
        """Write FIRST, then mark done, then checkpoint - the ordering
        `collect_snowball` documents. A crash before the write leaves ids
        unmarked and they retry; a crash between write and checkpoint refetches
        them next pass, which is harmless because reads dedup on
        (platform, platform_post_id)."""
        nonlocal batch, pending
        if batch:
            key = storage.write_posts(batch, target_type=TARGET_TYPE)
            counts["hydrated"] += len(batch)
            if key:
                log.info("parent backfill: wrote %d parents -> %s", len(batch), key)
        now_iso = _now_iso()
        for oid in pending:
            entries[oid] = BackfillEntry(
                fetched_at=now_iso, status=STATUS_OK, attempts=1, amplifiers=amps[oid]
            )
        save_state(entries, state_path)
        batch, pending = [], []

    for i, (oid, amp) in enumerate(candidates, 1):
        prior = entries.get(oid)
        try:
            got = [p async for p in collector.hydrate([oid])]
        except Exception:
            log.exception("parent backfill: request failed for %s", oid)
            entries[oid] = BackfillEntry(
                fetched_at=_now_iso(),
                status=STATUS_FAILED,
                attempts=(prior.attempts if prior else 0) + 1,
                amplifiers=amp,
            )
            counts["failed"] += 1
        else:
            if got:
                batch.extend(got)
                pending.append(oid)
            else:
                # Absent, not failed. `tweet_details` returning nothing means
                # deleted, suspended or protected, and the ledger is the only
                # place that can ever be recorded.
                entries[oid] = BackfillEntry(
                    fetched_at=_now_iso(),
                    status=STATUS_NOT_FOUND,
                    attempts=(prior.attempts if prior else 0) + 1,
                    amplifiers=amp,
                )
                counts["not_found"] += 1
        if i % max(1, flush_every) == 0:
            _flush()
    _flush()

    # `_to_post` records the parent's author snapshot as a side effect, and the
    # parent's author is half of what `fast_retweet` needs - a hydrated original
    # with no author row does not revive the trace.
    authors = collector.collected_authors()
    key = storage.write_authors(authors)
    counts["authors"] = len(authors)
    if key:
        log.info("parent backfill: wrote %d authors -> %s", len(authors), key)

    save_state(prune_state(entries), state_path)
    stats.update(counts)
    log.info("parent backfill: %s", counts)
    return counts

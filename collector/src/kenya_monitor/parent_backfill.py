"""Resumable, prioritised backfill of missing retweet parents.

Measured on snapshot `2026-09-05-promotion-off`: 228,272 distinct retweeted
objects, of which 61,052 are held and **167,220 are missing**.

The quantity this pass moves is `fast_retweet` COVERAGE, not ids fetched.
`fast_retweet_traces` times a retweet against `orig.created_at`, so a retweet
row only becomes a candidate trace row once its parent is in the corpus:
**161,146 of 364,287 retweet rows have a held parent, 44%**. That is why the
trace yields 222 edges against co_retweet's 371,259. Report a pass as the
coverage it buys - `retweet_rows_unlocked` in the returned counters - because
ids fetched is not the quantity anyone cares about.

Ranked by **distinct amplifier count DESCENDING, unbanded**. The trace's entity
is the retweeted AUTHOR (`coalesce(rt.retweet_user_id, orig.user_id)`, and
`retweet_user_id` is always NULL because the collector never stored it), so
every retweet row whose parent we hold becomes a candidate trace row regardless
of how many amplifiers that parent has. The quantity to maximise per request is
therefore retweet rows unlocked, which IS the amplifier count.

Deliberately NOT banded, unlike the census. The census bands because
`coordination.validated_edges` discards hubs - 420 of 422 objects censused by
raw popularity were hubs. v2 has no hub cap: TF-IDF down-weighting handles
popular entities, and that is the one place the paper's mechanism works as
advertised. So the reason to band does not apply, and descending-unbanded
naturally orders the 10 hubs first, then the 5,545 in-band objects, then the
two-amplifier objects, then the long tail of 151,215 single-amplifier ones
(90.4% of the backlog) - which is the right order for this objective. Banding
here would have skipped the 10 densest objects in the whole backlog.
`--band-only` remains available as an explicit opt-in, never the default.

**Hydration adds zero `co_retweet` edges.** `co_retweet_traces` reads
`retweet_post_id` off the retweet row; a hydrated parent's own row has a NULL
`retweet_post_id` and contributes no trace. The co-retweet gain is
readability - dossiers can show what was amplified - not edges.

Rows land in `posts/type=parent_backfill`, a TARGETED type, NOT the BASELINE
`hydrated` one. 167,220 rows into a baseline partition would shift the corpus
composition further than the 2026-08-06 conversation widening did, and that
widening is what made the raw toxicity series unpublishable: it moved the mix
from 70.7% search / 12.1% replies to 14.3% / 72.6% while replies carry 3.2x the
hate rate. Targeted types are excluded from every prevalence denominator by
`kma.db.latest_posts(scope="baseline")`; coordination reads every type, so v2
still sees this data, which is the entire point of collecting it.

**Feedback isolation is only half solved by the partition.** A hydrated parent
is a post BY its author, so backfilled-parent authors gain activity and can
newly clear v2's 2-entity minimum-activity floor - the floor that exists
because two users whose only action is the same object have identical one-hot
vectors and a cosine of 1.0 whatever the IDF weight. Selection here is ranked
by in-corpus amplifier count, so the authors who gain most are the ones our
corpus already amplifies most. `type=parent_backfill` keeps these rows out of
every prevalence denominator, but it does nothing about that ranking artefact:
**any v2 re-run after a bulk pass must report whether backfilled-parent authors
rose in rank**, because that would be the artefact and not a finding. Same trap
as deep timelines, reached from the other direction.

Prioritised, never drained. One `tweet_details` request per id - there is no
batch lookup on the GraphQL path - so the whole backlog is days of pool budget
and the order it is spent in decides how much coverage it buys.
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


def backfill_summary(entries: dict[str, BackfillEntry]) -> dict[str, int | float | str | None]:
    """What the ledger has bought so far, in coverage terms.

    `rows_unlocked` is the sum of amplifiers over successfully fetched parents,
    i.e. the retweet rows that became candidate `fast_retweet` trace rows.
    `rows_per_request` is the derivative worth watching: the ranking is
    densest-first, so this decays towards 1.0 and there is little point
    continuing once it gets there. Retention prunes `ok` entries after
    PARENT_BACKFILL_OK_RETAIN_HOURS, so both are a WINDOW on recent passes and
    not a lifetime total."""
    if not entries:
        return {
            "tracked": 0,
            "ok": 0,
            "not_found": 0,
            "failed": 0,
            "latest_fetch": None,
            "rows_unlocked": 0,
            "rows_per_request": 0.0,
        }
    statuses = [e.status for e in entries.values()]
    ok = [e for e in entries.values() if e.status == STATUS_OK]
    unlocked = sum(e.amplifiers for e in ok)
    return {
        "tracked": len(entries),
        "ok": len(ok),
        "not_found": sum(s == STATUS_NOT_FOUND for s in statuses),
        "failed": sum(s == STATUS_FAILED for s in statuses),
        "latest_fetch": max(e.fetched_at for e in entries.values()),
        "rows_unlocked": unlocked,
        "rows_per_request": round(unlocked / len(ok), 2) if ok else 0.0,
    }


def candidate_parents(
    con: duckdb.DuckDBPyConnection,
    posts_view: str,
    *,
    limit: int,
    blocked: Sequence[str] = (),
    lookback_days: int | None = PARENT_BACKFILL_LOOKBACK_DAYS,
    band_only: bool = False,
    band_min: int = SNOWBALL_BAND_MIN,
    band_max: int = SNOWBALL_BAND_MAX,
    stats: dict | None = None,
) -> list[tuple[str, int]]:
    """Missing retweet parents with their distinct-amplifier counts, ranked.

    Returns `[(object_id, amplifiers), ...]`, densest first.

    Ranked by amplifier count DESCENDING and UNBANDED. Each amplifier is one
    retweet row that becomes a candidate `fast_retweet` trace row once the
    parent lands, so amplifier count IS coverage bought per request. See the
    module docstring for why the census's band does not carry over: v2 has no
    hub cap, so the 10 densest objects in the backlog are the best requests
    available rather than the worst. `band_only` restores the banded selection
    as an explicit opt-in.

    Amplifiers are counted as DISTINCT retweeter accounts in our own corpus, not
    from `repost_count`. `repost_count` is the platform's counter and includes
    retweets we never collected, which unlock nothing; the two diverge badly, so
    a counter of 5,000 on an object we saw twice would rank at the head while
    buying two rows.

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
        # Object counts AND row counts. `sum(amplifiers)` is the population of
        # retweet rows a parent would unlock, which is the `fast_retweet`
        # coverage denominator - 161,146 of 364,287 rows had a held parent on
        # 2026-09-05, 44%. Objects fetched is not the quantity anyone cares
        # about, so both are recorded and the coverage share is derived here
        # rather than left to whoever reads the log.
        objects, rows_total = con.sql(
            "SELECT count(*), coalesce(sum(amplifiers), 0) FROM _pb_amps"
        ).fetchone()
        missing_objects, missing_rows, in_band = con.sql(
            f"""
            SELECT count(*), coalesce(sum(amplifiers), 0),
                   count(*) FILTER (
                       amplifiers BETWEEN {int(band_min)} AND {int(band_max)}
                   )
            FROM _pb_missing
            """
        ).fetchone()
        stats["retweeted_objects"] = int(objects)
        stats["retweet_rows"] = int(rows_total)
        stats["held_parents"] = int(objects) - int(missing_objects)
        stats["held_rows"] = int(rows_total) - int(missing_rows)
        stats["missing_parents"] = int(missing_objects)
        stats["missing_rows"] = int(missing_rows)
        stats["missing_in_band"] = int(in_band)
        stats["coverage"] = (
            round((int(rows_total) - int(missing_rows)) / int(rows_total), 4)
            if rows_total
            else 0.0
        )
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
    band_filter = (
        f"AND m.amplifiers BETWEEN {int(band_min)} AND {int(band_max)}"
        if band_only
        else ""
    )
    try:
        rows = con.sql(
            f"""
            SELECT m.oid, m.amplifiers
            FROM _pb_missing m
            WHERE NOT EXISTS (
                SELECT 1 FROM _pb_blocked b WHERE b.oid = m.oid
            ) {band_filter}
            -- Densest first, unbanded. One amplifier is one retweet row that
            -- becomes a candidate `fast_retweet` trace row once the parent
            -- lands, so this maximises coverage bought per request. `oid`
            -- breaks ties so the ranking is deterministic and a resumed pass
            -- continues where the ledger left off rather than reshuffling.
            ORDER BY m.amplifiers DESC, m.oid
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
        stats["selected_rows"] = sum(int(amp) for _, amp in rows)
        stats["band_only"] = bool(band_only)
        stats["blocked_by_ledger"] = len(blocked)
    return [(str(oid), int(amp)) for oid, amp in rows]


def _log_coverage(candidates: Sequence[tuple[str, int]], stats: dict) -> None:
    """Announce the pass in coverage, not in ids.

    `fast_retweet` coverage was 161,146 of 364,287 retweet rows with a held
    parent on 2026-09-05, 44%. What a pass buys is the retweet rows its
    candidates unlock, and because the ranking is densest-first that number
    falls steeply with each pass: 90.4% of the backlog is single-amplifier
    objects, so late passes unlock roughly one row per request. Logging ids
    fetched hides exactly that decay."""
    if not candidates:
        return
    unlocked = sum(amp for _, amp in candidates)
    total = stats.get("retweet_rows") or 0
    held = stats.get("held_rows") or 0
    gain = f", coverage {held / total:.1%} -> {(held + unlocked) / total:.1%}" if total else ""
    log.info(
        "parent backfill: %d ids unlock %d retweet row(s), %d..%d amplifiers%s",
        len(candidates),
        unlocked,
        candidates[-1][1],
        candidates[0][1],
        gain,
    )


async def backfill_parents(
    collector: Collector,
    storage: Storage,
    *,
    limit: int = PARENT_BACKFILL_LIMIT,
    state_path: Path = PARENT_BACKFILL_STATE_PATH,
    lookback_days: int | None = PARENT_BACKFILL_LOOKBACK_DAYS,
    band_only: bool = False,
    band_min: int = SNOWBALL_BAND_MIN,
    band_max: int = SNOWBALL_BAND_MAX,
    flush_every: int = PARENT_BACKFILL_FLUSH_EVERY,
    max_attempts: int = PARENT_BACKFILL_MAX_ATTEMPTS,
    candidates: list[tuple[str, int]] | None = None,
    stats: dict | None = None,
) -> dict[str, int]:
    """One bounded pass: hydrate up to `limit` missing parents, densest first.

    `band_only` restricts selection to `[band_min, band_max]`. Opt-in only: the
    census's reason to band does not apply here (see the module docstring), and
    banding by default would skip the 10 densest objects in the whole backlog.

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
            band_only=band_only,
            band_min=band_min,
            band_max=band_max,
            stats=stats,
        )
    candidates = candidates[: max(0, int(limit))]
    _log_coverage(candidates, stats)

    counts = {
        "selected": len(candidates),
        "hydrated": 0,
        "not_found": 0,
        "failed": 0,
        "authors": 0,
        # The quantity of record. Counted on what came BACK, not on what was
        # selected: an absent parent unlocks nothing.
        "retweet_rows_unlocked": 0,
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
            counts["retweet_rows_unlocked"] += amps[oid]
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

"""Replay harness: score a candidate collection policy against a frozen snapshot.

    from kma.db import connect
    from kma import bench, replay
    con = connect()
    m = bench.load("2026-09-05-promotion-off", con=con)
    times = replay.census_pass_times(m)
    base = replay.replay(con, policy=replay.IncumbentCensus(), manifest=m,
                         times=times, snapshot="2026-09-05-promotion-off")
    cand = replay.replay(con, policy=replay.UnbandedCensus(), manifest=m,
                         times=times, snapshot="2026-09-05-promotion-off")
    print(replay.compare(con, m, base, cand))

Step B1/B2 of `docs/plans/2026-09-05-v2-next-steps.md`. A collection policy is a
function from the state observable at time *t* to a set of fetch requests;
replay asks what a policy would have fetched given only what it could have
known, and how much coordination signal that would have bought.

THE LIMITATION THAT TRAVELS WITH EVERY NUMBER THIS MODULE PRODUCES
==================================================================
The frozen corpus is itself the OUTPUT of the incumbent policy, so replay
systematically favours policies that resemble the incumbent. A policy that
would have fetched objects the incumbent never touched looks worthless here,
because the data to score it does not exist - not because the policy is bad.
`Yield.unscorable_requests` is that quantity made explicit, and `LIMITATION` is
attached to every `ReplayResult` and every `Yield` so a number cannot travel
without it. The retweeter census is the PARTIAL exception: for the objects it
covers, its account x object incidence is near-total, so yield on a re-selection
of already-censused objects is real rather than inferred.

Read `Yield.unscorable_requests` before reading `edges_per_request`. A candidate
whose requests are 90% unscorable has not been measured; it has been silently
scored against absent data.

HOW LEAKAGE IS PREVENTED, since that is the harness's only correctness property
==============================================================================
Two mechanisms, deliberately redundant:

  1. OBJECT level. R2 keys carry their write time (`run=20260905T081903Z`), so
     the snapshot manifest alone says which objects existed at *t* - no data is
     read to establish it. A state at *t* is pinned to exactly those paths, so a
     later object is never even opened.
  2. ROW level. `WHERE collected_at <= t` on top, because the collector flushes
     mid-pass and a single object can straddle *t*.

Object level alone would over-admit; row level alone would read the whole
corpus. And a policy can still leak by construction - by requesting a target it
had no way to know about - so `replay(audit=True)` semi-joins every request
against the visible state and raises `Leakage` rather than scoring it.

WHAT A REQUEST COSTS, and the one number here that conflicts with the docs
=========================================================================
`docs/analysis/census-tuning.md` §2c states `retweeters()` pages at 100, and
prices a hub at ~3 requests and a mid-band object at 1. The installed
twscrape 0.20.1 sends `count: 20` (`twscrape/api.py`, `retweeters_raw`), so on
the code that is actually deployed a 300-retweeter fetch is 15 requests and a
27-retweeter object is 2. That makes §2c's conclusion - banding is cheaper per
object AND more useful - stronger, not weaker, but it also means every
per-request figure in that document is out by up to 5x. `RETWEETERS_PAGE` is a
parameter so the arithmetic can be redone once the true page size is read off
pi0's logs, which is the only place it is observable.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

import duckdb
import pandas as pd

from kma import coord2

log = logging.getLogger(__name__)


LIMITATION = (
    "The frozen corpus is the output of the incumbent policy, so replay favours "
    "policies resembling the incumbent: a policy that would have fetched objects "
    "the incumbent never touched scores zero because the data to score it does "
    "not exist. The retweeter census is the partial exception - its account x "
    "object incidence is near-total for the objects it covers."
)

# twscrape 0.20.1 `retweeters_raw` sends `count: 20`. NOT the 100 that
# census-tuning.md §2c prices requests at; see the module docstring.
RETWEETERS_PAGE = 20

# A census pass flushes engagements every SNOWBALL_FLUSH_EVERY=25 objects, so one
# pass writes ~10 objects a few minutes apart, and cycles are hours apart. 30
# minutes separates passes without splitting one - but it is a parameter, because
# a rate-limited pass stretches and a truncated one shortens.
PASS_GAP_MINUTES = 30

_RUN_TS = re.compile(r"/run=(\d{8}T\d{6}Z)\.parquet$")


class Leakage(RuntimeError):
    """A policy requested a target that was not observable at *t*."""


# --------------------------------------------------------------------------
# Fetch requests and the observable state
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchRequest:
    """One unit of collection work a policy asks for.

    `requests` is the API cost the policy BUDGETED, computed from what it could
    see at *t*. The realised cost is only knowable after the fetch, so both are
    reported and neither is allowed to stand in for the other.
    """

    kind: str
    target: str
    requests: int = 1
    rank: int = 0

    def __post_init__(self) -> None:
        if self.requests < 1:
            raise ValueError(f"a fetch costs at least one request, got {self.requests}")


@dataclass(frozen=True)
class ReplayState:
    """Everything observable at *t*, and nothing observable after it.

    `posts` and `engagements` are SQL relation expressions already clipped at
    both the object and the row level. A policy that reads anything else is
    outside the harness's guarantee, which is why they are the only corpus
    handles on this object.
    """

    con: duckdb.DuckDBPyConnection
    t: datetime
    posts: str
    engagements: str | None = None
    # This policy's OWN fetch history, not the corpus's. A counterfactual policy
    # censused a different set, so its TTL must be evaluated against what IT
    # fetched; the driver seeds this from the corpus before the replay window and
    # maintains it from the policy's own requests afterwards.
    censused_at: Mapping[str, datetime] = field(default_factory=dict)

    def censused_relation(self, name: str = "_replay_ledger") -> str:
        """The ledger as a relation, so a policy can exclude inside selection.

        Inside, not after. Selection is deterministic, so a TTL applied after
        `ORDER BY ... LIMIT` re-picks completed work every pass and discards it:
        495 selected against 10-35 fetched, measured 2026-08-01
        (census-tuning.md §2e).
        """
        frame = pd.DataFrame(
            {
                "target": list(self.censused_at.keys()),
                "censused_at": pd.to_datetime(
                    list(self.censused_at.values()), utc=True
                ),
            }
        )
        if frame.empty:
            frame = pd.DataFrame(
                {
                    "target": pd.Series(dtype="object"),
                    "censused_at": pd.Series(dtype="datetime64[us, UTC]"),
                }
            )
        self.con.register(name, frame)
        return name


@runtime_checkable
class Policy(Protocol):
    """A collection policy: observable state at *t* -> requested fetches."""

    name: str

    def params(self) -> dict[str, object]:
        """Everything that would change the selection, for the result record."""

    def candidates(self, state: ReplayState) -> list[FetchRequest]:
        ...


# --------------------------------------------------------------------------
# Pinning a state to a moment
# --------------------------------------------------------------------------


def run_timestamp(key: str) -> datetime | None:
    """The write time in an R2 key: `.../run=20260905T081903Z.parquet`.

    `storage.run_id` formats it `%Y%m%dT%H%M%SZ`, so an object's own key dates
    it and the manifest can establish what existed at *t* without reading a
    single row.
    """
    m = _RUN_TS.search(key)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def with_run_times(manifest: pd.DataFrame) -> pd.DataFrame:
    """The manifest plus a `run_ts` column parsed from each key."""
    out = manifest.copy()
    out["run_ts"] = [run_timestamp(k) for k in out["key"]]
    return out


def census_pass_times(
    manifest: pd.DataFrame,
    *,
    prefix: str = "engagements",
    gap_minutes: int = PASS_GAP_MINUTES,
) -> list[datetime]:
    """Recover the collector's own census pass boundaries from object keys.

    Replay steps have to line up with real passes or the reproduction check
    compares a selection at *t* against fetches made hours either side of it.
    Consecutive writes closer together than `gap_minutes` are one pass.
    """
    m = with_run_times(manifest)
    stamps = sorted(t for t in m.loc[m["prefix"] == prefix, "run_ts"] if t is not None)
    if not stamps:
        return []
    gap = timedelta(minutes=gap_minutes)
    starts = [stamps[0]]
    for prev, cur in zip(stamps, stamps[1:]):
        if cur - prev > gap:
            starts.append(cur)
    return starts


def census_pass_metrics(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """The collector's own record of what each census pass selected and fetched.

    `census_runs/` has carried this since the census-metrics rollout, which is
    the evidence that makes reproduction checkable at all: `selected_retweeted`,
    `fetched_retweeted`, `skipped_ttl_retweeted` and `top_retweeted`, tagged with
    `code_version`. It is written BY the collector at the time, not
    reconstructed afterwards by the thing under test.
    """
    from kma import db

    runs = db.census_runs(con).df()
    keep = [
        "collected_at", "code_version", "top_retweeted", "selected_retweeted",
        "fetched_retweeted", "skipped_ttl_retweeted", "engagement_rows",
    ]
    out = runs[[c for c in keep if c in runs.columns]].copy()
    # Positive excess means a second arm appended to the baseline selection.
    out["excess"] = out["selected_retweeted"] - out["top_retweeted"]
    return out.sort_values("collected_at").reset_index(drop=True)


def clean_pass_times(
    con: duckdb.DuckDBPyConnection,
    times: Sequence[datetime],
    *,
    window_minutes: int = PASS_GAP_MINUTES,
) -> list[datetime]:
    """Keep only pass times where the BASELINE arm selected alone.

    `IncumbentCensus` ports `runner.hot_objects`, the baseline arm. On a merged
    pass the collector appends `hot_toxic_objects` and records no baseline-only
    count, so a per-id comparison there scores a one-arm policy against two-arm
    reality and cannot succeed however correct the port is.

    Measured 2026-09-08: of 279 recorded passes, 62 selected within
    `top_retweeted` and 217 exceeded it, by a mean of 111 and a maximum of 250.
    The first reproduction run drew 9 passes and NONE were clean - live selection
    ran 270..500 against the port's fixed 250, which is most of why recall came
    back at 0.404.
    """
    metrics = census_pass_metrics(con)
    if metrics.empty:
        return list(times)
    stamps = pd.to_datetime(metrics["collected_at"], utc=True)
    tolerance = timedelta(minutes=window_minutes)

    out = []
    for t in times:
        target = pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tz else pd.Timestamp(t, tz="UTC")
        delta = (stamps - target).abs()
        nearest = int(delta.values.argmin())
        if delta.iloc[nearest] > tolerance:
            continue
        excess = metrics.iloc[nearest]["excess"]
        # NA rather than a number means the pass predates the column: new
        # `census_runs` columns are unbindable, not NULL, until a pass writes
        # them. An unverifiable pass is not a clean pass - excluding it keeps
        # the filter conservative rather than optimistic.
        if pd.isna(excess):
            continue
        if float(excess) <= 0:
            out.append(t)
    return out


def selection_agreement(
    con: duckdb.DuckDBPyConnection,
    per_pass: pd.DataFrame,
    *,
    window_minutes: int = PASS_GAP_MINUTES,
) -> pd.DataFrame:
    """Per-pass agreement between what replay selected and what the collector
    recorded selecting and fetching.

    A WEAKER claim than per-id matching and must be read as one: it establishes
    that a policy selects like the incumbent in count and degree, not that it
    selects the same objects. It is worth having because `fetched_retweeted` is
    the only quantity that can leave a trace in the corpus - one pass recorded
    500 selected, 202 fetched, 298 skipped on TTL - so it bounds what per-id
    recall could ever be.
    """
    metrics = census_pass_metrics(con)
    stamps = pd.to_datetime(metrics["collected_at"], utc=True)
    tolerance = timedelta(minutes=window_minutes)

    rows = []
    for _, step in per_pass.iterrows():
        target = pd.Timestamp(step["t"]).tz_convert("UTC")
        delta = (stamps - target).abs()
        nearest = int(delta.values.argmin())
        if delta.iloc[nearest] > tolerance:
            continue
        m = metrics.iloc[nearest]
        replayed = int(step["replayed"])
        rows.append({
            "t": step["t"],
            "replay_selected": replayed,
            "live_selected": int(m["selected_retweeted"]),
            "live_fetched": int(m["fetched_retweeted"]),
            "live_ttl_skipped": int(m["skipped_ttl_retweeted"]),
            "excess": int(m["excess"]),
            "selected_ratio": round(replayed / max(int(m["selected_retweeted"]), 1), 3),
            "fetched_ratio": round(replayed / max(int(m["fetched_retweeted"]), 1), 3),
        })
    return pd.DataFrame(rows)


def _paths_before(manifest: pd.DataFrame, prefix: str, t: datetime, **filters: str) -> list[str]:
    m = with_run_times(manifest)
    sel = m[m["prefix"] == prefix]
    for column, value in filters.items():
        if column not in sel.columns:
            raise ValueError(f"unknown partition column {column!r} for prefix {prefix!r}")
        sel = sel[sel[column] == value]
    # An object with no parsable run time cannot be dated, so it cannot be
    # proven to predate t and is excluded. Silently including it is how leakage
    # gets in.
    sel = sel[sel["run_ts"].notna() & (sel["run_ts"] <= t)]
    return sorted(sel["path"].dropna().tolist())


def clipped_source(
    manifest: pd.DataFrame,
    prefix: str,
    t: datetime,
    *,
    column: str = "collected_at",
    **filters: str,
) -> str | None:
    """A relation over exactly the rows of `prefix` observable at `t`.

    None when nothing under the prefix predates `t` - an empty corpus is a
    legitimate state at the start of a replay, not an error, and it is the
    caller's job to decide what a policy does with it.
    """
    paths = _paths_before(manifest, prefix, t, **filters)
    if not paths:
        return None
    listed = ", ".join(f"'{p}'" for p in paths)
    src = f"read_parquet([{listed}], union_by_name=true, hive_partitioning=true)"
    # Both levels, deliberately: object-level pruning keeps the read small,
    # row-level clipping is what makes it correct when a write straddles t.
    return f"(SELECT * FROM {src} WHERE {column} <= TIMESTAMPTZ '{_iso(t)}')"


def _iso(t: datetime) -> str:
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")


def state_at(
    con: duckdb.DuckDBPyConnection,
    manifest: pd.DataFrame,
    t: datetime,
    *,
    censused_at: Mapping[str, datetime] | None = None,
    platform: str | None = "x",
) -> ReplayState:
    """The observable state at `t`, pinned to the snapshot."""
    filters = {"platform": platform} if platform else {}
    posts = clipped_source(manifest, "posts", t, **filters)
    if posts is None:
        raise ValueError(f"snapshot holds no posts written at or before {_iso(t)}")
    return ReplayState(
        con=con,
        t=t,
        posts=posts,
        engagements=clipped_source(manifest, "engagements", t, **filters),
        censused_at=dict(censused_at or {}),
    )


# --------------------------------------------------------------------------
# The incumbent policy - the baseline every candidate is scored against
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IncumbentCensus:
    """The deployed retweeter census, as a replayable policy.

    A PORT of `kenya_monitor.runner.hot_objects`'s retweeted arm, not a call
    into it: the analysis project takes no dependency on the collector, and the
    live function reads `now()` off the wall clock, which a replay cannot use.
    Port fidelity is therefore a real risk and it is tested rather than
    asserted - `analysis/tests/fixtures/replay/` holds selections generated by
    the live collector code, and `test_replay.py` replays against them.

    Defaults are the deployed values (`kenya_monitor.config`): band 3..100 with
    the upper bound tied to v1's `HUB_CAP_MAX`, a 2-day created_at window, 250
    objects per pass, a 12-hour per-object TTL, 300 retweeters per object.

    Only the retweeted arm. The conversation and hydration arms of the same pass
    write to `posts/`, where a replay cannot distinguish a row a policy would
    have fetched from one it did fetch; the retweeted arm writes to
    `engagements/`, whose incidence is near-total for the objects it covers,
    which is the one place the frozen corpus can answer a counterfactual.
    """

    band_min: int = 3
    band_max: int | None = 100
    lookback_days: int = 2
    top_retweeted: int = 250
    refresh_hours: int = 12
    retweeters_limit: int = 300
    page_size: int = RETWEETERS_PAGE
    name: str = "incumbent"

    def params(self) -> dict[str, object]:
        return {
            "band_min": self.band_min,
            "band_max": self.band_max,
            "lookback_days": self.lookback_days,
            "top_retweeted": self.top_retweeted,
            "refresh_hours": self.refresh_hours,
            "retweeters_limit": self.retweeters_limit,
            "page_size": self.page_size,
        }

    def request_cost(self, degree: int) -> int:
        """API requests one census of an object of this degree costs.

        `max(1, ceil(min(degree, limit) / page))`. A FLOOR: twscrape issues one
        further request to discover the end of a full last page, so a saturated
        fetch costs one more than this. Understating the incumbent's cost is the
        conservative direction for a candidate that spends more per object.
        """
        capped = min(int(degree), int(self.retweeters_limit))
        return max(1, math.ceil(capped / self.page_size))

    def _band_clause(self) -> str:
        if self.band_max is None:
            return f"HAVING max(repost_count) >= {int(self.band_min)}"
        return (
            f"HAVING max(repost_count) BETWEEN {int(self.band_min)} "
            f"AND {int(self.band_max)}"
        )

    def candidates(self, state: ReplayState) -> list[FetchRequest]:
        ledger = state.censused_relation()
        cutoff = _iso(state.t - timedelta(hours=self.refresh_hours))
        rows = state.con.sql(
            f"""
            WITH recent AS (
                SELECT * FROM (
                    SELECT * FROM {state.posts}
                    -- `dt` is the COLLECTION date and `created_at` the post
                    -- date, so every row inside the created_at window also
                    -- lands in a dt partition at or after the same cutoff:
                    -- this cannot drop a row the created_at filter would keep.
                    -- It is a pure pruning hint, and it is the difference
                    -- between reading two days of parquet and reading the
                    -- whole pinned corpus on every step. The live selector
                    -- carries the same predicate for the same reason, after
                    -- its absence pushed collector cycles from ~150min to
                    -- 1,159min.
                    WHERE dt >= CAST(TIMESTAMPTZ '{_iso(state.t)}'
                                     - INTERVAL {int(self.lookback_days) + 1} DAY AS DATE)
                    QUALIFY row_number() OVER (
                        PARTITION BY platform, platform_post_id
                        ORDER BY collected_at DESC
                    ) = 1
                )
                WHERE created_at > TIMESTAMPTZ '{_iso(state.t)}'
                                   - INTERVAL {int(self.lookback_days)} DAY
            ), cand AS (
                SELECT recent.repost_of_id AS oid,
                       max(repost_count) AS deg,
                       count(*) AS n_rt_rows,
                       -- The TTL is applied INSIDE selection, before the LIMIT.
                       EXISTS (
                           SELECT 1 FROM {ledger} l
                           WHERE l.target = recent.repost_of_id
                             AND l.censused_at > TIMESTAMPTZ '{cutoff}'
                       ) AS censused
                FROM recent
                WHERE repost_of_id IS NOT NULL
                GROUP BY 1
                {self._band_clause()}
            )
            SELECT oid, deg FROM cand
            WHERE NOT censused
            ORDER BY deg DESC, n_rt_rows DESC, oid
            LIMIT {int(self.top_retweeted)}
            """
        ).fetchall()
        return [
            FetchRequest("retweeters", str(oid), self.request_cost(int(deg)), rank)
            for rank, (oid, deg) in enumerate(rows)
        ]


@dataclass(frozen=True)
class UnbandedCensus(IncumbentCensus):
    """B3 candidate 1: drop the band's upper bound, keep everything else.

    `SNOWBALL_BAND_MAX` is 3..100 with the top tied to v1's `HUB_CAP_MAX`, which
    DELETED hub entities before pairing accounts. v2 has no hub cap: TF-IDF
    down-weights popular entities instead (`coord2.bipartite_tfidf`), so the
    reason the ceiling existed does not carry over. Independent evidence from
    the other direction: the parent-hydration pass measured its amplifier range
    reaching 20..242 on pi0 on 2026-09-08, and banding there would have forfeited
    the ten densest objects in the whole backlog
    (`docs/plans/2026-09-08-collector-depth.md`, Step 7).

    Chosen over the other three B3 candidates because it is the only one the
    frozen corpus can say anything about at all. Budget split and TTL both
    change which objects get censused into a corpus that already recorded one
    answer, and object ranking within the band reorders a population the census
    has nearly worked through (742 selectable of 3,496 in-band, 2026-08-02).
    The band's ceiling is different: the corpus holds a pre-banding era - the
    ceiling shipped 16:19 on 2026-08-01 - in which hub objects WERE censused,
    at 420 of 422 objects above 100 amplifiers. So the incidence needed to score
    hub selection exists for that era, and `Yield.unscorable_requests` measures
    how much of it exists for any other.

    Which means the replay times matter as much as the policy: score this
    candidate over passes from BEFORE 2026-08-01 16:19 and its hub selections
    are scorable; score it over recent passes and almost none of them are.
    `kma-replay --passes N` takes the most RECENT N, so it is the wrong window
    for this candidate and `unscorable_share` will say so.
    """

    band_max: int | None = None
    name: str = "unbanded"


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


@dataclass
class StepRecord:
    t: datetime
    requested: int
    requests: int
    targets: tuple[str, ...]
    degrees: tuple[int, ...]


@dataclass
class ReplayResult:
    snapshot: str
    policy: str
    params: dict[str, object]
    steps: list[StepRecord]
    limitation: str = LIMITATION

    @property
    def fetches(self) -> list[FetchRequest]:
        return [
            FetchRequest("retweeters", target, 1)
            for step in self.steps
            for target in step.targets
        ]

    @property
    def targets(self) -> list[str]:
        """Every target fetched, in fetch order, duplicates kept: a re-census
        after the TTL is a second request and costs a second time."""
        return [target for step in self.steps for target in step.targets]

    @property
    def planned_requests(self) -> int:
        return sum(step.requests for step in self.steps)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "snapshot": self.snapshot,
                "policy": self.policy,
                "t": [s.t for s in self.steps],
                "requested": [s.requested for s in self.steps],
                "requests": [s.requests for s in self.steps],
                "limitation": self.limitation,
            }
        )


def _audit(state: ReplayState, requests: Sequence[FetchRequest]) -> None:
    """Every target must be derivable from the visible state.

    A policy can honour the clipped views and still leak by construction - by
    naming an object it had no way to know about. The visible target space is
    post ids plus the ids visible posts REFERENCE, because the hydration arm
    legitimately asks for objects that are referenced and not held.
    """
    if not requests:
        return
    frame = pd.DataFrame({"target": [r.target for r in requests]})
    state.con.register("_replay_audit", frame)
    unseen = state.con.sql(
        f"""
        WITH visible AS (
            -- One scan of five projected columns, not five scans: the audit runs
            -- on every step and the pinned relation is the whole corpus to date.
            SELECT unnest([platform_post_id, repost_of_id, quoted_post_id,
                           in_reply_to_id, conversation_id]) AS id
            FROM {state.posts}
        )
        SELECT DISTINCT a.target FROM _replay_audit a
        WHERE NOT EXISTS (SELECT 1 FROM visible v WHERE v.id = a.target)
        LIMIT 5
        """
    ).fetchall()
    if unseen:
        raise Leakage(
            f"policy requested {len(unseen)} target(s) not observable at {_iso(state.t)}: "
            f"{[u[0] for u in unseen]}"
        )


def seed_ledger(
    con: duckdb.DuckDBPyConnection,
    manifest: pd.DataFrame,
    t: datetime,
    *,
    platform: str | None = "x",
) -> dict[str, datetime]:
    """Last-censused time per object as of `t`, preferring the recorded ledger.

    A replay window that does not start at the corpus's first day inherits a
    census history, and starting empty would let the policy re-fetch work
    already done - inflating its request count and its yield together.

    TWO SOURCES, and which one is available decides whether per-id reproduction
    can succeed at all.

    `census_ttl/` is the collector's own TTL ledger, one row per object with the
    time it was censused. It is exact. It has only been captured since
    2026-09-08, so it is absent from earlier snapshots.

    `engagements/` is the fallback: infer "censused" from "wrote engagement
    rows". That inference is wrong in one specific and common way - an object
    selected that returned no retweeters writes no engagement row, so it looks
    un-censused and the policy re-selects it while the live collector skipped it
    on TTL. Measured 2026-09-08: **69.4% of objects are re-censused, mean span
    47.6 hours** against a 12-hour TTL, so the ledger is doing heavy work and
    errors in it move a large share of the 250 objects selected per pass. This
    is the main suspect for per-id reproduction sitting at 0.683 rather than
    near 1.0.
    """
    # `censused_at` is stored as text, so the clip casts rather than comparing a
    # varchar to a timestamp; `collected_at` does not exist on this prefix.
    ttl = clipped_source(
        manifest,
        "census_ttl",
        t,
        column="CAST(censused_at AS TIMESTAMPTZ)",
        **({"platform": platform} if platform else {}),
    )
    if ttl is not None:
        rows = con.sql(
            f"""SELECT object_id, max(CAST(censused_at AS TIMESTAMPTZ)) AS censused_at
                FROM {ttl} GROUP BY 1"""
        ).fetchall()
        if rows:
            log.info("seed_ledger: %d objects from the recorded census_ttl ledger", len(rows))
            return {str(oid): ts for oid, ts in rows if oid is not None}

    src = clipped_source(manifest, "engagements", t, **({"platform": platform} if platform else {}))
    if src is None:
        return {}
    rows = con.sql(
        f"SELECT platform_post_id, max(collected_at) FROM {src} GROUP BY 1"
    ).fetchall()
    log.info(
        "seed_ledger: %d objects INFERRED from engagement writes - no census_ttl "
        "for this window, so censused-but-empty objects are invisible",
        len(rows),
    )
    return {str(oid): ts for oid, ts in rows if oid is not None}


def replay(
    con: duckdb.DuckDBPyConnection,
    *,
    policy: Policy,
    manifest: pd.DataFrame,
    times: Sequence[datetime],
    snapshot: str,
    budget: int | None = None,
    audit: bool = True,
    platform: str | None = "x",
    seed: bool = True,
) -> ReplayResult:
    """Run `policy` over the snapshot, one step per time in `times`.

    Times are sorted ascending before use: a driver that steps backwards would
    hand a policy a state it has already acted on and quietly break the ledger.

    `budget` caps requests per step, applied in the policy's own ranking order,
    because that is how a rate-limit abort truncates a real pass
    (`collect_snowball` flushes every 25 objects and loses the tail).
    """
    ordered = sorted(times)
    if not ordered:
        raise ValueError("replay needs at least one time to step to")
    ledger = seed_ledger(con, manifest, ordered[0], platform=platform) if seed else {}
    steps: list[StepRecord] = []

    for t in ordered:
        state = state_at(con, manifest, t, censused_at=ledger, platform=platform)
        requests = list(policy.candidates(state))
        if audit:
            _audit(state, requests)
        taken, spent = [], 0
        for req in sorted(requests, key=lambda r: r.rank):
            if budget is not None and spent + req.requests > budget:
                break
            taken.append(req)
            spent += req.requests
        for req in taken:
            ledger[req.target] = t
        steps.append(
            StepRecord(
                t=t,
                requested=len(requests),
                requests=spent,
                targets=tuple(r.target for r in taken),
                degrees=tuple(r.requests for r in taken),
            )
        )
        log.info(
            "%s @ %s: %d candidates, %d taken, %d requests",
            policy.name, _iso(t), len(requests), len(taken), spent,
        )

    return ReplayResult(
        snapshot=snapshot,
        policy=policy.name,
        params=policy.params(),
        steps=steps,
    )


# --------------------------------------------------------------------------
# B2: the scoring metric - coordination-useful yield per request
# --------------------------------------------------------------------------


@dataclass
class Yield:
    """Fused-network edges bought per API request spent.

    `unscorable_requests` is not a footnote. It is the share of the policy's
    spend aimed at objects the incumbent never censused, for which the corpus
    holds no incidence and this harness can only return zero. Read it first.
    """

    snapshot: str
    policy: str
    edges: int
    users: int
    planned_requests: int
    realised_requests: int
    scorable_targets: int
    unscorable_targets: int
    incidence_rows: int
    limitation: str = LIMITATION

    @property
    def edges_per_request(self) -> float:
        return self.edges / self.planned_requests if self.planned_requests else 0.0

    @property
    def edges_per_realised_request(self) -> float:
        return self.edges / self.realised_requests if self.realised_requests else 0.0

    @property
    def unscorable_share(self) -> float:
        total = self.scorable_targets + self.unscorable_targets
        return self.unscorable_targets / total if total else 0.0

    def as_row(self) -> dict[str, object]:
        return {
            "snapshot": self.snapshot,
            "policy": self.policy,
            "edges": self.edges,
            "users": self.users,
            "planned_requests": self.planned_requests,
            "realised_requests": self.realised_requests,
            "edges_per_request": self.edges_per_request,
            "edges_per_realised_request": self.edges_per_realised_request,
            "scorable_targets": self.scorable_targets,
            "unscorable_targets": self.unscorable_targets,
            "unscorable_share": self.unscorable_share,
            "incidence_rows": self.incidence_rows,
            "limitation": self.limitation,
        }


def census_incidence(
    con: duckdb.DuckDBPyConnection,
    manifest: pd.DataFrame,
    targets: Iterable[str],
    *,
    platform: str | None = "x",
) -> pd.DataFrame:
    """(user_id, entity) rows for the census of `targets`, over the whole snapshot.

    Unclipped in time on purpose: the question is what the census of these
    objects yields, not when it happened. Deduplicated on (object, account),
    matching `kma.db.latest_engagements` - engagements carry no tombstones, so an
    edge seen once is retained forever and a repeat sighting is not new signal.
    """
    wanted = sorted({str(t) for t in targets})
    empty = pd.DataFrame(
        {"user_id": pd.Series(dtype="object"), "entity": pd.Series(dtype="object")}
    )
    if not wanted:
        return empty
    src = clipped_source(
        manifest,
        "engagements",
        datetime.now(timezone.utc),
        **({"platform": platform} if platform else {}),
    )
    if src is None:
        return empty
    con.register("_replay_targets", pd.DataFrame({"target": wanted}))
    return con.sql(
        f"""
        SELECT DISTINCT e.platform_user_id AS user_id,
                        e.platform_post_id AS entity
        FROM {src} e
        JOIN _replay_targets t ON t.target = e.platform_post_id
        WHERE e.platform_user_id IS NOT NULL
        """
    ).df()


def score(
    con: duckdb.DuckDBPyConnection,
    manifest: pd.DataFrame,
    result: ReplayResult,
    *,
    percentile: float | None = None,
    min_entities: int = coord2.MIN_ENTITIES_PER_USER,
    platform: str | None = "x",
) -> Yield:
    """Coordination-useful yield of a replayed policy.

    The census of an object is an account x entity incidence, which is exactly
    `co_retweet`'s bipartite input, so the network is built by `kma.coord2` -
    `similarity_network` for the projection and `fuse` for the union - and not
    reimplemented here.

    `percentile` defaults to None rather than `EDGE_PERCENTILE["co_retweet"]`
    (80.0). A percentile is taken over the realised weights of the network being
    scored, so two policies filtered at the 80th percentile are each cut against
    their OWN distribution and their edge counts are not comparable. Pass 80.0
    to reproduce a single policy's deployed network; leave it None to compare
    policies.
    """
    targets = result.targets
    unique = sorted(set(targets))
    traces = census_incidence(con, manifest, unique, platform=platform)
    covered = set(traces["entity"].astype(str)) if not traces.empty else set()

    network = coord2.similarity_network(
        traces, percentile=percentile, min_entities=min_entities
    )
    fused = coord2.fuse({"co_retweet": network})

    # Realised cost prices each fetch by the incidence that actually came back;
    # planned cost is what the policy could budget at t. Neither substitutes for
    # the other, so both are reported.
    per_target = (
        traces.groupby("entity")["user_id"].nunique().to_dict() if not traces.empty else {}
    )
    cost = result_policy_cost(result)
    realised = sum(cost(int(per_target.get(target, 0))) for target in targets)

    return Yield(
        snapshot=result.snapshot,
        policy=result.policy,
        edges=int(fused.number_of_edges()),
        users=int(fused.number_of_nodes()),
        planned_requests=result.planned_requests,
        realised_requests=realised,
        scorable_targets=len(covered),
        unscorable_targets=len(set(unique) - covered),
        incidence_rows=int(len(traces)),
    )


def result_policy_cost(result: ReplayResult) -> Callable[[int], int]:
    """The request-cost model the replay ran under, rebuilt from its params.

    A candidate that changes `retweeters_limit` or `page_size` changes what a
    fetch costs, so pricing the realised fetch with the module default would
    quietly score it under the incumbent's economics. A policy that records
    neither falls back to the deployed values.
    """
    return IncumbentCensus(
        retweeters_limit=int(result.params.get("retweeters_limit", 300)),
        page_size=int(result.params.get("page_size", RETWEETERS_PAGE)),
    ).request_cost


def compare(
    con: duckdb.DuckDBPyConnection,
    manifest: pd.DataFrame,
    baseline: ReplayResult,
    candidate: ReplayResult,
    *,
    percentile: float | None = None,
    platform: str | None = "x",
) -> pd.DataFrame:
    """Both yields side by side, with the delta and the limitation on every row."""
    rows = [
        score(con, manifest, r, percentile=percentile, platform=platform).as_row()
        for r in (baseline, candidate)
    ]
    frame = pd.DataFrame(rows)
    base, cand = frame.iloc[0], frame.iloc[1]
    frame["ratio_vs_baseline"] = [
        1.0,
        (cand["edges_per_request"] / base["edges_per_request"])
        if base["edges_per_request"]
        else float("nan"),
    ]
    return frame


# --------------------------------------------------------------------------
# The acceptance test: does replaying the incumbent reproduce reality?
# --------------------------------------------------------------------------


@dataclass
class Reproduction:
    """How closely a replayed policy matches what the collector actually did.

    A harness that cannot reproduce what happened cannot be trusted to score
    what did not, so this is the gate on the whole module, not a diagnostic.
    """

    snapshot: str
    policy: str
    per_pass: pd.DataFrame
    limitation: str = LIMITATION

    @property
    def recall(self) -> float:
        """Share of actually-fetched objects the replay also selected."""
        got, want = self.per_pass["matched"].sum(), self.per_pass["observed"].sum()
        return float(got / want) if want else float("nan")

    @property
    def precision(self) -> float:
        got, want = self.per_pass["matched"].sum(), self.per_pass["replayed"].sum()
        return float(got / want) if want else float("nan")

    @property
    def jaccard(self) -> float:
        m = self.per_pass["matched"].sum()
        union = (
            self.per_pass["observed"].sum() + self.per_pass["replayed"].sum() - m
        )
        return float(m / union) if union else float("nan")


def observed_fetches(
    con: duckdb.DuckDBPyConnection,
    manifest: pd.DataFrame,
    start: datetime,
    end: datetime,
    *,
    platform: str | None = "x",
) -> set[str]:
    """Objects the collector actually censused in [start, end).

    Ground truth for the reproduction check. An object whose retweeter fetch
    returned nothing leaves NO engagement row, so it is invisible here and
    counts against the replay's precision through no fault of the replay - one
    of the tolerance terms in `reproduce`'s docstring.
    """
    src = clipped_source(
        manifest, "engagements", end, **({"platform": platform} if platform else {})
    )
    if src is None:
        return set()
    rows = con.sql(
        f"""
        SELECT DISTINCT platform_post_id FROM {src}
        WHERE collected_at >= TIMESTAMPTZ '{_iso(start)}'
          AND collected_at <  TIMESTAMPTZ '{_iso(end)}'
        """
    ).fetchall()
    return {str(r[0]) for r in rows if r[0] is not None}


def reproduce(
    con: duckdb.DuckDBPyConnection,
    *,
    manifest: pd.DataFrame,
    times: Sequence[datetime],
    snapshot: str,
    policy: Policy | None = None,
    platform: str | None = "x",
    audit: bool = True,
) -> Reproduction:
    """Replay the incumbent and compare, pass by pass, against the census that ran.

    Five terms make up the tolerance, all of them structural rather than
    fixable, and all of them push recall DOWN and precision DOWN:

    1. A pass whose `pass_kind` is `merged` appends `hate_signal.hot_toxic_objects`
       to the baseline selection, and `census_runs/` records no baseline-only
       count, so those extra objects appear as observed-but-not-replayed.
    2. The live TTL is two mechanisms - `censused_expr` over R2 and `_due` over a
       local JSON ledger that is checkpointed every 25 objects and pruned at
       twice the TTL. Only the R2 half survives in the corpus.
    3. A pass truncated by a rate limit fetches a prefix of its selection; the
       replay always fetches the whole thing.
    4. An object whose fetch returned no retweeters writes no engagement row, so
       it is observed as not-fetched.
    5. `repost_count` is read at selection time in both, but the corpus keeps
       only the latest row per post id, so a post re-collected AFTER the pass
       carries a later count. Median growth was measured at 1.00 across 717
       objects (census-tuning.md Q2), which bounds this term rather than
       removing it.
    6. `census_pass_times` dates a pass by its FIRST engagement write, which is
       minutes after selection ran - the census fetches before it flushes. The
       replay therefore stands a few minutes late and can see posts the real
       selector could not. Pass earlier times to close it; the snapshot cannot
       recover the true selection instant, because nothing records it.
    """
    policy = policy or IncumbentCensus()
    ordered = sorted(times)
    result = replay(
        con,
        policy=policy,
        manifest=manifest,
        times=ordered,
        snapshot=snapshot,
        platform=platform,
        audit=audit,
    )
    # A pass's fetches land between its start and the next pass's start; the
    # final pass is closed at its own start plus one gap, since the snapshot
    # cannot say when it ended.
    bounds = list(ordered[1:]) + [ordered[-1] + timedelta(minutes=PASS_GAP_MINUTES)]
    rows = []
    for step, end in zip(result.steps, bounds):
        replayed = set(step.targets)
        observed = observed_fetches(con, manifest, step.t, end, platform=platform)
        rows.append(
            {
                "t": step.t,
                "replayed": len(replayed),
                "observed": len(observed),
                "matched": len(replayed & observed),
                "replayed_only": len(replayed - observed),
                "observed_only": len(observed - replayed),
            }
        )
    return Reproduction(snapshot=snapshot, policy=policy.name, per_pass=pd.DataFrame(rows))


# --------------------------------------------------------------------------


def main() -> None:
    import argparse

    from kma import bench
    from kma.db import connect

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Score a candidate collection policy against a frozen snapshot."
    )
    ap.add_argument("snapshot", help="snapshot name, e.g. 2026-09-05-promotion-off")
    ap.add_argument("--passes", type=int, default=10, help="most recent census passes to replay")
    ap.add_argument(
        "--reproduce",
        action="store_true",
        help="run the acceptance check only: incumbent replay against the census that ran",
    )
    ap.add_argument("--band-max", type=int, default=None, help="candidate band ceiling; omit for unbanded")
    ap.add_argument(
        "--clean-only",
        action="store_true",
        help="replay only passes where the baseline arm selected alone (see clean_pass_times)",
    )
    args = ap.parse_args()

    con = connect()
    manifest = bench.load(args.snapshot, con=con)
    times = census_pass_times(manifest)
    if not times:
        raise SystemExit(f"snapshot {args.snapshot!r} holds no engagements objects to replay")
    if args.clean_only:
        before = len(times)
        times = clean_pass_times(con, times)
        print(f"clean-pass filter: {len(times)} of {before} passes had the baseline arm select alone")
        if not times:
            raise SystemExit(
                "no clean passes in this snapshot: every recorded pass appended a second arm, "
                "so a one-arm port cannot be compared per id here"
            )
    times = times[-args.passes :]
    print(f"replaying {len(times)} passes, {_iso(times[0])} .. {_iso(times[-1])}")

    if args.reproduce:
        rep = reproduce(con, manifest=manifest, times=times, snapshot=args.snapshot)
        print(rep.per_pass.to_string(index=False))
        agree = selection_agreement(con, rep.per_pass)
        if not agree.empty:
            print("\nselection agreement against the collector's own record:")
            print(agree.to_string(index=False))
            print(
                f"\nmedian replay/live selected {agree['selected_ratio'].median():.3f}, "
                f"replay/live fetched {agree['fetched_ratio'].median():.3f}"
            )
        print(
            f"\nsnapshot={rep.snapshot} policy={rep.policy} "
            f"recall={rep.recall:.3f} precision={rep.precision:.3f} jaccard={rep.jaccard:.3f}"
        )
        print(f"\n{rep.limitation}")
        return

    base = replay(
        con, policy=IncumbentCensus(), manifest=manifest, times=times, snapshot=args.snapshot
    )
    candidate = (
        replace(UnbandedCensus(), band_max=args.band_max)
        if args.band_max is not None
        else UnbandedCensus()
    )
    cand = replay(
        con, policy=candidate, manifest=manifest, times=times, snapshot=args.snapshot
    )
    frame = compare(con, manifest, base, cand)
    print(frame.drop(columns=["limitation"]).to_string(index=False))
    print(f"\n{LIMITATION}")


if __name__ == "__main__":
    main()

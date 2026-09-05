"""Dashboard producer: aggregate the R2 dataset into compact, public-safe JSON
summaries for the kenya.tomfr.net dashboard to read live.

The website is a Cloudflare Worker; it must not run DuckDB or touch raw Parquet.
So the heavy aggregation lives here (reusing the same readers the notebooks use)
and the result is written to a single stable R2 key `dashboard/summary.json`
(plus a timestamped archive). The Worker binds the bucket and serves that file.

    kma-dashboard --once      # build + write once
    kma-dashboard --loop      # forever, ~60 min cadence

Scheduling with no new infra: this runs as its own lightweight service on the
analysis image (`docker compose --profile dashboard up -d dashboard`, beside
enrich on tf1). It reads only PERSISTED artifacts from R2 - no torch, and no
coordination rebuild.

Why it reads the persisted coordination run rather than recomputing it: a
rebuild here would publish a different clustering from the one the collector's
targeting acts on, computed minutes apart from the same code, and two
disagreeing "N clusters" numbers is exactly the failure the census-tuning work
existed to stop. It is also the heaviest job in the repo and would contend with
`enrich` on a 3.7GB host.

PUBLIC-SAFE BY CONSTRUCTION: every field here is an aggregate (counts,
histograms, distributions, cluster keyword names). No account handles, author
ids, post ids or example post text are emitted. `tests/test_dashboard.py`
asserts this against the built payload rather than trusting the claim.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import duckdb
import pandas as pd

from kma import db, measure
from kma.db import BUCKET, connect

log = logging.getLogger("kma.dashboard")

SUMMARY_KEY = "dashboard/summary.json"
SCHEMA_VERSION = 1
DASHBOARD_INTERVAL_S = int(os.getenv("DASHBOARD_INTERVAL_S", "3600"))

# The Worker binds THIS bucket, which holds only the JSON blobs below. An R2
# binding cannot be scoped to a key prefix, so binding the data bucket would put
# the entire raw corpus - handles, post text - one line of Worker code from the
# public internet. Defaults to the data bucket so an unset var degrades to the
# old single-bucket behaviour rather than crashing.
DASHBOARD_BUCKET = os.getenv("R2_DASHBOARD_BUCKET", BUCKET)

# The persisted edge partition matching the run of record. `coordination_run`
# clusters on `co.DEFAULT_EDGE_METHOD` ("bonferroni"), which `EDGE_METHODS`
# writes under `method=svn_bonf`. Reading any other partition would draw a
# network the published cluster count was not derived from.
PERSISTED_METHOD = "svn_bonf"

NET_MIN_SIZE = 4       # ignore coordinating pairs/triples - show real groups
NET_TOP_CLUSTERS = 50  # clusters to draw
NET_MAX_EDGES = 4000   # payload/perf cap (kept by weight)


# `collection_start` costs a listing of the whole posts prefix (31s cold,
# measured 2026-08-05) to produce a value that only moves when a backfill
# reaches further back than we have ever collected. Recomputing it hourly is the
# single largest cost in the build, so it is cached for a day.
_START_TTL_S = 86_400
_start_cache: tuple[float, str | None] | None = None


@contextmanager
def _timed(timings: dict, name: str):
    """Record a section's wall time. The cadence is chosen from these numbers,
    so they are logged rather than guessed at."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round(time.perf_counter() - t0, 2)
        log.info("section %s: %.2fs", name, timings[name])


# The exact 97.5th percentile of the standard normal, not the 1.96 shorthand.
# At n in the thousands the rounding moves an interval bound in the 5th decimal,
# which is invisible on a chart but makes the statsmodels equivalence test - the
# only thing establishing this implementation is correct - fail.
_Z_975 = 1.959963984540054


def _wilson(k: int, n: int, z: float = _Z_975) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Not the normal approximation: these rates run to a handful of events in a
    day, where a Wald interval routinely covers negative values and understates
    the width. Matches `statsmodels.stats.proportion.proportion_confint(
    method="wilson")`, which the tests check it against."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _f(row, col, default=0.0) -> float:
    v = row.get(col, default)
    return default if v is None or pd.isna(v) else float(v)


def _collection_start(con: duckdb.DuckDBPyConnection, platform: str = "x") -> str | None:
    """When we first collected anything, cached for `_START_TTL_S`.

    `collected_at`, not `created_at`: the oldest authored post in the corpus is
    from 2010, surfaced by a timeline fetch, and publishing that as the start of
    the series would imply 16 years of observation we do not have."""
    global _start_cache
    now = time.time()
    if _start_cache and now - _start_cache[0] < _START_TTL_S:
        return _start_cache[1]
    row = con.sql(f"SELECT min(collected_at) FROM {db.posts_source(platform)}").fetchone()
    value = row[0].isoformat() if row and row[0] else None
    _start_cache = (now, value)
    return value


def build_meta(con: duckdb.DuckDBPyConnection, platform: str = "x") -> dict:
    """The methodology block. The spec requires these rules to be visible on the
    page rather than buried, so they travel with the data instead of being
    hardcoded in the site - a scoping change here cannot leave a stale caveat
    rendered next to a fresh number."""
    # Read off the SQL that will actually be generated, not off targets.yaml
    # being readable. The two disagreed: `leak_corrected` reported True while
    # `_toxicity_frame` hand-rolled its scoping without `effective_type_expr`.
    scoped = db.scoped_posts_cte(platform, "baseline")
    return {
        "code_version": os.getenv("GIT_SHA") or None,
        "scope": "baseline",
        "leak_corrected": scoped.applied,
        "leak_note": (
            "Accounts promoted by coordination targeting wrote to a baseline "
            "timeline partition before 2026-08-01. Reclassified as targeted by "
            "first-seen type; measured 2026-08-13 at 4,312 posts, 1.10% of the "
            "baseline denominator."
        ),
        "collection_start": _collection_start(con, platform),
        "search_horizon_days": 14,
        "horizon_note": (
            "X search reaches back 14 days, so the series begins when collection "
            "began and cannot be backfilled earlier."
        ),
        "excluded_types": list(db.TARGETED_TYPES),
        "coverage_note": (
            "We sample X rather than censusing it. Absence of evidence is not "
            "evidence of absence; recall is bounded, precision is not affected."
        ),
        "model_limits": {
            "hate_classifier": (
                "afro-xlmr, triage threshold p_hate >= 0.28. Blind to the coded "
                "register: on 14 known-coded posts its mean p_hate dropped."
            ),
            "coordination": (
                "Detection is triage, not a verdict. We do not publish that any "
                "account is a bot."
            ),
        },
    }


MIN_DAY_N = 20  # matches notebooks/hatespeech.py; thin days make the rate jumpy


def _toxicity_frame(con: duckdb.DuckDBPyConnection, platform: str = "x") -> pd.DataFrame:
    """The enriched post frame behind statistic B.

    Lifted from `notebooks/hatespeech.py` so the published series and the
    notebook are the same measurement. Scoping is on FIRST-seen type: a post
    found by a baseline search and later re-collected by a hate pass keeps its
    latest `type`, so scoping on that would drain exactly the toxic tail out of
    the baseline denominator and read as a falling trend.

    Unlike the notebook this selects the PERSISTED measurement columns and only
    recomputes the rows that predate their rollout - the notebook recomputes
    every row, which is affordable interactively and is not affordable hourly.

    Scoping goes through `db.scoped_posts_cte`, which is the only spelling that
    carries the promoted-account leak correction along with the first-seen
    typing. Hand-rolling the CTE and predicate here omitted it while `build_meta`
    still published `leak_corrected: true` - a provenance claim the query did not
    honour, worth 4,312 posts (1.10%) of the baseline denominator.
    """
    scoped = db.scoped_posts_cte(platform, "baseline", name="p")
    # A hive glob matching zero files raises at CTE resolution rather than
    # contributing no rows, so an absent `incitement/` prefix would take the
    # whole statement down - and the producer's blanket except would then just
    # retry hourly, leaving summary.json unpublished. The NLI columns are
    # LEFT JOINed and already optional, so substituting NULLs is exactly the
    # same result the join gives on a prefix with no matching rows.
    has_nli = db.prefix_readable(con, db.incitement_source(platform))
    nli_cte = (
        f""", i AS (
            SELECT * FROM {db.incitement_source(platform)}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY scored_at DESC
            ) = 1
        )"""
        if has_nli
        else ""
    )
    nli_cols = (
        """i.dehumanisation_score, i.violence_call_score,
               i.othering_score, i.political_criticism_score"""
        if has_nli
        else """CAST(NULL AS DOUBLE) AS dehumanisation_score,
               CAST(NULL AS DOUBLE) AS violence_call_score,
               CAST(NULL AS DOUBLE) AS othering_score,
               CAST(NULL AS DOUBLE) AS political_criticism_score"""
    )
    nli_join = "LEFT JOIN i ON p.platform_post_id = i.platform_post_id" if has_nli else ""
    if not has_nli:
        log.warning("toxicity: incitement prefix unreadable; coded series will be empty")
    df = con.sql(
        f"""
        WITH {scoped.cte},
        h AS (
            SELECT * FROM {db.hatespeech_source(platform)}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY scored_at DESC
            ) = 1
        ){nli_cte}
        SELECT p.platform_post_id, p.created_at, p.text, p.first_type,
               h.label, h.p_hate, h.hate_flag,
               h.domain, h.in_kenya_scope, h.coded_suspect, h.explicit_toxic,
               {nli_cols}
        FROM p
        JOIN h USING (platform_post_id)
        {nli_join}
        """
    ).df()

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["is_offensive"] = df["label"] == "offensive"
    df["is_hate"] = df["label"] == "hate"

    needs = df["domain"].isna() if "domain" in df.columns else pd.Series(True, index=df.index)
    if needs.any():
        live = measure.attach_measurement_columns(df[needs])
        for c in ("domain", "in_kenya_scope", "coded_suspect", "explicit_toxic"):
            df.loc[needs, c] = live[c]
    df["in_kenya_scope"] = df["in_kenya_scope"].fillna(False).astype(bool)
    df["coded_suspect"] = df["coded_suspect"].fillna(False).astype(bool)
    return df


def _rate(k, n: int) -> dict:
    k = int(k)
    lo, hi = _wilson(k, n)
    return {"k": k, "p": round(k / n, 6), "lo": round(lo, 6), "hi": round(hi, 6)}


#: Series that are standardised. Each is a boolean column on the toxicity frame.
_STANDARDISED_SERIES = ("explicit_toxic", "is_offensive", "is_hate", "coded_suspect")


def _composition_block(d: pd.DataFrame) -> dict:
    """Weekly raw vs composition-standardised rates, plus the mix that drives
    the gap between them.

    The raw baseline rate is not comparable across time. `replies` is a baseline
    type whose volume the collector tunes for coordination reasons, and it
    carries ~3.2x the hate rate of `search`. When the 2026-08-06 conversation
    widening moved the mix from 70.7% search / 12.1% replies to 16.3% / 71.2%,
    the raw weekly rate rose 73% while every within-stratum rate held or fell -
    the standardised series over the same period falls 21%.

    Publishing the composition alongside is what makes that self-evident rather
    than something a reader has to take on trust.
    """
    if d.empty or "first_type" not in d.columns:
        return {"weekly": [], "composition": [], "by_partition": []}

    w = d.copy()
    # Drop the tz before bucketing: `to_period` discards it with a warning, and
    # weeks are UTC-anchored here on purpose so they line up with the frozen
    # reference composition rather than with the display timezone.
    utc_naive = w["created_at"].dt.tz_convert("UTC").dt.tz_localize(None)
    w["week"] = utc_naive.dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")

    weekly: dict[str, dict] = {}
    for col in _STANDARDISED_SERIES:
        if col not in w.columns:
            continue
        part = measure.standardise_by_period(w, "week", "first_type", col)
        for _, r in part.iterrows():
            row = weekly.setdefault(r["week"], {"week": r["week"], "n": int(r["n"])})
            row[col] = {
                "raw": round(float(r["raw"]), 6),
                "standardised": round(float(r["standardised"]), 6),
            }
            if r["missing_strata"]:
                row.setdefault("missing_strata", {})[col] = r["missing_strata"]

    comp = measure.composition_by_period(w, "week", "first_type")
    composition = [
        {"week": week, **{c: round(float(v), 6) for c, v in row.items()}}
        for week, row in comp.iterrows()
    ]

    by_partition = []
    for (week, stratum), grp in w.groupby(["week", "first_type"]):
        entry = {"week": week, "first_type": stratum, "n": int(len(grp))}
        for col in _STANDARDISED_SERIES:
            if col in grp.columns:
                entry[col] = round(float(grp[col].mean()), 6)
        by_partition.append(entry)

    return {
        "weekly": [weekly[k] for k in sorted(weekly)],
        "composition": composition,
        "by_partition": by_partition,
        "reference_week": measure.REFERENCE_WEEK,
        "reference_composition": dict(measure.REFERENCE_COMPOSITION),
        "note": (
            "`raw` is NOT comparable across time: the baseline scope's "
            "composition is set by collection policy, and replies carry ~3.2x "
            "the toxicity rate of search. `standardised` holds the mix at the "
            "reference week. Read `composition` to see why they diverge."
        ),
    }


def build_toxicity(
    con: duckdb.DuckDBPyConnection, platform: str = "x", tz: str = "Africa/Nairobi"
) -> dict:
    """Statistic B: explicit toxicity and coded incitement, as two series.

    Deliberately never merged into one "hate" line. The classifier is good at
    explicit toxicity and demonstrably poor at the coded register - on 14 known
    coded posts its mean p_hate DROPPED - so one blended line would hide the
    thing a reader most needs to know.

    Days are Kenyan days: the audience is UTC+3 and an hour-of-day figure in UTC
    is actively misleading. `meta.timezone` states this.
    """
    df = _toxicity_frame(con, platform)
    d = df[df["in_kenya_scope"]].copy()

    # Clip to the observation window. Two separate artifacts to exclude:
    #
    # Timeline and hydration passes surface posts authored years before
    # collection began - 431 of them, spread thinly enough to manufacture 433
    # "suppressed days" that are not gaps in our observation at all.
    #
    # And the 14 days BEFORE collection started are only partially observed: a
    # post from that window is in the corpus only if one of the first searches
    # happened to reach it, so its denominator is a thin, biased sample. Measured
    # 2026-08-06, including them put a 16% spike on the first day against a ~5%
    # steady state. The floor is therefore the first collection, not the horizon
    # before it.
    start = _collection_start(con, platform)
    if start is not None and len(d):
        floor = pd.Timestamp(start).tz_convert("UTC")
        d = d[d["created_at"] >= floor]

    if d.empty:
        return {"denominator": "Kenya-scoped baseline posts", "series": [],
                "rolling7": [], "suppressed_days": [], "heatmap": [], "latest": None}

    local = d["created_at"].dt.tz_convert(tz)
    d["date"] = local.dt.floor("D").dt.strftime("%Y-%m-%d")
    d["dow"] = local.dt.dayofweek
    d["hour"] = local.dt.hour

    daily = d.groupby("date").agg(
        n=("label", "size"),
        off=("is_offensive", "sum"),
        hate=("is_hate", "sum"),
        coded=("coded_suspect", "sum"),
    ).sort_index()

    thin = daily[daily["n"] < MIN_DAY_N]
    kept = daily[daily["n"] >= MIN_DAY_N]

    series = [
        {
            "date": date,
            "n": int(r["n"]),
            "offensive": _rate(r["off"], int(r["n"])),
            "hate": _rate(r["hate"], int(r["n"])),
            "coded": _rate(r["coded"], int(r["n"])),
        }
        for date, r in kept.iterrows()
    ]

    # min_periods=3 matches the notebook: a mean over one or two days is not a
    # trend and should render as a gap rather than a confident line.
    roll = pd.DataFrame({
        "offensive": kept["off"] / kept["n"],
        "hate": kept["hate"] / kept["n"],
        "coded": kept["coded"] / kept["n"],
    }).rolling(7, min_periods=3).mean()
    rolling7 = [
        {
            "date": date,
            **{c: (None if pd.isna(r[c]) else round(float(r[c]), 6)) for c in roll.columns},
        }
        for date, r in roll.iterrows()
    ]

    cell = d.groupby(["dow", "hour"]).agg(
        n=("label", "size"), toxic=("explicit_toxic", "sum")
    ).reset_index()
    heatmap = [
        {
            "dow": int(r["dow"]), "hour": int(r["hour"]), "n": int(r["n"]),
            "p_toxic": round(float(r["toxic"]) / int(r["n"]), 6) if r["n"] else None,
            "suppressed": bool(r["n"] < MIN_DAY_N),
        }
        for _, r in cell.iterrows()
    ]

    n_total = int(len(d))
    # LEFT JOINed, so the column can be entirely absent on a corpus the NLI pass
    # has never touched.
    incitement_col = d.get("dehumanisation_score")
    n_incitement = int(incitement_col.notna().sum()) if incitement_col is not None else 0
    return {
        "denominator": (
            "Kenya-scoped baseline posts (measure.domain_bucket is not offdomain)"
        ),
        "min_day_n": MIN_DAY_N,
        "timezone": tz,
        # The daily `series` above is a RAW rate. It moves with collection
        # policy as well as with discourse, so anything presented as a trend
        # must come from here instead.
        "standardised": _composition_block(d),
        # The coded series reads as a flat zero, and that number needs its own
        # caveat rather than a reader concluding coded incitement is absent.
        # `coded_suspect` requires a documented lexicon hit corroborated by NLI,
        # and the NLI pass has only reached part of the corpus.
        "coded_coverage": {
            "posts": n_total,
            "with_incitement_scores": n_incitement,
            "share": round(n_incitement / n_total, 4) if n_total else None,
            "coded_suspect_total": int(d["coded_suspect"].sum()),
            "note": (
                "Coded incitement requires a documented NCIC/PeaceTech lexicon "
                "hit corroborated by the NLI pass. Where that pass has not run, "
                "a coded post cannot be counted - a low rate here is partly "
                "coverage, not only prevalence."
            ),
        },
        "series": series,
        "rolling7": rolling7,
        "suppressed_days": [
            {"date": date, "n": int(r["n"])} for date, r in thin.iterrows()
        ],
        "heatmap": heatmap,
        "latest": series[-1] if series else None,
    }


def _multiplex(edges: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-channel persisted edges into one multiplex edge list.

    Each layer is scaled to unit total weight before summing, mirroring
    `co.aggregate_layers(normalise="mass")`. Without it the ranking that picks
    which NET_MAX_EDGES to draw is decided entirely by the larger layer: measured
    2026-08-05, co_retweet carries 18,784 edges against co_reply's 28."""
    if edges.empty:
        return edges.assign(channels=None, n_channels=0)
    scaled = edges.copy()
    mass = scaled.groupby("channel")["weight"].transform("sum")
    scaled["w"] = scaled["weight"] / mass.where(mass > 0, 1.0)
    agg = scaled.groupby(["src", "dst"], as_index=False).agg(
        weight=("w", "sum"),
        min_gap=("min_gap", "min"),
        channels=("channel", lambda s: sorted(set(s))),
    )
    agg["n_channels"] = agg["channels"].str.len()
    return agg


def _cluster_rows(members: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster from the persisted per-member rows."""
    first = ["size", "n_channels", "internal_edge_share"]
    first += [c for c in ("name", "label", "channels") if c in members.columns]
    return (
        members.groupby("cluster_id", as_index=False)
        .agg({c: "first" for c in first})
        .assign(corr=lambda d: d["n_channels"] >= 2)
    )


def _corroboration_evidence(
    con: duckdb.DuckDBPyConnection,
    members: pd.DataFrame,
    metrics: pd.DataFrame,
    platform: str,
) -> dict:
    """What a reader needs in order to interpret the corroborated count.

    Two things, both of which the count alone hides.

    First the regime. Corroboration counts a CLUSTER whose internal edges span
    two channels, which a single bridge account can manufacture - so until
    bridge accounts are in the tens the number means "cannot be evaluated"
    rather than "nothing is coordinated". It sat at a 0/1 flicker for 23
    consecutive passes on exactly one bridge account, and the earlier decision
    to publish that zero as a structural finding was correct at the time and is
    now false: the census conversation-band fix made both real.

    Second the caveat that matters more. The corroborated tier is the LEAST
    Kenya-relevant one - reciprocal engagement pods are the most abundant form
    of coordination on the platform, and co_reply detects them best. Publishing
    the corroborated count without this invites reading engagement farming as
    election interference.
    """
    out: dict = {}
    if len(metrics):
        last = metrics[metrics["computed_at"] == metrics["computed_at"].max()]
        # Unbindable rather than NULL until a post-deploy pass writes them.
        for col in ("bridge_accounts", "shared_pairs"):
            if col in last.columns:
                v = last.iloc[0].get(col)
                out[col] = None if pd.isna(v) else int(v)

    out["regime_note"] = (
        "Read the corroborated cluster count next to bridge_accounts. One "
        "bridge account is enough to produce a corroborated cluster, so a low "
        "bridge count means the measurement cannot be evaluated, not that "
        "nothing is coordinated."
    )

    # The caveat ships whether or not the measurement behind it succeeds. If it
    # went missing on failure a reader would see a corroborated count with
    # nothing telling them it is mostly engagement farming, which is the whole
    # risk this key exists to cover.
    out["kenya_share"] = _cluster_kenya_share(con, members, platform) or None
    out["kenya_note"] = (
        "Corroborated clusters are the strongest evidence tier and currently "
        "the least on-topic: reciprocal engagement pods are the most abundant "
        "coordination on the platform and co_reply detects them best. Treat a "
        "corroborated cluster as evidence of coordination, not of "
        "election-related coordination."
    )
    return out


def _cluster_kenya_share(
    con: duckdb.DuckDBPyConnection, members: pd.DataFrame, platform: str
) -> dict:
    """Share of each tier's posts that reference Kenya, via the persisted
    `domain` column that `measure.attach_measurement_columns` writes."""
    if members.empty or "n_channels" not in members.columns:
        return {}
    try:
        posts = con.sql(
            f"""
            WITH lp AS (
                SELECT author_id, platform_post_id FROM {db.posts_source(platform)}
                QUALIFY row_number() OVER (
                    PARTITION BY platform, platform_post_id
                    ORDER BY collected_at DESC) = 1
            ), h AS (
                SELECT platform_post_id, domain FROM {db.hatespeech_source(platform)}
                QUALIFY row_number() OVER (
                    PARTITION BY platform_post_id ORDER BY scored_at DESC) = 1
            )
            SELECT lp.author_id, h.domain
            FROM lp JOIN h USING (platform_post_id)
            WHERE h.domain IS NOT NULL
            """
        ).df()
    except (duckdb.Error, AttributeError):
        # Loud, never silent: a failed measurement must not read as "no
        # off-domain clusters". The caveat text is published regardless.
        log.warning("coordination: Kenya-share query failed; publishing it as null")
        return {}
    if posts.empty:
        return {}

    tier = members.assign(corr=members["n_channels"] >= 2)[["author_id", "corr"]]
    joined = posts.merge(tier.drop_duplicates("author_id"), on="author_id", how="inner")
    if joined.empty:
        return {}

    out = {}
    for corr, label in ((True, "corroborated"), (False, "single_channel")):
        grp = joined[joined["corr"] == corr]
        if len(grp):
            out[label] = {
                "posts": int(len(grp)),
                "kenya": round(float((grp["domain"] == "kenya").mean()), 4),
            }
    return out


def build_coordination(con: duckdb.DuckDBPyConnection, platform: str = "x") -> dict:
    """Statistic C, read from the persisted run of record.

    Corroborated (>= 2 channels) and candidate clusters are kept in separate
    keys and never summed. That is not presentation preference: publishing one
    blended count would claim N validated networks when only the corroborated
    subset is defensible."""
    members = db.coordination_run_latest(con, "clusters", platform).df()
    edges = db.coordination_run_latest(
        con, "edges", platform, method=PERSISTED_METHOD
    ).df()
    metrics = db.coordination_metrics(con, platform).df()

    run: dict = {}
    if len(metrics):
        last = metrics[metrics["computed_at"] == metrics["computed_at"].max()]
        r = last.iloc[0]
        run = {
            "computed_at": str(r.get("computed_at")),
            "code_version": r.get("code_version"),
            "channels": sorted(set(last["channel"].dropna())),
            "method": r.get("method"),
            "resolution": _f(r, "resolution"),
            "min_size": int(_f(r, "min_size")),
            "lookback_days": int(_f(r, "lookback_days")),
        }

    if members.empty:
        return {
            "run": run,
            "headline": {
                "clusters_total": 0, "accounts_total": 0,
                "clusters_corroborated": 0, "accounts_corroborated": 0,
                "edges_total": 0, "edges_multichannel": 0,
                "shown_accounts": 0, "shown_clusters": 0,
            },
            "corroborated": {"clusters": [], "evidence": {}},
            "candidates": {"n": 0, "size_histogram": []},
            "network": {"nodes": [], "edges": [], "clusters": []},
            "share_series": _share_series(metrics),
        }

    clusters = _cluster_rows(members)
    agg = _multiplex(edges)
    triage = _triage_by_cluster(con, platform)
    verdicts = _verdicts_by_cluster(con, platform)
    for cid, v in verdicts.items():
        triage.setdefault(cid, {}).update(v)
    corr_ids = set(clusters.loc[clusters["corr"], "cluster_id"])
    handles = db.curated_handles(platform=platform)

    headline = {
        "clusters_total": int(len(clusters)),
        "accounts_total": int(members["author_id"].nunique()),
        "clusters_corroborated": len(corr_ids),
        "accounts_corroborated": int(members["cluster_id"].isin(corr_ids).sum()),
        "edges_total": int(len(agg)),
        "edges_multichannel": int((agg["n_channels"] >= 2).sum()) if len(agg) else 0,
    }

    drawable = clusters[clusters["size"] >= NET_MIN_SIZE].sort_values(
        ["corr", "size"], ascending=[False, False]
    ).head(NET_TOP_CLUSTERS)
    keep = set(drawable["cluster_id"])
    mem = members[members["cluster_id"].isin(keep)]
    in_net = set(mem["author_id"])
    e = agg[agg["src"].isin(in_net) & agg["dst"].isin(in_net)] if len(agg) else agg
    e = e.sort_values("weight", ascending=False).head(NET_MAX_EDGES)

    candidates = clusters[~clusters["corr"]]
    out = {
        "run": run,
        "headline": headline,
        "corroborated": {
            "clusters": _annotate(clusters[clusters["corr"]], None, handles, triage),
            "evidence": _corroboration_evidence(con, members, metrics, platform),
        },
        "candidates": {
            "n": int(len(candidates)),
            "size_histogram": [
                {"size": int(s), "n": int(n)}
                for s, n in candidates["size"].value_counts().sort_index().items()
            ],
        },
        "share_series": _share_series(metrics),
    }

    used = sorted(set(e["src"]) | set(e["dst"])) if len(e) else []
    if not used:
        out["headline"] |= {"shown_accounts": 0, "shown_clusters": 0}
        out["network"] = {"nodes": [], "edges": [], "clusters": []}
        return out

    node_idx = {a: i for i, a in enumerate(used)}
    cid_idx = {c: i for i, c in enumerate(sorted(keep))}
    a2c = dict(zip(mem["author_id"], mem["cluster_id"]))

    # Precompute a static layout so the browser only pans and zooms: no force
    # simulation, and therefore no graph library on the client.
    import igraph as ig

    g = ig.Graph(
        n=len(used),
        edges=[(node_idx[s], node_idx[d]) for s, d in zip(e["src"], e["dst"])],
    )
    layout = g.layout_drl() if g.ecount() else g.layout_circle()
    coords = layout.coords
    deg = g.degree()

    out["network"] = {
        "nodes": [
            {
                "id": i,
                "x": round(coords[i][0], 2),
                "y": round(coords[i][1], 2),
                "c": cid_idx.get(a2c.get(a), -1),
                "d": int(deg[i]),
                "corr": bool(a2c.get(a) in corr_ids),
            }
            for a, i in node_idx.items()
        ],
        "edges": [
            {"s": node_idx[s], "t": node_idx[d], "w": round(float(w), 4), "m": int(nc)}
            for s, d, w, nc in zip(e["src"], e["dst"], e["weight"], e["n_channels"])
        ],
        "clusters": _annotate(drawable, cid_idx, handles, triage),
    }
    out["headline"] |= {
        "shown_accounts": len(node_idx),
        "shown_clusters": len(cid_idx),
    }
    return out


def _safe_name(name, handles: set[str]) -> str | None:
    """Drop a cluster name that carries an account handle.

    Names are c-TF-IDF terms over member post text. `semantic._CLEAN` already
    strips `@mention` and `#hashtag` forms before tokenising, so the residual
    risk is a handle written as bare prose, which survives tokenisation and
    would put an identifiable account on a public page. Cheap to exclude, and
    the loss is one keyword on one cluster."""
    if not isinstance(name, str) or not name.strip():
        return None
    tokens = name.lower().split()
    if handles and any(t in handles for t in tokens):
        return None
    return name


# Triage columns lifted from `kind=scorecards` onto the cluster cards. All are
# per-CLUSTER aggregates - no handle, author id or post text among them - which
# is what lets them cross into a public payload at all.
#
# `hate_index` stays listed beside `inauthenticity_index` rather than folded
# into it: coordination and hate are different claims, and a reader who cannot
# see which one fired cannot act on either.
TRIAGE_COLUMNS = (
    "inauthenticity_index",
    "near_dup_rate",
    "self_amplification",
    "self_amplification_z",
    "hate_index",
    "topic_entropy",
)


def _triage_by_cluster(con: duckdb.DuckDBPyConnection, platform: str) -> dict:
    """Per-cluster triage scores of the newest scorecard run, keyed by cluster_id.

    Absent by design on a fresh bucket and after any run where scoring failed -
    scorecards ride a slower timer than the coordination pass that produces the
    clusters, so a lag of up to one scoring interval is normal, and an empty
    prefix raises rather than returning no rows."""
    try:
        if not db.prefix_readable(con, db.coordination_source("scorecards", platform)):
            return {}
        cards = db.coordination_run_latest(con, "scorecards", platform).df()
    except Exception:
        # Never fatal. Triage scores decorate the cluster cards; the counts and
        # the network are the payload, and losing the decoration must not cost
        # the site an hourly build.
        log.exception("dashboard: scorecards unreadable; cluster cards stay bare")
        return {}
    if cards.empty:
        return {}
    cols = [c for c in TRIAGE_COLUMNS if c in cards.columns]
    return {
        int(r["cluster_id"]): {
            c: round(float(r[c]), 3) for c in cols if pd.notna(r[c])
        }
        for _, r in cards.iterrows()
    }


# What a reader JUDGED the coordination to be. `rationale` is included and the
# free-text `what_would_change_this` is not: the first is the reason a reader
# needs to weigh the call, the second is an internal note, and every extra free
# text field is another way for a handle or a quoted post to reach a public
# payload. `tests/test_dashboard.py` walks the built payload for exactly that.
VERDICT_COLUMNS = ("cluster_type", "kenya_relevant", "confidence", "rationale",
                   "adjudicator")


def _verdicts_by_cluster(con: duckdb.DuckDBPyConnection, platform: str) -> dict:
    """Adjudications of the newest verdict run, keyed by cluster_id.

    Absent until layer three runs at all, and absent for any cluster the queue
    did not reach - adjudication is capped, so most clusters legitimately have
    no verdict. A missing verdict must read as "not yet judged", never as
    "judged and found harmless"."""
    try:
        if not db.prefix_readable(con, db.coordination_source("verdicts", platform)):
            return {}
        rows = db.coordination_run_latest(con, "verdicts", platform).df()
    except Exception:
        log.exception("dashboard: verdicts unreadable; cluster cards stay unjudged")
        return {}
    if rows.empty:
        return {}
    cols = [c for c in VERDICT_COLUMNS if c in rows.columns]
    return {
        int(r["cluster_id"]): {
            f"verdict_{c}" if c != "cluster_type" else "verdict": r[c]
            for c in cols if pd.notna(r[c])
        }
        for _, r in rows.iterrows()
    }


def _annotate(
    clusters: pd.DataFrame,
    cid_idx: dict | None,
    handles: set[str],
    triage: dict | None = None,
) -> list[dict]:
    """Cluster cards. `cid_idx` remaps to the opaque drawing index when the
    cluster is in the network; without it the raw per-run cluster_id is used,
    which is fine because it identifies nothing outside this run."""
    out = []
    for _, r in clusters.iterrows():
        cid = r["cluster_id"]
        card = {
            "id": cid_idx[cid] if cid_idx is not None else int(cid),
            "size": int(_f(r, "size")),
            "n_channels": int(_f(r, "n_channels")),
            "channels": list(r.get("channels") or []),
            "internal_edge_share": round(_f(r, "internal_edge_share"), 3),
            "name": _safe_name(r.get("name"), handles),
            "corr": bool(r.get("corr", False)),
        }
        # Merged rather than defaulted to 0: a missing score means the cluster
        # was not measured, which must not read as "measured and clean".
        card |= (triage or {}).get(int(cid), {})
        out.append(card)
    return out


def _share_series(metrics: pd.DataFrame) -> list[dict]:
    """Detection counters over time, carrying `code_version` on every point.

    The site segments on it rather than drawing one continuous line: these
    counters are not comparable across a methodology change, and smoothing over
    one is the exact error that made the census-tuning Q2 figure wrong."""
    if metrics.empty:
        return []
    per_run = (
        metrics.groupby("computed_at", as_index=False)
        .agg({
            "code_version": "first",
            "n_clusters": "first",
            "n_accounts": "first",
            "n_corroborated_clusters": "first",
            "n_corroborated_accounts": "first",
        })
        .sort_values("computed_at")
    )
    return [
        {
            "computed_at": str(r["computed_at"]),
            "code_version": r["code_version"],
            "n_clusters": int(_f(r, "n_clusters")),
            "n_accounts": int(_f(r, "n_accounts")),
            "n_corroborated_clusters": int(_f(r, "n_corroborated_clusters")),
            "n_corroborated_accounts": int(_f(r, "n_corroborated_accounts")),
        }
        for _, r in per_run.iterrows()
    ]


def build_summary(con: duckdb.DuckDBPyConnection, platform: str = "x") -> dict:
    """The public blob. `toxicity`, `share_of_voice` and `targeting` are null
    until their phases land; the site reserves layout for them rather than
    shifting when they appear."""
    timings: dict = {}
    with _timed(timings, "meta"):
        meta = build_meta(con, platform)
    with _timed(timings, "coordination"):
        coordination = build_coordination(con, platform)
    with _timed(timings, "toxicity"):
        toxicity = build_toxicity(con, platform)
    return {
        "schema_version": SCHEMA_VERSION,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform,
        "meta": meta,
        "coordination": coordination,
        "toxicity": toxicity,
        "share_of_voice": None,
        "targeting": None,
        "build": {"timings_s": timings},
    }


def _r2_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def persist_summary(
    summary: dict, key: str = SUMMARY_KEY, bucket: str = DASHBOARD_BUCKET
) -> str:
    """Write to the stable key plus a timestamped archive. `computed_at` also
    goes on the object metadata so the Worker's health route can check freshness
    with a HEAD instead of pulling the body."""
    body = json.dumps(summary, default=str, separators=(",", ":")).encode()
    now = datetime.now(timezone.utc)
    kind = key.rsplit("/", 1)[-1].removesuffix(".json")
    client = _r2_client()
    meta = {"computed_at": summary.get("computed_at", now.isoformat())}
    client.put_object(
        Bucket=bucket, Key=key, Body=body,
        ContentType="application/json", Metadata=meta,
    )
    archive = f"dashboard/archive/kind={kind}/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.json"
    client.put_object(
        Bucket=bucket, Key=archive, Body=body,
        ContentType="application/json", Metadata=meta,
    )
    log.info("wrote %s (%d bytes) + %s", key, len(body), archive)
    return key


def run_once(con: duckdb.DuckDBPyConnection | None = None, platform: str = "x") -> str:
    con = con or connect()
    return persist_summary(build_summary(con, platform=platform))


def run_loop(interval_s: int = DASHBOARD_INTERVAL_S, platform: str = "x") -> None:
    """Forever: rebuild, then sleep. A per-cycle failure is logged and retried so
    a transient R2 hiccup never kills the container."""
    while True:
        try:
            run_once(platform=platform)
        except Exception:
            log.exception("dashboard refresh failed")
        time.sleep(interval_s)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description="Build the kenya.tomfr.net summary.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--loop", action="store_true", help="run forever")
    mode.add_argument("--once", action="store_true", help="build + write once")
    ap.add_argument("--platform", default="x")
    ap.add_argument("--dry-run", action="store_true", help="build, print, write nothing")
    args = ap.parse_args()
    if args.loop:
        run_loop(platform=args.platform)
    elif args.dry_run:
        print(json.dumps(build_summary(connect(), platform=args.platform), default=str))
    else:
        print(f"dashboard: wrote {run_once(platform=args.platform)}")


if __name__ == "__main__":
    main()

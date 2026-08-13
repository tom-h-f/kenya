"""Connect DuckDB to the R2 election dataset and expose convenience relations.

    from kma.db import connect, latest_posts
    con = connect()
    df = latest_posts(con).pl()      # polars DataFrame
    pdf = latest_posts(con).df()     # pandas DataFrame
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

MONOREPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(MONOREPO_ROOT / ".env")

BUCKET = os.getenv("R2_BUCKET", "kenya-monitor-2027")


def connect() -> duckdb.DuckDBPyConnection:
    """A DuckDB connection with httpfs loaded and an R2 secret configured."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(
        "CREATE OR REPLACE SECRET r2 (TYPE r2, KEY_ID ?, SECRET ?, ACCOUNT_ID ?)",
        [
            os.environ["R2_ACCESS_KEY_ID"],
            os.environ["R2_SECRET_ACCESS_KEY"],
            os.environ["R2_ACCOUNT_ID"],
        ],
    )
    return con


def posts_source(platform: str = "*", type: str = "*") -> str:
    """A read_parquet(...) expression usable directly in SQL FROM clauses."""
    glob = f"r2://{BUCKET}/posts/platform={platform}/type={type}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


# Collection provenance. Baseline partitions sample the discourse; targeted ones
# chase hate speech and coordination, so they oversample the toxic tail by
# construction. Any prevalence or rate measurement must scope to `baseline`.
BASELINE_TYPES = ("search", "timeline", "replies", "hydrated")
TARGETED_TYPES = (
    "hate_search",
    "hate_target_search",
    "hate_timeline",
    "hate_replies",
    "hate_hydrated",
    "cib_timeline",
    # Timelines of accounts discovered by the retweeter census, sampled at
    # RANDOM rather than by cluster membership. Targeted, so it stays out of
    # every prevalence rate: census-discovered accounts are retweeters of
    # banded objects, not a sample of the population. It does feed coordination
    # traces, which is the point - see `coordination.corroborate`.
    "census_timeline",
)
KNOWN_TYPES = BASELINE_TYPES + TARGETED_TYPES
SCOPES = ("all", "baseline", "targeted")


# The curated target list lives on the collector side. Read as DATA, not code -
# the analysis project takes no dependency on `kenya_monitor`.
CURATED_TARGETS_PATH = MONOREPO_ROOT / "collector" / "config" / "targets.yaml"


def curated_handles(path: Path | None = None, platform: str = "x") -> set[str]:
    """Lowercased handles the collector tracks deliberately (`targets.yaml`).

    Used to repair a historical misclassification: until 2026-08-01 the accounts
    pass wrote BOTH curated targets and coordination-promoted accounts into
    `type=timeline`, a baseline partition. Promoted accounts are selected for
    looking coordinated, so their timelines are targeted collection and do not
    belong in a prevalence denominator. Measured leak: 41,595 posts from 407
    accounts, 6.7% of the baseline corpus.

    Returns an empty set when the file is unavailable, which disables the
    correction rather than silently dropping data - see `leak_corrected()`.
    """
    import yaml

    try:
        raw = yaml.safe_load((path or CURATED_TARGETS_PATH).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return set()
    return {h.strip().lower() for h in (raw.get(platform) or {}).get("accounts") or []}


def leak_corrected(path: Path | None = None) -> bool:
    """Whether the timeline-leak correction can be applied. Publish this: a rate
    computed without it carries a known 6.7% contamination."""
    return bool(curated_handles(path))


def effective_type_expr(
    curated: set[str] | None = None, col: str = "first_type", handle_col: str = "author_handle"
) -> str:
    """`first_type`, with pre-2026-08-01 promoted-account timelines reclassified
    to `cib_timeline`.

    A reclassification rather than an exclusion, so `baseline` and `targeted`
    stay complementary and every post still lands in exactly one scope. Only
    `timeline` is affected: it is the sole baseline partition the accounts pass
    ever wrote to."""
    curated = curated_handles() if curated is None else curated
    if not curated:
        return col
    return f"""CASE
        WHEN {col} = 'timeline' AND lower({handle_col}) NOT IN ({_sql_list(sorted(curated))})
            THEN 'cib_timeline'
        ELSE {col}
    END"""


def _sql_list(values) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def first_seen_types_cte(platform: str = "*", name: str = "_first_seen") -> str:
    """A CTE body mapping platform_post_id -> the partition that FIRST surfaced it.

    Prevalence must scope on this, never on the latest row's `type`. `latest_*`
    keeps the newest `collected_at`, so a post first found by a baseline search
    and later re-collected by a targeted one has its `type` flipped - and it is
    by construction a post the targeted query matched, i.e. the toxic tail.
    Scoping on the latest row would quietly drain those posts out of the
    baseline denominator and read as a falling trend.
    """
    return f"""{name} AS (
        SELECT platform, platform_post_id,
               min_by(type, collected_at) AS first_type,
               min(collected_at)          AS first_collected_at
        FROM {posts_source(platform)}
        GROUP BY platform, platform_post_id
    )"""


def scope_predicate(scope: str = "baseline", col: str = "first_type") -> str:
    """A SQL predicate selecting one collection scope by first-seen type.

    An unrecognised type raises rather than falling into a bucket: a future
    partition must not be able to silently pollute the baseline denominator,
    nor to vanish from both scopes.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r} (expected one of {SCOPES})")
    if scope == "all":
        return "TRUE"
    wanted = BASELINE_TYPES if scope == "baseline" else TARGETED_TYPES
    other = TARGETED_TYPES if scope == "baseline" else BASELINE_TYPES
    return f"""CASE
        WHEN {col} IN ({_sql_list(wanted)}) THEN TRUE
        WHEN {col} IN ({_sql_list(other)}) THEN FALSE
        ELSE error('unknown post type ' || coalesce({col}, 'NULL')
                   || ' - add it to kma.db.BASELINE_TYPES or TARGETED_TYPES')
    END"""


def posts(con: duckdb.DuckDBPyConnection, platform: str = "*", type: str = "*"):
    """All collected post rows (every engagement snapshot, not deduped)."""
    return con.sql(f"SELECT * FROM {posts_source(platform, type)}")


def latest_posts(
    con: duckdb.DuckDBPyConnection,
    platform: str = "*",
    type: str = "*",
    scope: str = "all",
):
    """One row per post: its most recently collected state.

    `scope` filters by collection provenance - "baseline" for any prevalence or
    rate measurement, "targeted" to inspect what the hate/CIB passes pulled,
    "all" (default) for the whole corpus. See `first_seen_types_cte`.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r} (expected one of {SCOPES})")
    dedup = """
        QUALIFY row_number() OVER (
            PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
        ) = 1
    """
    if scope == "all":
        return con.sql(f"SELECT * FROM {posts_source(platform, type)} {dedup}")
    eff = effective_type_expr(col="fs.first_type", handle_col="p.author_handle")
    return con.sql(
        f"""
        WITH {first_seen_types_cte(platform)}
        SELECT p.*, fs.first_collected_at, {eff} AS first_type
        FROM {posts_source(platform, type)} p
        JOIN _first_seen fs USING (platform, platform_post_id)
        WHERE {scope_predicate(scope, eff)}
        {dedup}
        """
    )


def metrics_source(platform: str = "*") -> str:
    glob = f"r2://{BUCKET}/metrics/platform={platform}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def authors_source(platform: str = "*") -> str:
    glob = f"r2://{BUCKET}/authors/platform={platform}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def latest_authors(con: duckdb.DuckDBPyConnection, platform: str = "*"):
    """One row per author: their most recently collected profile snapshot."""
    return con.sql(
        f"""
        SELECT * FROM {authors_source(platform)}
        QUALIFY row_number() OVER (
            PARTITION BY platform, platform_user_id ORDER BY collected_at DESC
        ) = 1
        """
    )


def embeddings_source(platform: str = "*", model: str = "*") -> str:
    glob = f"r2://{BUCKET}/embeddings/platform={platform}/model={model}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def latest_embeddings(con: duckdb.DuckDBPyConnection, platform: str = "*", model: str = "*"):
    """One embedding row per post (latest), for a given model."""
    return con.sql(
        f"""
        SELECT * FROM {embeddings_source(platform, model)}
        QUALIFY row_number() OVER (
            PARTITION BY platform_post_id, model ORDER BY embedded_at DESC
        ) = 1
        """
    )


def labels_source(platform: str = "*") -> str:
    glob = f"r2://{BUCKET}/labels/platform={platform}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def latest_labels(con: duckdb.DuckDBPyConnection, platform: str = "*"):
    """One sentiment/emotion label row per post (latest)."""
    return con.sql(
        f"""
        SELECT * FROM {labels_source(platform)}
        QUALIFY row_number() OVER (
            PARTITION BY platform_post_id ORDER BY labeled_at DESC
        ) = 1
        """
    )


def incitement_source(platform: str = "*") -> str:
    glob = f"r2://{BUCKET}/incitement/platform={platform}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def latest_incitement(con: duckdb.DuckDBPyConnection, platform: str = "*"):
    """One incitement-score row per post (latest). Separate prefix from labels/
    on purpose: latest_labels dedups on platform_post_id alone, so a second
    writer under labels/ would shadow sentiment/emotion rows."""
    return con.sql(
        f"""
        SELECT * FROM {incitement_source(platform)}
        QUALIFY row_number() OVER (
            PARTITION BY platform_post_id ORDER BY scored_at DESC
        ) = 1
        """
    )


def hatespeech_source(platform: str = "*") -> str:
    glob = f"r2://{BUCKET}/hatespeech/platform={platform}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def latest_hatespeech(con: duckdb.DuckDBPyConnection, platform: str = "*"):
    """One hate-speech-score row per post (latest). Separate prefix from labels/
    and incitement/ on purpose: latest_* dedups on platform_post_id alone, so a
    second writer under an existing prefix would shadow its rows."""
    return con.sql(
        f"""
        SELECT * FROM {hatespeech_source(platform)}
        QUALIFY row_number() OVER (
            PARTITION BY platform_post_id ORDER BY scored_at DESC
        ) = 1
        """
    )


def engagements_source(platform: str = "*") -> str:
    glob = f"r2://{BUCKET}/engagements/platform={platform}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def latest_engagements(con: duckdb.DuckDBPyConnection, platform: str = "*"):
    """One row per (post, user, kind) engagement edge (latest snapshot). Incidence
    only - the platform does not expose when a retweet happened."""
    return con.sql(
        f"""
        SELECT * FROM {engagements_source(platform)}
        QUALIFY row_number() OVER (
            PARTITION BY platform, platform_post_id, platform_user_id, kind
            ORDER BY collected_at DESC
        ) = 1
        """
    )


def follows_source(platform: str = "*") -> str:
    glob = f"r2://{BUCKET}/follows/platform={platform}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def latest_follows(con: duckdb.DuckDBPyConnection, platform: str = "*"):
    """One row per (follower, followed) edge (latest snapshot)."""
    return con.sql(
        f"""
        SELECT * FROM {follows_source(platform)}
        QUALIFY row_number() OVER (
            PARTITION BY platform, follower_id, followed_id ORDER BY collected_at DESC
        ) = 1
        """
    )


def coordination_source(
    kind: str = "edges",
    platform: str = "*",
    channel: str = "*",
    method: str = "*",
) -> str:
    """A read_parquet(...) expression for persisted coordination artifacts."""
    if kind == "edges":
        glob = (
            f"r2://{BUCKET}/coordination/platform={platform}/kind=edges"
            f"/channel={channel}/method={method}/dt=*/run=*.parquet"
        )
    elif kind == "clusters":
        glob = f"r2://{BUCKET}/coordination/platform={platform}/kind=clusters/dt=*/run=*.parquet"
    elif kind == "run_metrics":
        glob = (
            f"r2://{BUCKET}/coordination/platform={platform}"
            f"/kind=run_metrics/dt=*/run=*.parquet"
        )
    else:
        raise ValueError(f"unknown coordination kind {kind!r}")
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def coordination_run_latest(
    con: duckdb.DuckDBPyConnection,
    kind: str = "clusters",
    platform: str = "x",
    channel: str = "*",
    method: str = "*",
):
    """Rows of the most recent persisted coordination run, as a coherent set.

    `dense_rank` over `computed_at` rather than the per-entity `row_number` the
    `latest_coordination_*` helpers use, which is the same distinction
    `latest_stories` already draws. Anything published to a reader needs this
    one: a sticky union of several runs reports a cluster count that no single
    run ever produced.

    `kind="edges"` partitions by (channel, method) because `persist_edges` takes
    its own `now` per call and is invoked once per channel x method, so every
    partition carries a slightly different `computed_at`. Ranking them together
    would return whichever channel happened to be written last and silently drop
    the rest. The cost is that if a run dies midway, the newest run per partition
    can straddle two runs - visible as disagreeing `computed_at` values, which is
    better than a missing channel."""
    src = coordination_source(kind, platform, channel, method)
    partition = "PARTITION BY channel, method " if kind == "edges" else ""
    return con.sql(
        f"""
        SELECT * FROM {src}
        QUALIFY dense_rank() OVER ({partition}ORDER BY computed_at DESC) = 1
        """
    )


def latest_coordination_edges(
    con: duckdb.DuckDBPyConnection,
    platform: str = "x",
    channel: str = "*",
    method: str = "*",
):
    """STICKY UNION across runs - not the current edge set. Use
    `coordination_run_latest("edges", ...)` for anything a reader sees.

    `row_number()` per (src, dst, channel, method) keeps the newest row FOR EACH
    PAIR, so a pair that validated once and never again survives forever. The
    result is a monotonic high-water mark that only ever grows: measured
    2026-08-12 at 134,800 co_retweet edges against a true latest-run 97,240.

    Kept for deliberate cross-run archaeology ("was this pair ever validated"),
    which is the only question it answers correctly."""
    return con.sql(
        f"""
        SELECT * FROM {coordination_source('edges', platform, channel, method)}
        QUALIFY row_number() OVER (
            PARTITION BY src, dst, channel, method ORDER BY computed_at DESC
        ) = 1
        """
    )


def latest_coordination_clusters(con: duckdb.DuckDBPyConnection, platform: str = "x"):
    """STICKY UNION across runs, and badly so - use `coordination_run_latest`.

    `cluster_id` is a per-run Leiden label (`coordination.communities` returns
    `part.membership`) with no stability across passes, so partitioning on it
    never collapses to one clustering: the same author appears once per cluster
    id they have ever been assigned. Measured 2026-08-12: 44,585 member rows
    across 1,078 cluster ids drawn from 47 distinct passes, against a true latest
    run of 1,043 rows and 128 clusters - 42.7x inflation.

    There is no correct way to read a single run out of this helper. It is
    retained only because it is re-exported from `kma/__init__.py`."""
    return con.sql(
        f"""
        SELECT * FROM {coordination_source('clusters', platform)}
        QUALIFY row_number() OVER (
            PARTITION BY cluster_id, author_id ORDER BY computed_at DESC
        ) = 1
        """
    )


def coordination_run_latest(
    con: duckdb.DuckDBPyConnection,
    kind: str = "clusters",
    platform: str = "x",
    channel: str = "*",
    method: str = "*",
):
    """Rows of the most recent persisted coordination run, as a coherent set.

    `dense_rank` over `computed_at` rather than the per-entity `row_number` the
    `latest_coordination_*` helpers use, which is the same distinction
    `latest_stories` already draws. Anything published to a reader needs this
    one: a sticky union of several runs would report a cluster count that no
    single run ever produced.

    `kind="edges"` partitions by (channel, method) because `persist_edges` takes
    its own `now` per call and is invoked once per channel x method, so every
    partition carries a slightly different `computed_at`. Ranking them together
    would return whichever channel happened to be written last and silently drop
    the rest. The cost is that if a run dies midway, the newest run per partition
    can straddle two runs - visible as disagreeing `computed_at` values, which is
    better than a missing channel."""
    src = coordination_source(kind, platform, channel, method)
    partition = "PARTITION BY channel, method " if kind == "edges" else ""
    return con.sql(
        f"""
        SELECT * FROM {src}
        QUALIFY dense_rank() OVER ({partition}ORDER BY computed_at DESC) = 1
        """
    )


def coordination_metrics(con: duckdb.DuckDBPyConnection, platform: str = "x"):
    """Every persisted coordination-pass counter, oldest first.

    Deliberately NOT a `latest_*` helper. The other coordination artifacts are
    state - you want the newest. This one is a series: the questions it exists
    to answer ("did corroborated clusters move when the census changed") are
    only answerable across runs, and reading just the newest row reproduces the
    exact mistake that made the census-tuning Q2 figure wrong."""
    return con.sql(
        f"SELECT * FROM {coordination_source('run_metrics', platform)} "
        "ORDER BY computed_at, channel"
    )


def census_runs_source(platform: str = "*") -> str:
    """A read_parquet(...) expression for collector-written census-pass counters.

    Written by `kenya_monitor.storage.write_census_run`. Joining this to
    `coordination_metrics` is how census supply (candidates, backlog) is read
    against detection outcome (pairable accounts, corroborated clusters)."""
    glob = f"r2://{BUCKET}/census_runs/platform={platform}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def census_runs(con: duckdb.DuckDBPyConnection, platform: str = "x"):
    """Every census pass's counters, oldest first. A series, not a state - see
    `coordination_metrics`."""
    return con.sql(
        f"SELECT * FROM {census_runs_source(platform)} ORDER BY collected_at"
    )


def collection_runs_source(platform: str = "*") -> str:
    """A read_parquet(...) expression for the collector's per-query audit trail.

    Written by `kenya_monitor.storage.write_collection_run`. Carries the
    *rendered* query string, which is recorded nowhere else - so this is the only
    way to answer "what did we actually ask X for" after the fact."""
    glob = f"r2://{BUCKET}/collection_runs/platform={platform}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def collection_runs(con: duckdb.DuckDBPyConnection, platform: str = "x"):
    """Every collection pass's record, oldest first. A series, not a state - see
    `coordination_metrics`. `run_id` is unique per pass, so there is nothing to
    dedupe."""
    return con.sql(
        f"SELECT * FROM {collection_runs_source(platform)} ORDER BY collected_at"
    )


def stories_source(platform: str = "*") -> str:
    """A read_parquet(...) expression for persisted story artifacts (Phase 4)."""
    glob = f"r2://{BUCKET}/stories/platform={platform}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def latest_stories(con: duckdb.DuckDBPyConnection, platform: str = "x"):
    """Rows of the most recent persisted stories run for a platform."""
    return con.sql(
        f"""
        SELECT * FROM {stories_source(platform)}
        QUALIFY dense_rank() OVER (ORDER BY computed_at DESC) = 1
        """
    )

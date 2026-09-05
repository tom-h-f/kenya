"""Per-cluster evidence packages for human or LLM adjudication.

The coordination layer answers *do these accounts act together* and answers it
well. It cannot answer *why*, and that is not a tuning failure: the framework
`coordination.py` implements is intent-agnostic by construction, and the one
statistical discriminator we tried was measured against confirmed operations
and refuted (see analysis/investigations/2026-08-15-io-ground-truth). Intent
lives in the content, so it needs a reader.

A scorecard is numbers - `inauthenticity_index`, `near_dup_rate`, timing - and
there is nothing in it to reason about. A dossier is what a reader would
actually need to judge a cluster:

    what they jointly amplified, in their own words
    what they themselves post
    whose content they push outward
    how the accounts were provisioned
    how tightly the co-action is timed

PRIVATE BY CONSTRUCTION. Unlike everything in `dashboard.py`, this deliberately
carries post text and account handles, because an adjudicator cannot work
without them. It must never reach `DASHBOARD_BUCKET`, and no field here may be
copied into `build_summary`. `tests/test_dossier.py` asserts the dashboard
payload stays clean; keep it that way.
"""

from __future__ import annotations

import logging

import duckdb
import pandas as pd

from kma.coordination import _latest_posts_cte
from kma.db import engagements_source, hatespeech_source

log = logging.getLogger("kma")

DEFAULT_MAX_POSTS = 8
DEFAULT_MAX_OBJECTS = 8
DEFAULT_MAX_TARGETS = 8


def _member_table(con: duckdb.DuckDBPyConnection, members: pd.DataFrame) -> None:
    con.execute("DROP TABLE IF EXISTS _dos_members")
    con.register("_dos_members_df", members[["cluster_id", "author_id"]].drop_duplicates())
    con.execute("CREATE TEMP TABLE _dos_members AS SELECT * FROM _dos_members_df")
    con.unregister("_dos_members_df")


def _posts_table(con: duckdb.DuckDBPyConnection, platform: str) -> None:
    """Member posts only. `_latest_posts_cte` gets the author scope so the
    dedup window runs over ~1,200 accounts rather than the whole corpus."""
    con.execute("DROP TABLE IF EXISTS _dos_scope")
    con.execute(
        "CREATE TEMP TABLE _dos_scope AS SELECT DISTINCT author_id FROM _dos_members"
    )
    con.execute("DROP TABLE IF EXISTS _dos_posts")
    con.execute(
        "CREATE TEMP TABLE _dos_posts AS "
        + _latest_posts_cte(platform, author_scope="_dos_scope")
    )


def representative_posts(
    con: duckdb.DuckDBPyConnection, max_posts: int = DEFAULT_MAX_POSTS
) -> pd.DataFrame:
    """The most-amplified post per member, capped per cluster.

    Ranked by engagement rather than sampled at random: a reader has a handful
    of posts to form a view from, and the ones that travelled are the ones the
    cluster's behaviour is actually about. One post per author first, so a
    single prolific member cannot fill the whole exhibit."""
    return con.sql(
        f"""
        WITH scored AS (
            SELECT m.cluster_id, p.author_id, p.author_handle, p.text, p.created_at,
                   COALESCE(p.like_count, 0) + COALESCE(p.repost_count, 0)
                   + COALESCE(p.reply_count, 0) + COALESCE(p.quote_count, 0) AS engagement
            FROM _dos_members m
            JOIN _dos_posts p ON p.author_id = m.author_id
            WHERE p.text IS NOT NULL AND length(trim(p.text)) > 0
              AND COALESCE(p.is_repost, FALSE) = FALSE
        ), per_author AS (
            SELECT * FROM scored
            QUALIFY row_number() OVER (
                PARTITION BY cluster_id, author_id ORDER BY engagement DESC) = 1
        )
        SELECT * FROM per_author
        QUALIFY row_number() OVER (
            PARTITION BY cluster_id ORDER BY engagement DESC) <= {int(max_posts)}
        """
    ).df()


def _acts_table(con: duckdb.DuckDBPyConnection, platform: str) -> None:
    """The cluster's retweet acts, and a scoped read of the objects they hit.

    Both object queries need post rows for things the cluster amplified, which
    are authored by ANYONE - so they cannot be scoped by member author. They can
    be scoped by post id, and must be: reading the whole posts prefix twice more
    took a four-cluster build past 22 minutes. The semi-join goes INSIDE
    `_latest_posts_cte`'s subquery, ahead of its QUALIFY, for the same reason it
    does everywhere else - QUALIFY runs after the window, so filtering outside
    deduplicates the entire corpus first and then throws it away."""
    con.execute("DROP TABLE IF EXISTS _dos_acts")
    con.execute(
        f"""
        CREATE TEMP TABLE _dos_acts AS
        SELECT DISTINCT m.cluster_id, e.platform_post_id, e.platform_user_id
        FROM {engagements_source(platform)} e
        JOIN _dos_members m ON m.author_id = e.platform_user_id
        WHERE e.kind = 'retweet'
        """
    )
    con.execute("DROP TABLE IF EXISTS _dos_obj_ids")
    con.execute(
        "CREATE TEMP TABLE _dos_obj_ids AS "
        "SELECT DISTINCT platform_post_id FROM _dos_acts"
    )
    con.execute("DROP TABLE IF EXISTS _dos_objects")
    con.execute(
        "CREATE TEMP TABLE _dos_objects AS "
        # No lookback: an object the cluster amplified last week can easily have
        # been AUTHORED before the coordination window opened, and dropping it
        # would blank the most-shared exhibit in the packet.
        + _latest_posts_cte(platform, lookback_days=0, post_scope="_dos_obj_ids")
    )


def shared_objects(
    con: duckdb.DuckDBPyConnection,
    platform: str,
    max_objects: int = DEFAULT_MAX_OBJECTS,
) -> pd.DataFrame:
    """What the cluster JOINTLY amplified, with the text of the thing amplified.

    This is the evidence the detector actually fired on - the objects two or
    more members retweeted - so it is the first thing a reader should see. An
    object whose post was never collected is dropped rather than shown as a
    bare id: an unreadable exhibit is worse than one fewer exhibit."""
    return con.sql(
        f"""
        WITH agg AS (
            SELECT cluster_id, platform_post_id,
                   count(DISTINCT platform_user_id) AS n_members
            FROM _dos_acts GROUP BY 1, 2
            HAVING count(DISTINCT platform_user_id) >= 2
        )
        SELECT a.cluster_id, a.platform_post_id, a.n_members,
               p.author_handle AS object_author, p.text AS object_text,
               p.created_at AS object_created_at
        FROM agg a JOIN _dos_objects p USING (platform_post_id)
        WHERE p.text IS NOT NULL
        QUALIFY row_number() OVER (
            PARTITION BY a.cluster_id ORDER BY a.n_members DESC) <= {int(max_objects)}
        """
    ).df()


def amplification_targets(
    con: duckdb.DuckDBPyConnection,
    platform: str,
    max_targets: int = DEFAULT_MAX_TARGETS,
) -> pd.DataFrame:
    """Whose content the cluster pushes, ranked by acts.

    Often the single most diagnostic field. A cluster amplifying one politician
    reads very differently from one amplifying its own members, which reads
    differently again from one amplifying whatever is trending."""
    return con.sql(
        f"""
        WITH targeted AS (
            SELECT a.cluster_id, p.author_id AS target_id,
                   any_value(p.author_handle) AS target_handle,
                   count(*) AS acts
            FROM _dos_acts a JOIN _dos_objects p USING (platform_post_id)
            GROUP BY 1, 2
        )
        SELECT t.*,
               EXISTS (
                   SELECT 1 FROM _dos_members m
                   WHERE m.cluster_id = t.cluster_id AND m.author_id = t.target_id
               ) AS target_is_member
        FROM targeted t
        QUALIFY row_number() OVER (
            PARTITION BY cluster_id ORDER BY acts DESC) <= {int(max_targets)}
        """
    ).df()


def provenance(con: duckdb.DuckDBPyConnection, platform: str) -> pd.DataFrame:
    """How the accounts were provisioned, as per-cluster aggregates.

    Aggregates rather than per-account rows on purpose: the question a reader
    is answering is "were these set up together", which is a property of the
    group. Individual account profiling is what `authenticity.py` is for."""
    from kma.db import authors_source

    # The SEMI JOIN is inside the CTE, ahead of the QUALIFY. Outside it, DuckDB
    # deduplicates all ~461k known accounts to profile the ~300 in the clusters:
    # measured 2026-08-15 at 383s of a 1,370s build, 28% of the whole thing.
    return con.sql(
        f"""
        WITH la AS (
            SELECT * FROM {authors_source(platform)}
            SEMI JOIN (SELECT DISTINCT author_id FROM _dos_members) s
                   ON s.author_id = platform_user_id
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_user_id ORDER BY collected_at DESC) = 1
        )
        SELECT m.cluster_id,
               count(*) AS accounts_profiled,
               min(la.created_at) AS earliest_account,
               max(la.created_at) AS latest_account,
               date_diff('day', min(la.created_at), max(la.created_at)) AS creation_span_days,
               median(la.followers_count) AS median_followers,
               median(la.following_count) AS median_following,
               avg(CASE WHEN la.bio IS NULL OR la.bio = '' THEN 1.0 ELSE 0.0 END) AS share_empty_bio,
               avg(CASE WHEN la.profile_image_url ILIKE '%default_profile%'
                        THEN 1.0 ELSE 0.0 END) AS share_default_image,
               count(DISTINCT la.profile_image_url) AS distinct_profile_images
        FROM _dos_members m JOIN la ON la.platform_user_id = m.author_id
        GROUP BY 1
        """
    ).df()


def kenya_share(con: duckdb.DuckDBPyConnection, platform: str) -> pd.DataFrame:
    """Lexicon Kenya-relevance per cluster, from the persisted `domain` column.

    Included as EVIDENCE for the reader, never as a filter here. The gate has
    two documented blind spots - it drops a genuinely Kenyan cluster that avoids
    anchor vocabulary, and misses an operation discussing Kenya obliquely - and
    closing the second is a large part of why a reader is in the loop at all.
    A reader who can see the share can also disagree with it."""
    try:
        return con.sql(
            f"""
            WITH h AS (
                SELECT platform_post_id, domain FROM {hatespeech_source(platform)}
                QUALIFY row_number() OVER (
                    PARTITION BY platform_post_id ORDER BY scored_at DESC) = 1
            )
            SELECT m.cluster_id,
                   count(*) AS posts_classified,
                   avg(CASE WHEN h.domain = 'kenya' THEN 1.0 ELSE 0.0 END) AS kenya_share
            FROM _dos_members m
            JOIN _dos_posts p ON p.author_id = m.author_id
            JOIN h ON h.platform_post_id = p.platform_post_id
            GROUP BY 1
            """
        ).df()
    except duckdb.Error:
        log.warning("dossier: Kenya-share unavailable; leaving it out of the packet")
        return pd.DataFrame(columns=["cluster_id", "posts_classified", "kenya_share"])


def build(
    con: duckdb.DuckDBPyConnection,
    members: pd.DataFrame,
    scorecards: pd.DataFrame | None = None,
    platform: str = "x",
    cluster_ids: list | None = None,
    max_posts: int = DEFAULT_MAX_POSTS,
    max_objects: int = DEFAULT_MAX_OBJECTS,
    max_targets: int = DEFAULT_MAX_TARGETS,
) -> list[dict]:
    """One evidence packet per cluster, ready to hand to a reader.

    `cluster_ids` restricts the build - adjudication is the expensive step, so
    the caller is expected to have narrowed to corroborated or Kenya-relevant
    clusters first rather than packaging all ~150."""
    if members.empty:
        return []
    members = members[["cluster_id", "author_id"]].drop_duplicates()
    if cluster_ids is not None:
        members = members[members["cluster_id"].isin(cluster_ids)]
        if members.empty:
            return []

    _member_table(con, members)
    _posts_table(con, platform)
    _acts_table(con, platform)

    posts = representative_posts(con, max_posts)
    objects = shared_objects(con, platform, max_objects)
    targets = amplification_targets(con, platform, max_targets)
    prov = provenance(con, platform).set_index("cluster_id")
    kenya = kenya_share(con, platform).set_index("cluster_id")
    scores = (
        scorecards.set_index("cluster_id") if scorecards is not None and len(scorecards)
        else None
    )

    out = []
    for cid, grp in members.groupby("cluster_id"):
        packet: dict = {
            "cluster_id": int(cid),
            "size": int(len(grp)),
            "shared_objects": objects[objects["cluster_id"] == cid]
                .drop(columns=["cluster_id"]).to_dict("records"),
            "representative_posts": posts[posts["cluster_id"] == cid]
                .drop(columns=["cluster_id"]).to_dict("records"),
            "amplification_targets": targets[targets["cluster_id"] == cid]
                .drop(columns=["cluster_id"]).to_dict("records"),
        }
        if cid in prov.index:
            packet["provenance"] = prov.loc[cid].to_dict()
        if cid in kenya.index:
            packet["kenya"] = kenya.loc[cid].to_dict()
        if scores is not None and cid in scores.index:
            keep = [
                c for c in ("size", "n_channels", "channels", "name",
                            "inauthenticity_index", "near_dup_rate", "hate_index",
                            "self_amplification", "self_amplification_z",
                            "median_min_gap_s", "components_measured")
                if c in scores.columns
            ]
            packet["scores"] = scores.loc[cid, keep].to_dict()
        out.append(packet)
    return out

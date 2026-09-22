"""The base rate of disappearance, so a per-account deletion rate can be read.

Route 1 of `docs/plans/2026-09-15-finding-campaigns.md`. The collector re-checks
posts it holds and, since 2026-09-21, records the ones that come back empty as
`status='absent'` with an `absence_cause` (`kenya_monitor.collectors.base`).
A per-account deletion rate means nothing without the rate among everyone else,
so this module computes that rate and nothing more.

WHAT THE DENOMINATOR IS, AND IS NOT. `runner.collect_metrics` re-checks the top
5% of posts by engagement from the last 5 days, capped at ~400 per pass. So the
population here is *high-engagement recent posts*, not the corpus. A rate over
it is a rate among the posts most worth pushing and then deleting, which is the
population the concealment question is about - but it is not "the share of held
posts that vanish", and must never be quoted as one. A corpus-wide base rate
needs a random re-check arm, which does not exist.

Rows written before the status column carry no status. Their absences were
discarded at collection (`if tw is None: continue`), so they can say nothing
about disappearance and are excluded, not counted as present.

Partition membership is reported as flags, not an exclusive label: one post id
can sit in a baseline and a targeted partition at once, and forcing a single
label would hide exactly the overlap a reader needs to judge the comparison.
The control arm is the comparison that matters - it samples ordinary Kenyan
political posts by the minute they were posted in - and it will be thin in the
re-check population, because control posts are rarely the top 5% by engagement.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd

from kma.db import BASELINE_TYPES, BUCKET, CONTROL_TYPES, TARGETED_TYPES, metrics_source

ABSENT = "absent"
PRESENT = "present"

# How far before the first status-bearing re-check to look for the posts
# themselves. Re-checks cover posts up to 5 days old; the margin covers a post
# collected, then re-collected in a later partition.
POST_LOOKBACK_DAYS = 14


def _in_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


def _posts_on_dates(platform: str, first: date, last: date) -> str:
    """The posts prefix restricted to partitions in [first, last], by glob.

    Not `posts_source` plus a `dt` filter: with `union_by_name` DuckDB reads
    every file's schema before it prunes on the hive column, and the posts
    prefix is 6,248 files. Measured from the mac 2026-09-22: with the filter the
    report ran past 900s without returning; with these globs it returned in
    323s. Listing the whole prefix takes 6s, so the cost is schema reads."""
    days = (last - first).days
    globs = [
        f"'r2://{BUCKET}/posts/platform={platform}/type=*/"
        f"dt={first + timedelta(days=i)}/run=*.parquet'"
        for i in range(days + 1)
    ]
    return (
        f"read_parquet([{', '.join(globs)}], union_by_name=true, "
        "hive_partitioning=true)"
    )


def recheck_outcomes(
    con,
    platform: str = "x",
    since: str | None = None,
    posts_on: Callable[[str, date, date], str] = _posts_on_dates,
) -> pd.DataFrame:
    """One row per re-checked post: whether it was ever found absent, the cause
    at its first absence, how many times it was checked, and which partitions
    hold it.

    Absence is sticky on purpose. The collector retries before recording one,
    so a later `present` after an `absent` is a post that came back (an author
    unsuspended, a protected account opened) - reported as `returned`, never
    silently overwritten by the later observation."""
    window = f"AND dt >= DATE '{since}'" if since else ""
    # The metrics prefix is small - ~400 rows a pass - so it is read once and
    # held, rather than rescanned per post to find each first absence.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _del_rows AS
        SELECT platform_post_id, status, absence_cause, collected_at
        FROM {metrics_source(platform)}
        WHERE status IS NOT NULL {window}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _del_checks AS
        WITH fa AS (
            SELECT platform_post_id,
                   min(collected_at) FILTER (WHERE status = '{ABSENT}') AS first_absent_at
            FROM _del_rows GROUP BY 1
        )
        SELECT r.platform_post_id,
               count(*) AS n_checks,
               min(r.collected_at) AS first_checked,
               max(r.collected_at) AS last_checked,
               bool_or(r.status = '{ABSENT}') AS ever_absent,
               arg_min(r.absence_cause, r.collected_at)
                   FILTER (WHERE r.status = '{ABSENT}') AS first_cause,
               bool_or(r.status = '{PRESENT}' AND r.collected_at > fa.first_absent_at)
                   AS returned
        FROM _del_rows r JOIN fa USING (platform_post_id)
        GROUP BY r.platform_post_id
        """
    )
    first, last = con.sql(
        "SELECT min(first_checked)::DATE, max(last_checked)::DATE FROM _del_checks"
    ).fetchone()
    if first is None:
        return pd.DataFrame(
            columns=["platform_post_id", "author_id", "n_checks", "ever_absent",
                     "first_cause", "returned", "age_days_at_first_check",
                     "in_baseline", "in_targeted", "in_control"]
        )
    return con.sql(
        f"""
        WITH p AS (
            SELECT platform_post_id,
                   any_value(author_id) AS author_id,
                   min(created_at) AS created_at,
                   bool_or(type IN ({_in_list(BASELINE_TYPES)})) AS in_baseline,
                   bool_or(type IN ({_in_list(TARGETED_TYPES)})) AS in_targeted,
                   bool_or(type IN ({_in_list(CONTROL_TYPES)})) AS in_control
            FROM {posts_on(platform, first - timedelta(days=POST_LOOKBACK_DAYS), last)}
            SEMI JOIN _del_checks c USING (platform_post_id)
            GROUP BY platform_post_id
        )
        SELECT c.platform_post_id, p.author_id, c.n_checks, c.ever_absent, c.first_cause,
               coalesce(c.returned, false) AS returned,
               date_diff('hour', p.created_at, c.first_checked) / 24.0
                   AS age_days_at_first_check,
               coalesce(p.in_baseline, false) AS in_baseline,
               coalesce(p.in_targeted, false) AS in_targeted,
               coalesce(p.in_control, false) AS in_control
        FROM _del_checks c LEFT JOIN p USING (platform_post_id)
        """
    ).df()


def _rate_row(label: str, frame: pd.DataFrame) -> dict:
    n = len(frame)
    absent = int(frame["ever_absent"].sum()) if n else 0
    causes = frame.loc[frame["ever_absent"], "first_cause"].fillna("unresolved")
    return {
        "population": label,
        "posts": n,
        "absent": absent,
        "absent_share": absent / n if n else None,
        "post_deleted": int((causes == "post_deleted").sum()),
        "author_gone": int((causes == "author_gone").sum()),
        "unresolved": int((causes == "unresolved").sum()),
        "returned": int(frame["returned"].sum()) if n else 0,
    }


def base_rate(outcomes: pd.DataFrame) -> pd.DataFrame:
    """The table the concealment question needs: absence among all re-checked
    posts, then within each partition, split by cause.

    `absent_share` is None, not 0, for an empty population - an arm with no
    re-checked posts has no rate, and a zero would read as a measured one."""
    rows = [_rate_row("all re-checked", outcomes)]
    for flag, label in (
        ("in_baseline", "baseline"),
        ("in_targeted", "targeted"),
        ("in_control", "control"),
    ):
        rows.append(_rate_row(label, outcomes[outcomes[flag]]))
    unplaced = ~(outcomes["in_baseline"] | outcomes["in_targeted"] | outcomes["in_control"])
    rows.append(_rate_row("not found in posts", outcomes[unplaced]))
    return pd.DataFrame(rows)


def author_rates(outcomes: pd.DataFrame, min_checked: int = 3) -> pd.DataFrame:
    """Per-author absence among that author's re-checked posts, beside the base
    rate it has to be read against.

    `min_checked` is a floor on re-checked posts, not a significance test: an
    author with one re-checked post that vanished has an absence share of 1.0
    and tells us one fact. Below the floor the row is withheld."""
    cols = ["author_id", "checked", "absent", "absent_share"]
    known = outcomes.dropna(subset=["author_id"])
    if known.empty:
        return pd.DataFrame(columns=cols)
    out = (
        known.groupby("author_id")["ever_absent"]
        .agg(checked="size", absent="sum")
        .reset_index()
    )
    out["absent"] = out["absent"].astype(int)
    out["absent_share"] = out["absent"] / out["checked"]
    out = out[out["checked"] >= int(min_checked)]
    return out.sort_values(["absent_share", "checked"], ascending=False)[cols].reset_index(drop=True)

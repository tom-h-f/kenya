"""Shared census-selection primitives.

Its own module because two selectors must apply the SAME censused-TTL
exclusion and they drifted once already: `runner.hot_objects`
(engagement-ranked) got the filter on 2026-08-01, `hate_signal.hot_toxic_objects`
(toxicity-ranked) did not, and kept re-picking completed work on every pass
afterwards. They band on different columns - `repost_of_id` off retweet rows
versus the original's own `platform_post_id` - which is exactly why the shared
thing has to be a function taking a column, and not a copied SQL string.

`runner` does not import `hate_signal` and `hate_signal` does not import
`runner`, so neither is a valid home for this.
"""

from __future__ import annotations

import duckdb

from kenya_monitor.config import SNOWBALL_REFRESH_HOURS


def censused_expr(
    con: duckdb.DuckDBPyConnection,
    engagements_view: str | None,
    column: str,
    refresh_hours: int = SNOWBALL_REFRESH_HOURS,
) -> str:
    """A boolean SQL expression: was `column` censused inside the TTL?

    `column` MUST be table-qualified (`recent.repost_of_id`), not bare. The
    correlated subquery selects from the engagements view, which has its own
    `platform_post_id`; a bare outer column of the same name binds to the inner
    table instead, making the predicate `e.platform_post_id = e.platform_post_id`
    - true for every row, so every candidate is excluded and the census stalls
    completely. Bare `repost_of_id` happens to work only because engagements
    have no such column, which is luck, not design.

    "FALSE" when there is no engagement data to exclude against, so a fresh
    deployment selects normally instead of selecting nothing.

    R2 is the source of truth rather than the local JSON state, so this
    self-heals if that state is lost.

    EXISTS, not IN. Under three-valued logic a single NULL `platform_post_id`
    in engagements/ makes `NOT IN` evaluate to NULL for every candidate, and
    the selector silently returns zero objects - the same total stall, reached
    a different way.
    """
    if column and "." not in column:
        raise ValueError(
            f"column must be table-qualified to correlate correctly, got {column!r}"
        )
    if not engagements_view:
        return "FALSE"
    try:
        con.sql(f"SELECT 1 FROM {engagements_view} LIMIT 1").fetchall()
    except duckdb.Error:
        return "FALSE"  # no engagements yet; nothing to exclude
    return f"""EXISTS (
        SELECT 1 FROM {engagements_view} e
        WHERE e.platform_post_id = {column}
          AND e.collected_at > now() - INTERVAL {int(refresh_hours)} HOUR
    )"""


def censused_filter(
    con: duckdb.DuckDBPyConnection,
    engagements_view: str | None,
    column: str,
    refresh_hours: int = SNOWBALL_REFRESH_HOURS,
) -> str:
    """The same test as a WHERE-clause fragment (leading ` AND NOT ...`), or ""
    when there is nothing to exclude against.

    Use this where the exclusion only has to filter. Use `censused_expr` where
    the count of excluded candidates is also wanted, so selection and backlog
    come out of one query rather than two."""
    expr = censused_expr(con, engagements_view, column, refresh_hours)
    if expr == "FALSE":
        return ""
    return f" AND NOT {expr}"

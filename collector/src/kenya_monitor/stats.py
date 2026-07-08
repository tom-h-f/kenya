from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import duckdb

from kenya_monitor.storage import Storage


@dataclass(frozen=True)
class TypeCount:
    name: str
    raw_rows: int
    unique: int | None = None


@dataclass(frozen=True)
class DayCount:
    dt: str
    runs: int
    raw_posts: int
    unique_posts: int


@dataclass(frozen=True)
class CollectionStats:
    platform: str
    recent_hours: int
    daily_days: int
    generated_at: datetime
    posts: list[TypeCount] = field(default_factory=list)
    posts_raw_total: int = 0
    posts_unique_total: int = 0
    authors_raw_total: int = 0
    authors_unique_total: int = 0
    metrics_rows: int = 0
    engagements_rows: int = 0
    follow_edges: int = 0
    earliest_collected: datetime | None = None
    latest_collected: datetime | None = None
    recent_runs: int = 0
    recent_posts_raw: int = 0
    recent_posts_unique: int = 0
    recent_authors_raw: int = 0
    recent_metrics_rows: int = 0
    recent_engagements_rows: int = 0
    daily: list[DayCount] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _safe_scalar(con: duckdb.DuckDBPyConnection, sql: str, default: int | str | None = 0):
    try:
        row = con.sql(sql).fetchone()
        if row is None or row[0] is None:
            return default
        return row[0]
    except duckdb.Error:
        return default


def _safe_rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[tuple]:
    try:
        return con.sql(sql).fetchall()
    except duckdb.Error:
        return []


def _posts_latest_cte(posts: str) -> str:
    return f"""
        latest_posts AS (
            SELECT platform_post_id, type, collected_at
            FROM {posts}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY collected_at DESC
            ) = 1
        )
    """


def _authors_latest_cte(authors: str) -> str:
    return f"""
        latest_authors AS (
            SELECT platform_user_id
            FROM {authors}
            QUALIFY row_number() OVER (
                PARTITION BY platform_user_id ORDER BY collected_at DESC
            ) = 1
        )
    """


def gather_stats(
    storage: Storage,
    platform: str = "x",
    recent_hours: int = 24,
    daily_days: int = 7,
) -> CollectionStats:
    con = storage.con
    posts = storage.posts_view(platform=platform)
    authors = storage.authors_view(platform=platform)
    metrics = storage.metrics_view(platform=platform)
    engagements = storage.engagements_view(platform=platform)
    follows = storage.follows_view(platform=platform)

    missing: list[str] = []
    for label, view in (
        ("posts", posts),
        ("authors", authors),
        ("metrics", metrics),
        ("engagements", engagements),
        ("follows", follows),
    ):
        if _safe_scalar(con, f"SELECT count(*) FROM {view}", default=None) is None:
            missing.append(label)

    posts_by_type = _safe_rows(
        con,
        f"""
        SELECT type, count(*) AS raw_rows
        FROM {posts}
        GROUP BY 1
        ORDER BY raw_rows DESC
        """,
    )
    unique_by_type = {
        row[0]: row[1]
        for row in _safe_rows(
            con,
            f"""
            WITH {_posts_latest_cte(posts)}
            SELECT type, count(*) FROM latest_posts GROUP BY 1
            """,
        )
    }
    post_types = [
        TypeCount(name=typ, raw_rows=raw, unique=unique_by_type.get(typ))
        for typ, raw in posts_by_type
    ]

    posts_raw_total = int(_safe_scalar(con, f"SELECT count(*) FROM {posts}", 0))
    posts_unique_total = int(
        _safe_scalar(
            con,
            f"WITH {_posts_latest_cte(posts)} SELECT count(*) FROM latest_posts",
            0,
        )
    )
    authors_raw_total = int(_safe_scalar(con, f"SELECT count(*) FROM {authors}", 0))
    authors_unique_total = int(
        _safe_scalar(
            con,
            f"WITH {_authors_latest_cte(authors)} SELECT count(*) FROM latest_authors",
            0,
        )
    )
    metrics_rows = int(_safe_scalar(con, f"SELECT count(*) FROM {metrics}", 0))
    engagements_rows = int(_safe_scalar(con, f"SELECT count(*) FROM {engagements}", 0))
    follow_edges = int(_safe_scalar(con, f"SELECT count(*) FROM {follows}", 0))

    earliest = _safe_scalar(
        con,
        f"""
        SELECT min(collected_at) FROM (
            SELECT collected_at FROM {posts}
            UNION ALL SELECT collected_at FROM {authors}
            UNION ALL SELECT collected_at FROM {metrics}
            UNION ALL SELECT collected_at FROM {engagements}
            UNION ALL SELECT collected_at FROM {follows}
        )
        """,
        default=None,
    )
    latest = _safe_scalar(
        con,
        f"""
        SELECT max(collected_at) FROM (
            SELECT collected_at FROM {posts}
            UNION ALL SELECT collected_at FROM {authors}
            UNION ALL SELECT collected_at FROM {metrics}
            UNION ALL SELECT collected_at FROM {engagements}
            UNION ALL SELECT collected_at FROM {follows}
        )
        """,
        default=None,
    )

    recent_filter = f"collected_at > now() - INTERVAL {int(recent_hours)} HOUR"
    recent_runs = int(
        _safe_scalar(
            con,
            f"SELECT count(DISTINCT run) FROM {posts} WHERE {recent_filter}",
            0,
        )
    )
    recent_posts_raw = int(
        _safe_scalar(con, f"SELECT count(*) FROM {posts} WHERE {recent_filter}", 0)
    )
    recent_posts_unique = int(
        _safe_scalar(
            con,
            f"""
            WITH recent AS (
                SELECT platform_post_id
                FROM {posts}
                WHERE {recent_filter}
            )
            SELECT count(DISTINCT platform_post_id) FROM recent
            """,
            0,
        )
    )
    recent_authors_raw = int(
        _safe_scalar(con, f"SELECT count(*) FROM {authors} WHERE {recent_filter}", 0)
    )
    recent_metrics_rows = int(
        _safe_scalar(con, f"SELECT count(*) FROM {metrics} WHERE {recent_filter}", 0)
    )
    recent_engagements_rows = int(
        _safe_scalar(con, f"SELECT count(*) FROM {engagements} WHERE {recent_filter}", 0)
    )

    daily = [
        DayCount(dt=str(dt), runs=int(runs), raw_posts=int(raw), unique_posts=int(unique))
        for dt, runs, raw, unique in _safe_rows(
            con,
            f"""
            WITH daily_raw AS (
                SELECT dt, count(DISTINCT run) AS runs, count(*) AS raw_posts
                FROM {posts}
                WHERE dt >= (current_date - INTERVAL {int(daily_days) - 1} DAY)::VARCHAR
                GROUP BY 1
            ),
            daily_unique AS (
                SELECT dt, count(DISTINCT platform_post_id) AS unique_posts
                FROM {posts}
                WHERE dt >= (current_date - INTERVAL {int(daily_days) - 1} DAY)::VARCHAR
                GROUP BY 1
            )
            SELECT r.dt, r.runs, r.raw_posts, coalesce(u.unique_posts, 0)
            FROM daily_raw r
            LEFT JOIN daily_unique u USING (dt)
            ORDER BY r.dt DESC
            """,
        )
    ]

    return CollectionStats(
        platform=platform,
        recent_hours=recent_hours,
        daily_days=daily_days,
        generated_at=datetime.now(timezone.utc),
        posts=post_types,
        posts_raw_total=posts_raw_total,
        posts_unique_total=posts_unique_total,
        authors_raw_total=authors_raw_total,
        authors_unique_total=authors_unique_total,
        metrics_rows=metrics_rows,
        engagements_rows=engagements_rows,
        follow_edges=follow_edges,
        earliest_collected=earliest if isinstance(earliest, datetime) else None,
        latest_collected=latest if isinstance(latest, datetime) else None,
        recent_runs=recent_runs,
        recent_posts_raw=recent_posts_raw,
        recent_posts_unique=recent_posts_unique,
        recent_authors_raw=recent_authors_raw,
        recent_metrics_rows=recent_metrics_rows,
        recent_engagements_rows=recent_engagements_rows,
        daily=daily,
        missing=missing,
    )


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def format_stats(stats: CollectionStats) -> str:
    lines = [
        f"Collection stats (platform={stats.platform})",
        f"generated {_fmt_dt(stats.generated_at)}",
        "",
        "TOTAL",
        f"  posts (raw rows):        {_fmt_int(stats.posts_raw_total)}",
        f"  unique posts:            {_fmt_int(stats.posts_unique_total)}",
    ]
    for t in stats.posts:
        unique = f", {_fmt_int(t.unique)} unique" if t.unique is not None else ""
        lines.append(f"    {t.name:12} {_fmt_int(t.raw_rows)} raw{unique}")
    lines.extend(
        [
            f"  authors (raw rows):      {_fmt_int(stats.authors_raw_total)}",
            f"  unique authors:          {_fmt_int(stats.authors_unique_total)}",
            f"  metrics snapshots:       {_fmt_int(stats.metrics_rows)}",
            f"  engagements:             {_fmt_int(stats.engagements_rows)}",
            f"  follow edges:            {_fmt_int(stats.follow_edges)}",
            f"  collected between:       {_fmt_dt(stats.earliest_collected)} .. {_fmt_dt(stats.latest_collected)}",
            "",
            f"RECENT (last {stats.recent_hours}h)",
            f"  collection runs:         {_fmt_int(stats.recent_runs)}",
            f"  posts ingested:          {_fmt_int(stats.recent_posts_raw)}",
            f"  unique posts touched:    {_fmt_int(stats.recent_posts_unique)}",
            f"  authors ingested:        {_fmt_int(stats.recent_authors_raw)}",
            f"  metrics snapshots:       {_fmt_int(stats.recent_metrics_rows)}",
            f"  engagements:             {_fmt_int(stats.recent_engagements_rows)}",
            f"  latest activity:         {_fmt_dt(stats.latest_collected)}",
        ]
    )
    if stats.daily:
        lines.extend(["", f"BY DAY (last {stats.daily_days}d)"])
        lines.append(f"  {'dt':12} {'runs':>6} {'posts':>10} {'unique':>10}")
        for day in stats.daily:
            lines.append(
                f"  {day.dt:12} {day.runs:6} {day.raw_posts:10,} {day.unique_posts:10,}"
            )
    if stats.missing:
        lines.extend(["", f"unavailable prefixes: {', '.join(stats.missing)}"])
    return "\n".join(lines)

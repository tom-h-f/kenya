from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

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


def _coerce_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _safe_collected_range(
    con: duckdb.DuckDBPyConnection, view: str
) -> tuple[datetime | None, datetime | None]:
    row = _safe_rows(
        con,
        f"SELECT min(epoch_ms(collected_at)), max(epoch_ms(collected_at)) FROM {view}",
    )
    if not row:
        return None, None
    return _coerce_datetime(row[0][0]), _coerce_datetime(row[0][1])


def _merge_range(
    earliest: datetime | None,
    latest: datetime | None,
    candidate: tuple[datetime | None, datetime | None],
) -> tuple[datetime | None, datetime | None]:
    cand_earliest, cand_latest = candidate
    if cand_earliest is not None:
        earliest = cand_earliest if earliest is None else min(earliest, cand_earliest)
    if cand_latest is not None:
        latest = cand_latest if latest is None else max(latest, cand_latest)
    return earliest, latest


def _safe_rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[tuple]:
    try:
        return con.sql(sql).fetchall()
    except duckdb.Error:
        return []


def _optional_count(con: duckdb.DuckDBPyConnection, view: str, label: str) -> tuple[int, str | None]:
    try:
        return int(con.sql(f"SELECT count(*) FROM {view}").fetchone()[0]), None
    except duckdb.Error:
        return 0, label


def gather_stats(
    storage: Storage,
    platform: str = "x",
    recent_hours: int = 24,
    daily_days: int = 7,
) -> CollectionStats:
    con = storage.con
    posts = storage.posts_view(platform=platform)
    authors = storage.authors_view(platform=platform)
    recent_filter = f"collected_at > now() - INTERVAL {int(recent_hours)} HOUR"

    post_rows = _safe_rows(
        con,
        f"""
        WITH base AS (
            SELECT platform_post_id,
                   type AS target_type,
                   collected_at,
                   dt
            FROM {posts}
        ),
        latest AS (
            SELECT platform_post_id, target_type
            FROM base
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY collected_at DESC
            ) = 1
        ),
        by_type_raw AS (
            SELECT target_type, count(*) AS raw_rows FROM base GROUP BY 1
        ),
        by_type_unique AS (
            SELECT target_type, count(*) AS unique_rows FROM latest GROUP BY 1
        )
        SELECT r.target_type, r.raw_rows, coalesce(u.unique_rows, 0)
        FROM by_type_raw r
        LEFT JOIN by_type_unique u USING (target_type)
        ORDER BY r.raw_rows DESC
        """,
    )
    posts_raw_total = int(_safe_scalar(con, f"SELECT count(*) FROM {posts}", 0))
    posts_unique_total = int(
        _safe_scalar(
            con,
            f"""
            SELECT count(*) FROM (
                SELECT platform_post_id FROM {posts}
                QUALIFY row_number() OVER (
                    PARTITION BY platform_post_id ORDER BY collected_at DESC
                ) = 1
            )
            """,
            0,
        )
    )
    earliest, latest = _safe_collected_range(con, posts)
    post_types = [
        TypeCount(name=str(typ), raw_rows=int(raw), unique=int(unique))
        for typ, raw, unique in post_rows
    ]

    recent_rows = _safe_rows(
        con,
        f"""
        SELECT
            count(*),
            count(DISTINCT platform_post_id),
            count(DISTINCT date_trunc('minute', collected_at))
        FROM {posts}
        WHERE {recent_filter}
        """,
    )
    recent_posts_raw, recent_posts_unique, recent_runs = (
        (int(x) for x in recent_rows[0]) if recent_rows else (0, 0, 0)
    )

    author_rows = _safe_rows(
        con,
        f"""
        WITH base AS (SELECT platform_user_id, collected_at FROM {authors})
        SELECT
            count(*),
            count(DISTINCT platform_user_id),
            (SELECT count(*) FROM base WHERE {recent_filter})
        FROM base
        """,
    )
    authors_raw_total, authors_unique_total, recent_authors_raw = (
        (int(x) for x in author_rows[0]) if author_rows else (0, 0, 0)
    )

    daily = [
        DayCount(dt=str(dt), runs=int(runs), raw_posts=int(raw), unique_posts=int(unique))
        for dt, runs, raw, unique in _safe_rows(
            con,
            f"""
            SELECT dt,
                   count(DISTINCT date_trunc('minute', collected_at)) AS runs,
                   count(*) AS raw_posts,
                   count(DISTINCT platform_post_id) AS unique_posts
            FROM {posts}
            WHERE dt >= current_date - INTERVAL {int(daily_days) - 1} DAY
            GROUP BY 1
            ORDER BY 1 DESC
            """,
        )
    ]

    missing: list[str] = []
    metrics_rows, miss = _optional_count(con, storage.metrics_view(platform=platform), "metrics")
    if miss:
        missing.append(miss)
        recent_metrics_rows = 0
    else:
        recent_metrics_rows = int(
            _safe_scalar(
                con,
                f"SELECT count(*) FROM {storage.metrics_view(platform=platform)} WHERE {recent_filter}",
                0,
            )
        )
        earliest, latest = _merge_range(
            earliest,
            latest,
            _safe_collected_range(con, storage.metrics_view(platform=platform)),
        )

    engagements_rows, miss = _optional_count(
        con, storage.engagements_view(platform=platform), "engagements"
    )
    if miss:
        missing.append(miss)
        recent_engagements_rows = 0
    else:
        recent_engagements_rows = int(
            _safe_scalar(
                con,
                f"SELECT count(*) FROM {storage.engagements_view(platform=platform)} WHERE {recent_filter}",
                0,
            )
        )
        earliest, latest = _merge_range(
            earliest,
            latest,
            _safe_collected_range(con, storage.engagements_view(platform=platform)),
        )

    follow_edges, miss = _optional_count(con, storage.follows_view(platform=platform), "follows")
    if miss:
        missing.append(miss)

    return CollectionStats(
        platform=platform,
        recent_hours=recent_hours,
        daily_days=daily_days,
        generated_at=datetime.now(timezone.utc),
        posts=sorted(post_types, key=lambda t: t.raw_rows, reverse=True),
        posts_raw_total=posts_raw_total,
        posts_unique_total=posts_unique_total,
        authors_raw_total=authors_raw_total,
        authors_unique_total=authors_unique_total,
        metrics_rows=metrics_rows,
        engagements_rows=engagements_rows,
        follow_edges=follow_edges,
        earliest_collected=earliest,
        latest_collected=latest,
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

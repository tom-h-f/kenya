"""Modal spend against a monthly budget: where it went, the run-rate, the projection.

    uv run --with modal python -m kma.modal_spend                  # this month, $30
    uv run --with modal python -m kma.modal_spend --budget 25 --month 2026-09

Exit status: 0 within budget, 1 projected over budget, 2 no billing data (an
empty report is reported, never read as a $0 month).

The billing API reports only COMPLETE UTC days, so today's spend is invisible
until the day ends. The projection is month-to-date plus the trailing 7-day
daily rate over every day not yet reported, today included. A one-off run in
the window (a backfill) inflates the rate for a week; the per-app columns say
which app did it.

`modal` is imported only when the report is fetched, so the analysis package
and the pure core below import without it.
"""

from __future__ import annotations

import argparse
import os
import sys
from calendar import monthrange
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

DEFAULT_BUDGET = 30.0
BUDGET_ENV = "KMA_MODAL_BUDGET"
RATE_WINDOW_DAYS = 7
MAX_SPAN_DAYS = 31  # the billing API rejects a longer range

EXIT_OK, EXIT_OVER, EXIT_NO_DATA = 0, 1, 2


@dataclass(frozen=True)
class DayCost:
    app: str
    day: date
    cost: float


@dataclass(frozen=True)
class AppSpend:
    app: str
    month_to_date: float
    trailing: float
    projected: float


@dataclass(frozen=True)
class Summary:
    month_first: date
    month_next: date
    as_of: date
    budget: float
    apps: tuple[AppSpend, ...]
    month_to_date: float
    trailing: float
    daily_rate: float
    days_remaining: int
    projected: float

    @property
    def over_budget(self) -> bool:
        return self.projected > self.budget


def month_bounds(month: str | None, today: date) -> tuple[date, date]:
    """First day of the month and first day of the next, for `YYYY-MM` or today's."""
    if month is None:
        first = today.replace(day=1)
    else:
        try:
            year, mon = month.split("-")
            first = date(int(year), int(mon), 1)
        except ValueError:
            raise ValueError(f"month must be YYYY-MM, got {month!r}") from None
    if first > today:
        raise ValueError(f"month {first:%Y-%m} has not started")
    return first, first + timedelta(days=monthrange(first.year, first.month)[1])


def date_chunks(
    start: date, end: date, max_days: int = MAX_SPAN_DAYS
) -> list[tuple[date, date]]:
    """Contiguous [start, end) pieces, none longer than `max_days`."""
    chunks = []
    while start < end:
        stop = min(start + timedelta(days=max_days), end)
        chunks.append((start, stop))
        start = stop
    return chunks


def summarise(
    rows: Iterable[DayCost],
    month_first: date,
    month_next: date,
    as_of: date,
    budget: float,
    window_days: int = RATE_WINDOW_DAYS,
) -> Summary:
    """Month-to-date and trailing spend per app, and the projection to month end.

    `as_of` is the first day NOT yet reported (today, or the month end for a
    finished month). Days with no row cost nothing: Modal bills only on use.
    """
    window_first = as_of - timedelta(days=window_days)
    days_remaining = (month_next - as_of).days
    mtd: dict[str, float] = defaultdict(float)
    trailing: dict[str, float] = defaultdict(float)
    for row in rows:
        if month_first <= row.day < as_of:
            mtd[row.app] += row.cost
        if window_first <= row.day < as_of:
            trailing[row.app] += row.cost
    apps = tuple(
        sorted(
            (
                AppSpend(
                    app=app,
                    month_to_date=mtd[app],
                    trailing=trailing[app],
                    projected=mtd[app] + trailing[app] / window_days * days_remaining,
                )
                for app in set(mtd) | set(trailing)
            ),
            key=lambda a: (-a.month_to_date, -a.trailing, a.app),
        )
    )
    total_mtd = sum(a.month_to_date for a in apps)
    total_trailing = sum(a.trailing for a in apps)
    rate = total_trailing / window_days
    return Summary(
        month_first=month_first,
        month_next=month_next,
        as_of=as_of,
        budget=budget,
        apps=apps,
        month_to_date=total_mtd,
        trailing=total_trailing,
        daily_rate=rate,
        days_remaining=days_remaining,
        projected=total_mtd + rate * days_remaining,
    )


def render(summary: Summary) -> str:
    last = summary.as_of - timedelta(days=1)
    to_go = (
        f"{summary.days_remaining} to go ({summary.as_of:%m-%d} onward)"
        if summary.days_remaining
        else "month complete"
    )
    lines = [
        f"Modal spend {summary.month_first:%Y-%m}: complete UTC days "
        f"{summary.month_first:%m-%d}..{last:%m-%d}, {to_go}",
        "",
        f"{'app':<28}{'month-to-date':>14}{'last 7d':>10}{'per day':>10}{'projected':>11}",
    ]
    for a in summary.apps:
        lines.append(
            f"{a.app:<28}{_usd(a.month_to_date):>14}{_usd(a.trailing):>10}"
            f"{_usd(a.trailing / RATE_WINDOW_DAYS, 3):>10}{_usd(a.projected):>11}"
        )
    lines.append(
        f"{'total':<28}{_usd(summary.month_to_date):>14}{_usd(summary.trailing):>10}"
        f"{_usd(summary.daily_rate, 3):>10}{_usd(summary.projected):>11}"
    )
    verdict = "OVER BUDGET" if summary.over_budget else "within budget"
    lines += [
        "",
        f"budget {_usd(summary.budget)}, projected {_usd(summary.projected)} "
        f"({summary.projected / summary.budget:.0%}): {verdict}",
    ]
    return "\n".join(lines)


def _usd(value: float, places: int = 2) -> str:
    return f"${value:,.{places}f}"


def fetch_rows(start: date, end: date) -> list[DayCost]:
    """Daily cost per app for [start, end) from the Modal billing API."""
    try:
        import modal
    except ImportError:
        raise SystemExit(
            "modal is not installed: run with "
            "`uv run --with modal python -m kma.modal_spend`"
        ) from None

    billing = modal.Workspace.from_context().billing
    rows: list[DayCost] = []
    for chunk_start, chunk_end in date_chunks(start, end):
        report = billing.report(
            start=datetime.combine(chunk_start, datetime.min.time(), timezone.utc),
            end=datetime.combine(chunk_end, datetime.min.time(), timezone.utc),
            resolution="d",
        )
        rows.extend(
            DayCost(item.description, item.interval_start.date(), float(item.cost))
            for item in report
        )
    return rows


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="kma.modal_spend", description=__doc__.split("\n")[0])
    env_budget = os.getenv(BUDGET_ENV)
    try:
        default_budget = float(env_budget) if env_budget else DEFAULT_BUDGET
    except ValueError:
        parser.error(f"{BUDGET_ENV}={env_budget!r} is not a number")
    parser.add_argument(
        "--budget",
        type=float,
        default=default_budget,
        help=f"monthly budget in USD (env {BUDGET_ENV}, default {DEFAULT_BUDGET:g})",
    )
    parser.add_argument("--month", help="YYYY-MM, default the current UTC month")
    args = parser.parse_args(argv)
    if args.budget <= 0:
        parser.error("budget must be positive")
    return args


def main(
    argv: list[str] | None = None,
    today: date | None = None,
    fetch: Callable[[date, date], list[DayCost]] = fetch_rows,
) -> int:
    args = _parse_args(argv)
    today = today or datetime.now(timezone.utc).date()
    try:
        month_first, month_next = month_bounds(args.month, today)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return EXIT_NO_DATA

    as_of = min(today, month_next)
    fetch_start = min(month_first, as_of - timedelta(days=RATE_WINDOW_DAYS))
    rows = fetch(fetch_start, as_of)
    if not rows:
        print(
            f"Modal billing report returned no rows for {fetch_start}..{as_of} "
            "(complete UTC days only); nothing to project.",
            file=sys.stderr,
        )
        return EXIT_NO_DATA

    summary = summarise(rows, month_first, month_next, as_of, args.budget)
    print(render(summary))
    return EXIT_OVER if summary.over_budget else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

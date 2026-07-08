from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from twscrape import API
from twscrape.accounts_pool import AccountsPool
from twscrape.utils import parse_cookies

from kenya_monitor.config import XAccount

log = logging.getLogger("kenya_monitor")

# twscrape picks the next account with: ORDER BY <this> LIMIT 1
from kenya_monitor.config import TWS_ACCOUNT_ORDER
from kenya_monitor.pacing import install_per_account_pacing


@dataclass(frozen=True)
class SyncResult:
    added: int
    updated: int
    relogin_attempted: int
    active: int
    inactive: int


def configure_pool(pool: AccountsPool) -> None:
    """Least-recently-used rotation by default; override with TWS_ACCOUNT_ORDER."""
    pool._order_by = TWS_ACCOUNT_ORDER
    install_per_account_pacing(pool)


async def active_count(pool: AccountsPool) -> int:
    stats = await pool.stats()
    return int(stats.get("active") or 0)


async def sync_accounts(
    api: API,
    accounts: list[XAccount],
    *,
    relogin_failed: bool = False,
) -> SyncResult:
    """Add new accounts, refresh proxy/cookie changes, log in inactive ones."""
    configure_pool(api.pool)
    known = {a["username"] for a in await api.pool.accounts_info()}
    added = 0
    updated = 0

    for acc in accounts:
        if acc.username not in known:
            await api.pool.add_account(
                username=acc.username,
                password=acc.password,
                email=acc.email,
                email_password=acc.email_password,
                cookies=acc.cookies or None,
                proxy=acc.proxy or None,
            )
            added += 1
            continue

        db_acc = await api.pool.get_account(acc.username)
        if db_acc is None:
            continue

        proxy = acc.proxy or None
        cookies = parse_cookies(acc.cookies) if acc.cookies else db_acc.cookies
        changed = False
        if proxy != db_acc.proxy:
            db_acc.proxy = proxy
            changed = True
        if acc.cookies and cookies != db_acc.cookies:
            db_acc.cookies = cookies
            if "ct0" in cookies:
                db_acc.active = True
                db_acc.error_msg = None
            changed = True
        if changed:
            await api.pool.save(db_acc)
            updated += 1

    await api.pool.login_all()

    relogin_attempted = 0
    if relogin_failed:
        failed = [
            a["username"]
            for a in await api.pool.accounts_info()
            if not a["active"] and a.get("error_msg")
        ]
        relogin_attempted = len(failed)
        if failed:
            await api.pool.relogin_failed()

    health = await pool_health(api.pool)
    if added or updated or relogin_attempted:
        log.info(
            "account pool: +%d new, %d updated, %d relogin attempts; %d active / %d total",
            added,
            updated,
            relogin_attempted,
            health.active,
            health.total,
        )
    return SyncResult(
        added=added,
        updated=updated,
        relogin_attempted=relogin_attempted,
        active=health.active,
        inactive=health.inactive,
    )


@dataclass(frozen=True)
class PoolHealth:
    total: int
    active: int
    inactive: int
    locked: int


async def pool_health(pool: AccountsPool) -> PoolHealth:
    stats = await pool.stats()
    total = int(stats.get("total") or 0)
    active = int(stats.get("active") or 0)
    inactive = int(stats.get("inactive") or 0)
    locked = sum(v for k, v in stats.items() if k.startswith("locked_"))
    return PoolHealth(total=total, active=active, inactive=inactive, locked=locked)


def metrics_cap(active_accounts: int, per_account: int, floor: int) -> int:
    """Scale metrics refresh volume with pool size."""
    if active_accounts <= 0:
        return floor
    return max(floor, active_accounts * per_account)


def posts_gap_hours(active_accounts: int, lo: float, hi: float) -> tuple[float, float]:
    """Shorter gaps when more accounts can share the request load."""
    if active_accounts >= 40:
        return (lo * 0.5, hi * 0.5)
    if active_accounts >= 15:
        return (lo * 0.7, hi * 0.7)
    return (lo, hi)

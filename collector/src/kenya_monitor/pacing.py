from __future__ import annotations

import asyncio
import os
import random
import time
from collections import defaultdict

from twscrape.accounts_pool import AccountsPool

# Randomized cooldown between successive requests on the same scraping account.
DELAY_MIN = float(os.getenv("REQUEST_DELAY_MIN", "3"))
DELAY_MAX = float(os.getenv("REQUEST_DELAY_MAX", "12"))


class AccountPacer:
    """Per-account cooldown. Different accounts proceed in parallel."""

    def __init__(self, lo: float = DELAY_MIN, hi: float = DELAY_MAX) -> None:
        self._lo = lo
        self._hi = hi
        self._ready_at: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _delay(self) -> float:
        return random.uniform(self._lo, self._hi)

    async def acquire(self, username: str) -> None:
        async with self._locks[username]:
            wait = self._ready_at.get(username, 0.0) - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)

    def release(self, username: str) -> None:
        self._ready_at[username] = time.monotonic() + self._delay()

    def reset(self) -> None:
        self._ready_at.clear()


_pacer = AccountPacer()


def install_per_account_pacing(pool: AccountsPool) -> None:
    """Hook twscrape pool acquire/release so pacing is per account, not global."""
    if getattr(pool, "_km_pacing_installed", False):
        return
    pool._km_pacing_installed = True

    _get = pool.get_for_queue_or_wait
    _unlock = pool.unlock

    async def paced_get(queue: str):
        account = await _get(queue)
        if account is not None:
            await _pacer.acquire(account.username)
        return account

    async def paced_unlock(username: str, queue: str, req_count: int = 0):
        _pacer.release(username)
        return await _unlock(username, queue, req_count)

    pool.get_for_queue_or_wait = paced_get  # type: ignore[method-assign]
    pool.unlock = paced_unlock  # type: ignore[method-assign]


def reset_pacer() -> None:
    """Clear cooldown state (tests)."""
    _pacer.reset()

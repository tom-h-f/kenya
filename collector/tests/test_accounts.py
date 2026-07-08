import asyncio
import time

import pytest

from kenya_monitor.accounts import metrics_cap, posts_gap_hours
from kenya_monitor.pacing import AccountPacer


def test_metrics_cap_scales_with_pool():
    assert metrics_cap(0, 8, 200) == 200
    assert metrics_cap(10, 8, 200) == 200
    assert metrics_cap(54, 8, 200) == 432


def test_posts_gap_scales_down_with_large_pool():
    lo, hi = posts_gap_hours(50, 3.0, 5.0)
    assert lo == 1.5
    assert hi == 2.5
    lo, hi = posts_gap_hours(5, 3.0, 5.0)
    assert lo == 3.0
    assert hi == 5.0


@pytest.mark.asyncio
async def test_per_account_pacer_allows_parallel_accounts():
    pacer = AccountPacer(lo=0.2, hi=0.2)
    started = time.monotonic()
    await asyncio.gather(pacer.acquire("a"), pacer.acquire("b"))
    assert time.monotonic() - started < 0.1


@pytest.mark.asyncio
async def test_per_account_pacer_blocks_same_account():
    pacer = AccountPacer(lo=0.15, hi=0.15)
    await pacer.acquire("a")
    pacer.release("a")
    started = time.monotonic()
    await pacer.acquire("a")
    assert time.monotonic() - started >= 0.14

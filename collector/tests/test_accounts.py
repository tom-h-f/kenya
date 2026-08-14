import asyncio
import time

import pytest

from kenya_monitor.accounts import metrics_cap
from kenya_monitor.pacing import AccountPacer


def test_metrics_cap_scales_with_pool():
    assert metrics_cap(0, 8, 200) == 200
    assert metrics_cap(10, 8, 200) == 200
    assert metrics_cap(54, 8, 200) == 432


def test_the_wall_clock_pacing_model_is_gone():
    """`posts_gap_hours` and its two env knobs described an inter-pass gap that
    nothing called - cycles run back to back, throttled by per-account pacing.
    They were documented as live, so tuning them changed nothing."""
    import kenya_monitor.accounts as accounts
    import kenya_monitor.config as config

    assert not hasattr(accounts, "posts_gap_hours")
    assert not hasattr(config, "POSTS_MIN_GAP_HOURS")
    assert not hasattr(config, "POSTS_MAX_GAP_HOURS")


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

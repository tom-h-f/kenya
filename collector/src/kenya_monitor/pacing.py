from __future__ import annotations

import asyncio
import math
import os
import random

# Randomized delay between individual requests to make timing look less robotic.
DELAY_MIN = float(os.getenv("REQUEST_DELAY_MIN", "3"))
DELAY_MAX = float(os.getenv("REQUEST_DELAY_MAX", "12"))
REQUEST_DELAY_SCALE = float(os.getenv("REQUEST_DELAY_SCALE", "0"))

_pool_size = 1


def set_pool_size(active_accounts: int) -> None:
    global _pool_size
    _pool_size = max(1, active_accounts)


def _effective_bounds(lo: float, hi: float) -> tuple[float, float]:
    if REQUEST_DELAY_SCALE > 0:
        return lo * REQUEST_DELAY_SCALE, hi * REQUEST_DELAY_SCALE
    # auto: shrink delay as pool grows (floor at 1s)
    factor = max(1.0, math.sqrt(_pool_size))
    return max(1.0, lo / factor), max(1.5, hi / factor)


async def human_pause(lo: float = DELAY_MIN, hi: float = DELAY_MAX) -> None:
    """Sleep a random interval in [lo, hi] seconds, scaled for pool size."""
    eff_lo, eff_hi = _effective_bounds(lo, hi)
    await asyncio.sleep(random.uniform(eff_lo, eff_hi))

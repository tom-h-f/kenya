from __future__ import annotations

import asyncio
import os
import random

# Randomized delay between individual requests to make timing look less robotic.
DELAY_MIN = float(os.getenv("REQUEST_DELAY_MIN", "3"))
DELAY_MAX = float(os.getenv("REQUEST_DELAY_MAX", "12"))


async def human_pause(lo: float = DELAY_MIN, hi: float = DELAY_MAX) -> None:
    """Sleep a random interval in [lo, hi] seconds."""
    await asyncio.sleep(random.uniform(lo, hi))

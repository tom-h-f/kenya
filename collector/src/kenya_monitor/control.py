"""The control arm: a sample of Kenyan political discourse that is exogenous
to everything else this collector does.

WHY IT EXISTS
=============
Every partition we hold was collected because something about it looked
interesting. `search` is 34 chosen keywords, `timeline` is 30-odd curated
handles, and the targeted types are selected on detector output. `BASELINE_TYPES`
separates the sampling-biased collection from the rest and does that job, but a
rate computed over the baseline is still "the rate among posts matching our
keyword list", which is a statement about the keyword list.

So no prevalence claim is available today - `docs/analysis/2026-09-13-publishable-
statistics.md` finds every other kind of statement publishable and this one not.
This module is the denominator.

THE SAMPLING DESIGN, AND THE ONE THING IT RESTS ON
==================================================
There is no sampled stream and no `place_country` operator on this access path,
so a random sample of posts cannot be drawn directly. What can be drawn is a
random sample of TIME.

Search results are ranked, so the first K results of a query are not a random K.
That only matters while the result set is larger than one pass can take. Narrow
the window enough and it stops mattering: a query over a short window whose
results fit under the cap returns EVERY post in the frame for that window - a
census of the window, not a sample of it. `SEARCH_PRODUCT` is already `Latest`,
so results come back reverse-chronologically rather than by relevance, which is
what makes exhausting a window meaningful at all.

So: sample windows uniformly at random, census each sampled window, and report
the denominator as "frame posts in sampled windows".

Randomness enters through the window and never through the results. Nothing
about a post decides whether it is collected except the minute it was posted in,
which is the property `runner.census_discovered_handles` gets from `ORDER BY
random()` and for the same reason.

**Every window records whether it was truncated.** A window that came back at
the cap may have had more posts in it, so it is a sample rather than a census
and the census claim does not hold for it. Without that flag a later reader
cannot tell a quiet window from a truncated one, and the difference is the whole
validity of the arm. Truncated windows are recorded, kept, and flagged - not
silently dropped, because dropping them would bias the sample toward quiet
windows, which is worse than knowing.

PROVENANCE
==========
Rows land in `posts/type=control`, its own partition, in neither
`BASELINE_TYPES` nor `TARGETED_TYPES`. Not baseline: mixing it into the existing
denominator would repeat the 2026-08-06 composition break that made the raw
toxicity series unpublishable. Not targeted: it is the opposite of targeted.
A reader that unions it into either scope has destroyed the only thing it is
for, which is why it needs its own scope rather than a flag.

THE HORIZON, WHICH IS SHORTER THAN THE SEARCH HORIZON
=====================================================
X search accepts queries back to 14 days, but it does not RETURN comparable
results that far back - see `HORIZON_DAYS` for the measurement. The window
population is therefore a rolling 7 days, and coverage of the collection period
accumulates only by running continuously. A one-off pass samples the last week
and nothing else. That is the same lesson depth collection learned: this is a
standing cost, not a campaign.

Window age is recoverable from every row (`window_start` against
`collected_at`), so an analysis can condition on it rather than trust this
boundary. Do condition on it: even inside the band, age 1 returned about twice
what ages 2-7 did.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import yaml

from kenya_monitor.collectors.base import Collector, Post
from kenya_monitor.config import (
    CONTROL_FRAME_PATH,
    CONTROL_INCLUDE_RETWEETS,
    CONTROL_STATE_PATH,
    CONTROL_WINDOWS_PER_PASS,
    CONTROL_WINDOW_CAP,
    CONTROL_WINDOW_MINUTES,
    SEARCH_PRODUCT,
)
from kenya_monitor.storage import Storage

log = logging.getLogger("kenya_monitor")

TARGET_TYPE = "control"

# How far back a window can be sampled and still be a census.
#
# NOT `collectors.x.MAX_AGE_DAYS`, which is 14. Measured 2026-09-14 by running
# the frame query over the SAME 15-minute slot (12:00-12:15 UTC) at every age
# from 1 to 13 days, so time of day, query and cap are all held fixed and only
# age varies:
#
#   age 1..7 days:  72, 20, 24, 33, 23, 34, 18   (mean ~32)
#   age 8..13 days:  2,  3,  3,  5,  5,  4       (mean ~4)
#
# A ~9x cliff between day 7 and day 8. That is X's search index thinning, not
# Kenyan posting volume - a window at 18:15 Nairobi time returned 3 posts at
# age 8 and 39 at age 4. Sampling uniformly over 14 days would therefore make
# every rate a function of how old the window happened to be, which is exactly
# the confound the arm exists to remove.
#
# 7 rather than 8 because the cliff is measured between them and the
# conservative side of a measured boundary is the one inside it.
HORIZON_DAYS = 7

# The last window before now is deliberately excluded: a window that has not
# finished cannot be censused, and one that finished seconds ago may not be
# fully indexed. One window of slack rather than a guessed number of minutes.
SETTLE_WINDOWS = 1


@dataclass(frozen=True)
class Frame:
    """The pre-registered sampling frame, read as DATA from `control_frame.yaml`."""

    terms: tuple[str, ...]
    anchors: tuple[str, ...]

    @property
    def keyword(self) -> str:
        """The OR-group, with multi-word terms quoted.

        Unquoted, `county government` inside an OR-group is two terms with an
        implicit AND between them, which binds tighter than the OR - so the
        phrase silently becomes a different, narrower query than the one the
        frame file declares."""
        return " OR ".join(f'"{t}"' if " " in t else t for t in self.terms)


def load_frame(path: Path = CONTROL_FRAME_PATH, platform: str = "x") -> Frame:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    block = raw.get(platform) or {}
    return Frame(
        terms=tuple(block.get("terms") or ()),
        anchors=tuple(block.get("anchors") or ()),
    )


def window_population(
    now: datetime | None = None,
    *,
    minutes: int = CONTROL_WINDOW_MINUTES,
    horizon_days: int = HORIZON_DAYS,
) -> list[datetime]:
    """Every window start currently inside the searchable horizon.

    Aligned to the epoch rather than to `now`, so a window has the same
    identity on every pass. Without that, two passes would sample from two
    different grids and "this window has already been sampled" would be
    unanswerable.
    """
    now = now or datetime.now(timezone.utc)
    step = timedelta(minutes=minutes)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    last = epoch + (((now - epoch) // step) - SETTLE_WINDOWS) * step
    first = epoch + (((now - timedelta(days=horizon_days) - epoch) // step) + 1) * step
    out = []
    t = first
    while t <= last:
        out.append(t)
        t += step
    return out


def window_key(start: datetime) -> str:
    return start.strftime("%Y%m%dT%H%M%SZ")


def sample_windows(
    population: Sequence[datetime],
    n: int,
    *,
    seed: int,
    done: Sequence[str] = (),
) -> list[datetime]:
    """`n` unsampled windows, uniformly at random.

    Seeded on the pass, not on the process, so a pass is reproducible and the
    draw can be stated after the fact. Sampling without replacement across
    passes is what `done` is for: a window censused twice is not a second
    observation, it is the same one.
    """
    seen = set(done)
    pool = [w for w in population if window_key(w) not in seen]
    if not pool:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(pool, min(int(n), len(pool))))


def load_state(path: Path = CONTROL_STATE_PATH) -> dict[str, dict]:
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text()).get("windows") or {}


def save_state(windows: dict[str, dict], path: Path = CONTROL_STATE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            {"updated_at": datetime.now(timezone.utc).isoformat(), "windows": windows},
            indent=2,
        )
    )
    tmp.replace(path)


def _stamp(t: datetime) -> str:
    """X's datetime form for `since:` / `until:`, which its operators accept
    alongside the bare date the rest of this collector uses."""
    return t.strftime("%Y-%m-%d_%H:%M:%S_UTC")


async def collect_control(
    collector: Collector,
    storage: Storage,
    *,
    windows: int = CONTROL_WINDOWS_PER_PASS,
    minutes: int = CONTROL_WINDOW_MINUTES,
    cap: int = CONTROL_WINDOW_CAP,
    include_retweets: bool = CONTROL_INCLUDE_RETWEETS,
    seed: int | None = None,
    frame: Frame | None = None,
    state_path: Path = CONTROL_STATE_PATH,
    now: datetime | None = None,
) -> dict[str, int]:
    """One bounded pass: census `windows` randomly chosen windows of the frame."""
    frame = frame or load_frame()
    if not frame.terms:
        log.warning("control: the frame has no terms; nothing to sample")
        return {"windows": 0, "posts": 0, "truncated": 0}

    state = load_state(state_path)
    now = now or datetime.now(timezone.utc)
    seed = int(now.timestamp()) if seed is None else seed
    population = window_population(now, minutes=minutes)
    chosen = sample_windows(population, windows, seed=seed, done=list(state))
    log.info(
        "control: %d window(s) of %d min chosen at random from %d in the horizon "
        "(%d already sampled), seed %d",
        len(chosen), minutes, len(population), len(state), seed,
    )
    counts = {"windows": 0, "posts": 0, "truncated": 0, "authors": 0}
    if not chosen:
        return counts

    rows = []
    posts: list[Post] = []
    for start in chosen:
        end = start + timedelta(minutes=minutes)
        got = [
            p
            async for p in collector.search(
                frame.keyword,
                limit=cap,
                since=_stamp(start),
                until=_stamp(end),
                product=SEARCH_PRODUCT,
                include_retweets=include_retweets,
                anchors=list(frame.anchors),
            )
        ]
        truncated = len(got) >= cap
        posts.extend(got)
        counts["windows"] += 1
        counts["posts"] += len(got)
        counts["truncated"] += int(truncated)
        rows.append(
            {
                "window_start": start,
                "window_end": end,
                "window_minutes": int(minutes),
                "frame_terms": len(frame.terms),
                "frame_keyword": frame.keyword,
                "frame_anchors": " OR ".join(frame.anchors),
                "cap": int(cap),
                "include_retweets": bool(include_retweets),
                "posts": len(got),
                # The census claim holds only where this is false. A window at
                # the cap is a sample of an unknown larger set.
                "truncated": truncated,
                "seed": int(seed),
                "population": len(population),
            }
        )
        state[window_key(start)] = {
            "sampled_at": now.isoformat(),
            "posts": len(got),
            "truncated": truncated,
        }
        log.info(
            "control: %s..%s -> %d post(s)%s",
            _stamp(start), _stamp(end), len(got), " (TRUNCATED)" if truncated else "",
        )

    key = storage.write_posts(posts, target_type=TARGET_TYPE)
    if key:
        log.info("control: wrote %d post(s) -> %s", len(posts), key)
    # The window record goes down whether or not any post did: a window with no
    # frame posts in it is an observation, and dropping it would bias every rate
    # computed from the arm upward.
    run_key = storage.write_control_run(rows, platform=collector.platform)
    if run_key:
        log.info("control: wrote %d window record(s) -> %s", len(rows), run_key)
    authors = collector.collected_authors()
    akey = storage.write_authors(authors)
    counts["authors"] = len(authors)
    if akey:
        log.info("control: wrote %d author(s) -> %s", len(authors), akey)
    save_state(state, state_path)
    return counts

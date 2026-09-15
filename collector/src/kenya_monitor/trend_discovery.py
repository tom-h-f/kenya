"""Find campaign hashtags our keyword list cannot see, in the control arm.

WHY NOT X'S TREND LIST
======================
The obvious design was to pull X's own trends. Measured 2026-09-15 against the
live pool, it does not work: `trends()` returns nothing for both `trending` and
`news`, and the raw call reports

    API unknown error: 200 - GenericTimelineById - (-1) Internal server error

which is X rejecting the operation under an HTTP 200, not an empty list and not
a locale problem. No change of account fixes it. Re-test after a twscrape
upgrade; until then discovery has to come from data we collect ourselves.

WHY THE CONTROL ARM IS THE RIGHT PLACE
======================================
`kenya_monitor.control` samples 15-minute windows of Kenyan political discourse
uniformly at random and censuses each whole. Nothing about a post decides
whether it is collected except the minute it was posted in, so hashtag
frequency over it is an EXOGENOUS view of what is being pushed - and in
particular it cannot be biased by the 34 keywords in `targets.yaml`, which is
exactly the blind spot a manufactured hashtag exploits.

It already works. In the first 502 control posts, `#LindaMwananchiNairobi`
appeared 24 times from 11 authors (4.8% of the sample), and at least two
repeated tags - `#KaNairoTunamuOk`, `#NairobiNaFei` - are outside the keyword
list.

THE SELECTION RULE, FIXED IN ADVANCE
====================================
EMERGENCE: a tag whose rate of appearance across sampled windows rises sharply
against its own prior. Chosen over the two alternatives deliberately:

  - VOLUME tracks what is popular, not what is new, and the popular tags are
    the ones the keyword list already has.
  - AUTHOR CONCENTRATION (posts per author) is the most campaign-specific
    signal, and that is precisely why it must not select: selecting on it would
    assume the conclusion. It is recorded beside every candidate so it stays
    usable as EVIDENCE about a tag that emergence found.

Fixing the rule in advance is the same discipline as the pre-registered frame:
a rule tuned on outcomes turns an exogenous sample back into a keyword list.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("kenya_monitor")

# A tag has to appear in at least this many distinct sampled windows before it
# can be a candidate. One window is one moment: a tag seen once is not emerging,
# it is a post.
MIN_WINDOWS = 2

# How much the recent rate must exceed the prior rate. 3.0 rather than something
# tuned: the quantity is a ratio of per-window rates, and a tag appearing three
# times as often as it used to is the smallest move worth a request. Revisit
# against measured candidate volume, not against how many it "should" return.
EMERGENCE_FACTOR = 3.0

# A tag never seen before has an infinite ratio, so novelty needs its own floor
# rather than falling out of the arithmetic.
MIN_RECENT_WINDOWS = 2

_TAG = re.compile(r"#(\w+)")


def _parse(stamp: str) -> datetime:
    """duckdb renders TIMESTAMPTZ as `2026-09-15 12:00:00+00`, which
    `fromisoformat` accepts on 3.11+. Naive values are treated as UTC, which is
    what every timestamp in this corpus is."""
    dt = datetime.fromisoformat(str(stamp))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class Candidate:
    """One emerging tag, with the evidence that is NOT used to select it."""

    tag: str
    recent_windows: int
    prior_windows: int
    recent_rate: float
    prior_rate: float
    emergence: float
    posts: int
    authors: int
    first_seen: datetime
    last_seen: datetime

    @property
    def posts_per_author(self) -> float:
        """Recorded, never selected on. Ordinary discourse sits near 1.0;
        `#LindaMwananchiNairobi` measured 2.2 (24 posts, 11 authors)."""
        return self.posts / max(self.authors, 1)

    def as_row(self) -> dict:
        return {
            "tag": self.tag,
            "recent_windows": self.recent_windows,
            "prior_windows": self.prior_windows,
            "recent_rate": self.recent_rate,
            "prior_rate": self.prior_rate,
            "emergence": self.emergence,
            "posts": self.posts,
            "authors": self.authors,
            "posts_per_author": self.posts_per_author,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


def tags_in(text: str | None, hashtags=None) -> set[str]:
    """Tags on a post, lowercased.

    Prefers the platform's own `hashtags` field and falls back to the text,
    because the field is empty on some collection paths and a tag missed is a
    campaign missed.
    """
    out = {str(t).lower().lstrip("#") for t in (hashtags or []) if t}
    if text:
        out |= {m.lower() for m in _TAG.findall(text)}
    return {t for t in out if t}


def candidates(
    con,
    control_view: str,
    *,
    now: datetime | None = None,
    recent_days: int = 3,
    prior_days: int = 14,
    min_windows: int = MIN_WINDOWS,
    factor: float = EMERGENCE_FACTOR,
    min_recent_windows: int = MIN_RECENT_WINDOWS,
) -> list[Candidate]:
    """Tags emerging in the control arm's recent windows against their prior.

    Rates are per SAMPLED WINDOW, not per day. The arm samples a fixed number of
    windows per pass, but passes can be missed and the horizon rolls, so a
    per-day rate would move with our own sampling cadence rather than with the
    discourse. Per-window normalises that out.
    """
    now = now or datetime.now(timezone.utc)
    recent_start = now - timedelta(days=recent_days)
    prior_start = now - timedelta(days=prior_days)

    # Timestamps come back as strings deliberately. duckdb needs `pytz` to hand
    # a TIMESTAMPTZ to Python, the collector venv does not carry it, and adding
    # a dependency to a 1 GB container for a type conversion is the wrong trade.
    rows = con.sql(
        f"""
        SELECT CAST(window_start AS VARCHAR), text, hashtags, author_id,
               CAST(created_at AS VARCHAR)
        FROM ({control_view})
        WHERE created_at >= TIMESTAMPTZ '{prior_start.isoformat()}'
        """
    ).fetchall()

    recent_windows: dict[str, set] = {}
    prior_windows: dict[str, set] = {}
    posts: dict[str, int] = {}
    authors: dict[str, set] = {}
    seen: dict[str, list] = {}
    all_recent, all_prior = set(), set()

    for window_start, text, hashtags, author_id, created_at in rows:
        created_at = _parse(created_at)
        bucket = recent_windows if created_at >= recent_start else prior_windows
        (all_recent if created_at >= recent_start else all_prior).add(window_start)
        for tag in tags_in(text, hashtags):
            bucket.setdefault(tag, set()).add(window_start)
            if created_at >= recent_start:
                posts[tag] = posts.get(tag, 0) + 1
                authors.setdefault(tag, set()).add(author_id)
            stamps = seen.setdefault(tag, [created_at, created_at])
            stamps[0] = min(stamps[0], created_at)
            stamps[1] = max(stamps[1], created_at)

    n_recent, n_prior = max(len(all_recent), 1), max(len(all_prior), 1)
    out = []
    for tag, windows in recent_windows.items():
        if len(windows) < min_recent_windows:
            continue
        prior = prior_windows.get(tag, set())
        if len(windows) + len(prior) < min_windows:
            continue
        recent_rate = len(windows) / n_recent
        prior_rate = len(prior) / n_prior
        # A tag with no prior is novel, which is the strongest form of emerging;
        # inf would sort correctly but is not a number anyone can read.
        emergence = recent_rate / prior_rate if prior_rate else float("inf")
        if emergence < factor:
            continue
        out.append(
            Candidate(
                tag=tag,
                recent_windows=len(windows),
                prior_windows=len(prior),
                recent_rate=recent_rate,
                prior_rate=prior_rate,
                emergence=emergence,
                posts=posts.get(tag, 0),
                authors=len(authors.get(tag, ())),
                first_seen=seen[tag][0],
                last_seen=seen[tag][1],
            )
        )
    out.sort(key=lambda c: (c.emergence, c.posts), reverse=True)
    log.info(
        "trend discovery: %d candidate(s) from %d recent and %d prior windows",
        len(out), n_recent, n_prior,
    )
    return out


# --------------------------------------------------------------------------
# Collection


TARGET_TYPE = "trend"


def load_state(path=None) -> dict:
    """Last-run ledger, for the same reason every other arm has one: `cycle`
    resets on restart, so a cycle-modulo schedule silently never fires."""
    import json

    from kenya_monitor.config import TRENDS_STATE_PATH

    path = Path(path or TRENDS_STATE_PATH)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_state(state: dict, path=None) -> None:
    import json

    from kenya_monitor.config import TRENDS_STATE_PATH

    path = Path(path or TRENDS_STATE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


async def collect_trends(
    collector,
    storage,
    *,
    limit: int = 3,
    per_tag: int = 100,
    recent_days: int = 3,
    prior_days: int = 14,
    factor: float = EMERGENCE_FACTOR,
    now: datetime | None = None,
) -> dict[str, int]:
    """Discover emerging tags in the control arm, then census the top few.

    Rows land in `posts/type=trend`, a TARGETED partition. Selection is
    conditioned on a tag having been noticed, so these posts must stay out of
    every prevalence denominator - the control arm remains the denominator and
    this arm is what it points at.
    """
    from kenya_monitor.config import SEARCH_PRODUCT

    now = now or datetime.now(timezone.utc)
    found = candidates(
        storage.con,
        storage.control_posts_view(platform=collector.platform),
        now=now,
        recent_days=recent_days,
        prior_days=prior_days,
        factor=factor,
    )
    counts = {"candidates": len(found), "censused": 0, "posts": 0, "authors": 0}
    # Written even on an empty pass, so the cadence gate cannot spin.
    save_state({"last_run": now.isoformat(), "candidates": len(found)})
    if not found:
        return counts

    chosen = found[: max(0, int(limit))]
    collected: dict[str, int] = {}
    posts = []
    for cand in chosen:
        got = [
            p
            async for p in collector.search(
                f"#{cand.tag}", limit=per_tag, product=SEARCH_PRODUCT
            )
        ]
        collected[cand.tag] = len(got)
        posts.extend(got)
        log.info(
            "trend: #%s emergence %.1f (%d posts, %d authors, %.1f per author) -> %d collected",
            cand.tag, cand.emergence, cand.posts, cand.authors,
            cand.posts_per_author, len(got),
        )

    key = storage.write_posts(posts, target_type=TARGET_TYPE)
    if key:
        log.info("trend: wrote %d post(s) -> %s", len(posts), key)
    # Every candidate, censused or not.
    picked = {c.tag for c in chosen}
    rows = [
        {**c.as_row(), "censused": c.tag in picked, "collected_posts": collected.get(c.tag, 0)}
        for c in found
    ]
    ckey = storage.write_trend_candidates(rows, platform=collector.platform)
    if ckey:
        log.info("trend: wrote %d candidate row(s) -> %s", len(rows), ckey)
    authors = collector.collected_authors()
    if storage.write_authors(authors):
        counts["authors"] = len(authors)
    counts["censused"] = len(chosen)
    counts["posts"] = len(posts)
    return counts

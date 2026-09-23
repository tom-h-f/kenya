"""Rolling windows: a daily v2 run that is still reproducible from its id.

    from kma import window
    w = window.Window.ending(date(2026, 9, 21), days=90)   # daily-2026-09-21-90d
    manifest = window.ensure(w)                              # a bench snapshot

A window is a `kma.bench` snapshot whose object list is narrowed by collection
date, so everything that reads snapshots - `coord2_run`, the text trace, the
dossiers - reads a window unchanged, and a production run can be reproduced
from its window id alone, exactly as a hand run is from its snapshot name.

## Which objects a window pins

- `posts/` with `dt` in `(end - days, end]`. `dt` is the day a row was
  COLLECTED, not posted, so a deep timeline fetched inside the window brings
  its old posts with it - the history that makes an account rankable at all.
- `embeddings/` with `dt <= end`, not windowed at the start. An embedding is
  dated when it was computed, and a post re-collected inside the window may
  have been embedded weeks earlier; windowing both ends would drop its vector
  and silently shrink the text trace, which is the coverage effect measured in
  `2026-09-12-component-ranking` (62% coverage: 82,667 -> 50,085 text edges).

`end` defaults to yesterday, UTC. A `dt` partition is only closed once its day
is over, so a window ending today would list a different set of objects each
time it was rebuilt, and the id would stop naming one corpus.

## Why 90 days

Measured 2026-09-22 from the R2 listing: posts run from `dt=2026-07-16` to
`2026-09-22` over 56 collection days (69 calendar days, 694 MB), so a 90-day
window holds the whole corpus today. That is deliberate. v2 was validated
(6/6 reproduction, A2, the depth re-rank on `2026-09-14-deep500`) on
whole-corpus snapshots, and `coord2_run.run` documents why short windows fail:
a 3-day window fragmented into 207 components whose largest was 40 nodes. The
first production runs therefore reproduce the validated configuration, and the
window only starts to bite from mid-October, bounding the text trace's
quadratic growth from then on. Short windows for detecting events are a
separate pass (workstream D), not this one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import duckdb
import pandas as pd

from kma import bench
from kma.db import BUCKET

log = logging.getLogger(__name__)

DEFAULT_DAYS = 90
ID_PREFIX = "daily"
WINDOWED_PREFIXES = ("posts",)
UNBOUNDED_START_PREFIXES = ("embeddings",)


@dataclass(frozen=True)
class Window:
    end: date
    days: int = DEFAULT_DAYS

    def __post_init__(self) -> None:
        if self.days < 1:
            raise ValueError(f"a window needs at least one day, got {self.days}")

    @classmethod
    def ending(cls, end: date | str | None = None, days: int = DEFAULT_DAYS,
               now: datetime | None = None) -> Window:
        if end is None or end == "":
            end = default_end(now)
        elif isinstance(end, str):
            end = date.fromisoformat(end)
        return cls(end=end, days=days)

    @property
    def start(self) -> date:
        """First collection day inside the window."""
        return self.end - timedelta(days=self.days - 1)

    @property
    def id(self) -> str:
        return f"{ID_PREFIX}-{self.end:%Y-%m-%d}-{self.days}d"

    def keeps(self, prefix: str, partitions: dict[str, str]) -> bool:
        """The `bench.snapshot(keep=...)` predicate for this window."""
        if prefix not in WINDOWED_PREFIXES + UNBOUNDED_START_PREFIXES:
            return False
        dt = partitions.get("dt")
        if dt is None:
            return False
        try:
            day = date.fromisoformat(dt)
        except ValueError:
            return False
        if day > self.end:
            return False
        return prefix in UNBOUNDED_START_PREFIXES or day >= self.start


def default_end(now: datetime | None = None) -> date:
    """Yesterday in UTC: the latest day whose `dt` partition is closed."""
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).date() - timedelta(days=1)


def build(
    w: Window,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    client=None,
    bucket: str = BUCKET,
    uri=None,
    write: bool = True,
) -> pd.DataFrame:
    """Write the window's manifest, as a snapshot named by the window id."""
    return bench.snapshot(
        w.id,
        prefixes=WINDOWED_PREFIXES + UNBOUNDED_START_PREFIXES,
        con=con,
        client=client,
        bucket=bucket,
        uri=uri,
        write=write,
        keep=w.keeps,
    )


def ensure(
    w: Window,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    client=None,
    bucket: str = BUCKET,
    uri=None,
) -> pd.DataFrame:
    """The window's manifest: read back if it was written, built if not.

    Read back rather than rebuilt so a re-run of the same day - a retry after a
    failure, or a reproduction a month later - reads the objects the first run
    read, even if a backfill has since written into an old `dt` partition.
    """
    try:
        manifest = bench.load(w.id, con=con, bucket=bucket, uri=uri)
    except duckdb.Error:
        manifest = None
    if manifest is not None and not manifest.empty:
        log.info("window %s: reusing manifest (%d objects)", w.id, len(manifest))
        return manifest
    log.info("window %s: building manifest (%s to %s)", w.id, w.start, w.end)
    return build(w, con=con, client=client, bucket=bucket, uri=uri, write=True)

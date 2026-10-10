"""The v2 daily's text trace as its own stage, so it can run off the rank host.

    from kma import text_trace
    text_trace.run("", days, con=connect())          # what the Modal cron calls

The rank runs on a small CPU host, where the quadratic pair search over a full
window of vectors does not fit. This stage embeds the window's missing posts,
pins the window manifest and builds the trace by the same calls the rank
host's `--text-trace local` makes (`build` below is that code path), then
leaves the result under the key `v2_daily.text_trace_exists` reads. The rank
host finds it there and reuses it; R2 is the only coupling.

Modal-free on purpose: `modal_text_trace.py` supplies the GPU and the schedule.

## Scratch

`scratch="<prefix>"` writes the trace and its embeddings under that prefix and
holds the manifest in memory, so a measurement or an equivalence test touches
no production key. Reads still come from the live prefixes. The manifest is
also written under the prefix, with its production object paths intact, so a
second implementation can be run over exactly the objects the first read.
Embeddings written under a scratch prefix are not seen by the trace: the
manifest lists the live `embeddings/` prefix.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import duckdb
import pandas as pd

from kma import bench, coord2, embed_window, textsim_run, v2_daily, window
from kma.db import BUCKET

log = logging.getLogger(__name__)

MODES = ("modal", "local", "none")

# Values of the `text_trace` field on a `daily_runs` row.
BUILT = "built"
REUSED = "reused"
MISSING = "missing"
SKIPPED = "skipped"


def scratch_uri(prefix: str) -> Callable[[str], str]:
    root = prefix.strip("/")
    if not root:
        raise ValueError("a scratch prefix must not be empty: that is the production keyspace")
    return lambda key: f"r2://{BUCKET}/{root}/{key}"


def build(
    con: duckdb.DuckDBPyConnection,
    window_id: str,
    *,
    manifest: pd.DataFrame,
    persist: bool = True,
    uri: Callable[[str], str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """The text trace of a pinned window at the production cut: threshold,
    overlap floor and day buckets are `v2_daily`'s and `coord2`'s, not
    arguments. The cloud runner's local build and the Modal stage both call
    this, which is what makes their outputs one trace."""
    return textsim_run.build(
        con, window_id,
        threshold=v2_daily.TEXT_THRESHOLD, overlap=-1.0,
        similarity_for=coord2.gpu_cosine_pairs, time_bucket="day",
        persist=persist, manifest=manifest, uri=uri,
    )


def _manifest_exists(con: duckdb.DuckDBPyConnection, w: window.Window) -> bool:
    try:
        return not bench.load(w.id, con=con).empty
    except duckdb.Error:
        return False


def _pin(con: duckdb.DuckDBPyConnection, w: window.Window, *, embedded: int,
         uri: Callable[[str], str] | None) -> tuple[pd.DataFrame, bool]:
    """The window manifest, and whether it was rebuilt over an older one.

    A manifest lists the vectors that existed when it was written. If one was
    written before this run's embed stage, it cannot see the vectors the stage
    just added and the trace would be built over the old coverage without a
    word. Rebuilding is only done when the stage wrote something, so a covered
    window keeps reading back the manifest the first run pinned."""
    if uri is not None:
        manifest = window.build(w, con=con, write=False)
        key = f"{bench.SNAPSHOT_PREFIX}/snapshot={w.id}/manifest.parquet"
        con.register("_scratch_manifest", manifest)
        try:
            con.execute(f"COPY _scratch_manifest TO '{uri(key)}' (FORMAT parquet, COMPRESSION zstd)")
        finally:
            con.unregister("_scratch_manifest")
        return manifest, False
    if embedded and _manifest_exists(con, w):
        log.warning("window %s: manifest predates the %d vector(s) just embedded; rebuilding",
                    w.id, embedded)
        return window.build(w, con=con, write=True), True
    return window.ensure(w, con=con), False


def run(
    end: str | None,
    days: int,
    *,
    con: duckdb.DuckDBPyConnection,
    scratch: str = "",
    embed: bool = True,
    force: bool = False,
) -> dict:
    """Embed, pin, and build the window's text trace unless it exists.

    Idempotent: a window with every vector and a persisted trace costs the
    manifest read and the existence probe. `force` rebuilds the trace
    regardless, for a measurement."""
    w = window.Window.ending(end or None, days)
    uri = scratch_uri(scratch) if scratch else None
    out: dict = {"window_id": w.id, "scratch": scratch or None}

    t = time.monotonic()
    embedded = 0
    if embed:
        got = embed_window.embed_missing(con, w, uri=uri)
        embedded = got["embedded"]
        out["embed"] = got
    out["embed_s"] = round(time.monotonic() - t, 1)

    t = time.monotonic()
    manifest, rebuilt = _pin(con, w, embedded=embedded, uri=uri)
    posts = manifest[manifest["prefix"] == "posts"]
    if posts.empty:
        raise RuntimeError(f"window {w.id} holds no posts; refusing to build a trace over nothing")
    out.update(
        posts_objects=len(posts),
        manifest_rebuilt=rebuilt,
        window_s=round(time.monotonic() - t, 1),
    )

    t = time.monotonic()
    if not force and not rebuilt and v2_daily.text_trace_exists(con, w.id, uri=uri):
        out["textsim"] = REUSED
    else:
        _, info = build(con, w.id, manifest=manifest, uri=uri)
        out["textsim"] = BUILT
        out["trace"] = info
    out["textsim_s"] = round(time.monotonic() - t, 1)
    return out

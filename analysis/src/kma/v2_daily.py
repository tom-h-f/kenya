"""The daily v2 pass: window, text trace, rank, community report, persist.

    from kma import v2_daily
    v2_daily.orchestrate("", days=90, textsim=..., rank_fn=..., con=...)   # the Modal app

Modal-free on purpose: `modal_v2_daily.py` supplies where each stage runs (the
text trace on a GPU, the rank on a big CPU box) and this module supplies what
runs, so the logic is testable offline and a hand run is the same code.

## What a daily run writes

Under `coord2/platform=x/kind=daily_<table>/dt=<run day>/run=<run stamp>.parquet`,
every row carrying `run_id` and `window_id`:

- `daily_runs` - one row: window, trace sizes, counts, timings. The index a
  reader starts from.
- `daily_accounts` - the listed accounts, in listing order, with community,
  within-community score, the paper's global centrality and Kenya share.
- `daily_communities` - every community of `MIN_SIZE`+ accounts, kept or
  dropped by the relevance floor, with its trace mix and relevance coverage.
- `daily_members` - every member of those communities. Leiden ids are not
  stable across runs, so "is this community new?" has to be answered by
  membership overlap, and that needs the full membership, not the listing.

A separate kind from `kind=scores` deliberately: snapshot runs made by hand and
daily production runs answer different questions (a fixed corpus against a
moving window), and a reader globbing one must never pick up the other.

## Where windowed detection plugs in

`extra_passes` takes callables `PassContext -> {name: DataFrame}`. Each frame is
written as `kind=daily_<name>` beside the others, in the same run. Workstream D
(`feat/windowed-floor`) delivers its windowed scoring in that shape; wiring it
in is one argument in `modal_v2_daily.py`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

import duckdb
import networkx as nx
import pandas as pd

from kma import coord2, coord2_communities, coord2_run, window
from kma.db import BUCKET

log = logging.getLogger(__name__)

TEXT_THRESHOLD = 0.85
PLATFORM = "x"


@dataclass
class PassContext:
    """Everything a pass over one daily window may need, built once."""

    con: duckdb.DuckDBPyConnection
    window: window.Window
    view: str
    networks: dict[str, pd.DataFrame]
    graph: nx.Graph
    centrality: pd.Series
    run_id: str


ExtraPass = Callable[[PassContext], dict[str, pd.DataFrame]]


@dataclass
class RankResult:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    keys: dict[str, str] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


def run_stamp(now: datetime | None = None) -> datetime:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)


def daily_key(table: str, stamp: datetime, platform: str = PLATFORM) -> str:
    return (
        f"coord2/platform={platform}/kind=daily_{table}"
        f"/dt={stamp:%Y-%m-%d}/run={stamp:%Y%m%dT%H%M%SZ}.parquet"
    )


def text_trace_exists(
    con: duckdb.DuckDBPyConnection,
    snapshot: str,
    threshold: float = TEXT_THRESHOLD,
    overlap: float | None = coord2.TEXT_MIN_OVERLAP,
    uri: Callable[[str], str] | None = None,
) -> bool:
    uri = uri or (lambda key: f"r2://{BUCKET}/{key}")
    key = coord2_run.textsim_key(snapshot, threshold, overlap=overlap)
    try:
        con.sql(f"SELECT 1 FROM read_parquet('{uri(key)}') LIMIT 0")
    except duckdb.Error:
        return False
    return True


def _write(
    con: duckdb.DuckDBPyConnection, frame: pd.DataFrame, key: str, uri: Callable[[str], str]
) -> str:
    con.register("_daily_buf", frame)
    try:
        con.execute(f"COPY _daily_buf TO '{uri(key)}' (FORMAT parquet, COMPRESSION zstd)")
    finally:
        con.unregister("_daily_buf")
    log.info("wrote %s (%d rows)", key, len(frame))
    return key


def rank(
    con: duckdb.DuckDBPyConnection,
    w: window.Window,
    *,
    text_threshold: float | None = TEXT_THRESHOLD,
    text_overlap: float | None = coord2.TEXT_MIN_OVERLAP,
    min_entities: int = coord2.MIN_ENTITIES_PER_USER,
    relevance: bool = True,
    budget: int = coord2_communities.BUDGET,
    per_group: int = coord2_communities.PER_GROUP,
    min_size: int = coord2_communities.MIN_SIZE,
    min_kenya: float = coord2_communities.MIN_KENYA,
    extra_passes: Sequence[ExtraPass] = (),
    persist: bool = True,
    uri: Callable[[str], str] | None = None,
    now: datetime | None = None,
    networks: tuple[dict[str, pd.DataFrame], pd.DataFrame, str] | None = None,
) -> RankResult:
    """Rank one window and persist every table. `networks` injects prebuilt
    `(networks, sizes, view)` in place of `coord2_run.snapshot_networks`, which
    is how the offline tests run without R2."""
    uri = uri or (lambda key: f"r2://{BUCKET}/{key}")
    stamp = run_stamp(now)
    run_id = f"{stamp:%Y%m%dT%H%M%SZ}"
    timings: dict[str, float] = {}

    t = time.monotonic()
    nets, sizes, view = networks or coord2_run.snapshot_networks(
        con, w.id, text_threshold=text_threshold, text_overlap=text_overlap,
        min_entities=min_entities,
    )
    timings["networks_s"] = time.monotonic() - t

    t = time.monotonic()
    graph = coord2.fuse(nets)
    glob = coord2.centrality(graph)
    timings["fuse_centrality_s"] = time.monotonic() - t

    t = time.monotonic()
    kenya = (
        coord2_communities.model_kenya_share(con, view, [str(n) for n in graph.nodes])
        if relevance and graph.number_of_nodes() else None
    )
    timings["relevance_s"] = time.monotonic() - t

    t = time.monotonic()
    rep = coord2_communities.report(
        graph, glob, kenya=kenya, budget=budget, per_group=per_group,
        min_size=min_size, min_kenya=min_kenya,
    )
    timings["communities_s"] = time.monotonic() - t

    listed = rep.listed
    if relevance and not listed.empty:
        gate = coord2_run.attach_relevance(con, listed[["user_id", "centrality"]], view)
        listed = listed.merge(gate[["user_id", "n_posts", "kenya_share"]], on="user_id", how="left")

    tag = {"run_id": run_id, "window_id": w.id, "window_end": w.end.isoformat(),
           "window_days": w.days}
    tables = {
        "accounts": listed.assign(**tag),
        "communities": rep.communities.assign(**tag),
        "members": rep.members[["user_id", "community", "within_score", "centrality"]].assign(**tag),
    }

    ctx = PassContext(con=con, window=w, view=view, networks=nets, graph=graph,
                      centrality=glob, run_id=run_id)
    for extra in extra_passes:
        t = time.monotonic()
        for name, frame in extra(ctx).items():
            if name in tables or name == "runs":
                raise ValueError(f"extra pass table {name!r} collides with a daily table")
            tables[name] = frame.assign(**tag)
        timings[f"pass_{getattr(extra, '__name__', 'extra')}_s"] = time.monotonic() - t

    communities = rep.communities
    summary = {
        **tag,
        "computed_at": stamp,
        "accounts_in_graph": graph.number_of_nodes(),
        "edges_in_graph": graph.number_of_edges(),
        "communities": len(communities),
        "communities_kept": int(communities["kept"].sum()) if len(communities) else 0,
        "communities_unscored": (
            int((~communities["relevance_scored"]).sum()) if len(communities) else 0
        ),
        "listed_accounts": len(listed),
        "listed_communities": int(listed["community"].nunique()) if len(listed) else 0,
        "text_threshold": text_threshold,
        "text_overlap": text_overlap,
        "min_entities": min_entities,
        "min_kenya": min_kenya,
        **{f"edges_{r['trace']}": int(r["edges"]) for _, r in sizes.iterrows()},
        **{k: round(v, 1) for k, v in timings.items()},
    }
    result = RankResult(tables=tables, summary=summary)

    if persist:
        for name, frame in tables.items():
            result.keys[name] = _write(con, frame, daily_key(name, stamp), uri)
        result.keys["runs"] = _write(con, pd.DataFrame([summary]), daily_key("runs", stamp), uri)
    return result


def orchestrate(
    end: str | None,
    days: int = window.DEFAULT_DAYS,
    *,
    textsim: Callable[[str], dict],
    rank_fn: Callable[[window.Window], dict],
    con: duckdb.DuckDBPyConnection,
    ensure: Callable[[window.Window], pd.DataFrame] | None = None,
    trace_exists: Callable[[str], bool] | None = None,
) -> dict:
    """Window -> text trace (only if missing) -> rank. Each stage is a callable
    so the Modal app can put it on the right hardware and the tests can fake it.
    """
    w = window.Window.ending(end, days)
    t0 = time.monotonic()
    manifest = (ensure or (lambda win: window.ensure(win, con=con)))(w)
    posts = manifest[manifest["prefix"] == "posts"]
    out: dict = {
        "window_id": w.id,
        "posts_objects": len(posts),
        "posts_rows": int(pd.to_numeric(posts["rows"], errors="coerce").sum()),
        "window_s": round(time.monotonic() - t0, 1),
    }
    if posts.empty:
        raise RuntimeError(f"window {w.id} holds no posts; refusing to rank an empty corpus")

    exists = trace_exists or (lambda sid: text_trace_exists(con, sid))
    t = time.monotonic()
    if exists(w.id):
        out["textsim"] = "reused"
    else:
        out["textsim"] = textsim(w.id)
    out["textsim_s"] = round(time.monotonic() - t, 1)

    t = time.monotonic()
    out["rank"] = rank_fn(w)
    out["rank_s"] = round(time.monotonic() - t, 1)
    return out

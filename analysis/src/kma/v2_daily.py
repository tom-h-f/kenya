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
- `daily_edges` - every fused edge: who is linked to whom, a bitmask of the
  traces that linked them (`TRACE_BITS`), the largest trace weight and the
  number of retweeted posts both accounts retweeted. Built from the networks
  already in memory plus the trace rows the extractors returned, so it adds a
  write and no extraction.
- `daily_traces` - the (user, trace, entity) rows of the members of kept
  communities, the evidence an edge rests on. Not written when empty. Text similarity has none: its
  trace is pair-level.

A separate kind from `kind=scores` deliberately: snapshot runs made by hand and
daily production runs answer different questions (a fixed corpus against a
moving window). The CPU runner (`kma.v2_daily_run`) also writes `kind=scores`
from the listed accounts, with `snapshot` set to the window id, because the
collector's targeting reads that kind; a reader of it picks runs by
`computed_at` and can tell the two apart by `snapshot` and `window_since`.

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
import numpy as np
import pandas as pd
from scipy import sparse

from kma import canary, coord2, coord2_communities, coord2_run, window
from kma.db import BUCKET

log = logging.getLogger(__name__)

TEXT_THRESHOLD = 0.85
PLATFORM = "x"

TRACE_BITS = {name: 1 << i for i, name in enumerate(coord2.TRACES)}
"""`daily_edges.mask`: bit 1 co_retweet, 2 co_url, 4 hashtag_sequence,
8 fast_retweet, 16 text_similarity. The explorer contract's TraceMask."""

_SHARED_CHUNK = 100_000


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
    trace_sizes: pd.DataFrame = field(default_factory=pd.DataFrame)


def edges_frame(
    networks: dict[str, pd.DataFrame],
    retweets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The fused graph's edge list, one row per linked pair.

    `source` < `target` as strings. `weight_max` is the largest similarity over
    the traces that linked the pair, a cosine and not a probability.
    `shared_retweets` counts the distinct retweeted posts both accounts
    retweeted and is 0 for a pair with no co_retweet link; it is null when the
    pair has one but `retweets` (the co_retweet trace rows) was not given.
    """
    parts = []
    for name, edges in networks.items():
        if edges is None or edges.empty:
            continue
        a = edges["source"].astype(str).to_numpy()
        b = edges["target"].astype(str).to_numpy()
        swap = a > b
        one = pd.DataFrame({
            "source": np.where(swap, b, a),
            "target": np.where(swap, a, b),
            "mask": np.int16(TRACE_BITS[name]),
            "weight_max": (edges["weight"].to_numpy(dtype="float64") if "weight" in edges
                           else np.full(len(edges), np.nan)),
        })
        parts.append(one.groupby(["source", "target"], as_index=False).agg(
            mask=("mask", "first"), weight_max=("weight_max", "max")))
    if not parts:
        return pd.DataFrame({"source": pd.Series(dtype=str), "target": pd.Series(dtype=str),
                             "mask": pd.Series(dtype="int16"),
                             "weight_max": pd.Series(dtype="float64"),
                             "shared_retweets": pd.Series(dtype="Int32")})
    both = pd.concat(parts, ignore_index=True)
    fused = both.groupby(["source", "target"], as_index=False, sort=True).agg(
        mask=("mask", "sum"), weight_max=("weight_max", "max"))
    fused["mask"] = fused["mask"].astype("int16")
    fused["shared_retweets"] = shared_retweets(fused, retweets)
    return fused


def shared_retweets(edges: pd.DataFrame, retweets: pd.DataFrame | None) -> pd.Series:
    """Per edge, the number of distinct posts both ends retweeted."""
    has = (edges["mask"].to_numpy() & TRACE_BITS["co_retweet"]) > 0
    out = pd.Series(np.where(has, pd.NA, 0), index=edges.index, dtype="Int32")
    if not has.any() or retweets is None or retweets.empty:
        return out
    pairs = retweets[["user_id", "entity"]].astype(str).drop_duplicates()
    users, u_idx = np.unique(pairs["user_id"].to_numpy(), return_inverse=True)
    posts, p_idx = np.unique(pairs["entity"].to_numpy(), return_inverse=True)
    incidence = sparse.csr_matrix(
        (np.ones(len(pairs), dtype=np.int32), (u_idx, p_idx)), shape=(len(users), len(posts)))
    where = np.flatnonzero(has)
    source = np.searchsorted(users, edges["source"].to_numpy()[where])
    target = np.searchsorted(users, edges["target"].to_numpy()[where])
    known = (source < len(users)) & (target < len(users))
    known[known] = (users[source[known]] == edges["source"].to_numpy()[where][known]) & (
        users[target[known]] == edges["target"].to_numpy()[where][known])
    counts = np.zeros(len(where), dtype=np.int64)
    for start in range(0, len(where), _SHARED_CHUNK):
        sl = slice(start, start + _SHARED_CHUNK)
        ok = known[sl]
        if not ok.any():
            continue
        both = incidence[source[sl][ok]].multiply(incidence[target[sl][ok]])
        counts[sl][ok] = np.asarray(both.sum(axis=1)).ravel()
    out.iloc[where[known]] = counts[known]
    return out


def traces_frame(
    trace_rows: dict[str, pd.DataFrame], members: pd.Series | list | set,
) -> pd.DataFrame:
    """(user_id, trace, entity) rows of the given members, one per distinct row."""
    keep = {str(m) for m in members}
    parts = []
    for name, rows in trace_rows.items():
        if rows is None or rows.empty:
            continue
        sub = rows[["user_id", "entity"]].astype(str)
        sub = sub[sub["user_id"].isin(keep)].drop_duplicates()
        parts.append(sub.assign(trace=name)[["user_id", "trace", "entity"]])
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype=str) for c in ("user_id", "trace", "entity")})
    return pd.concat(parts, ignore_index=True)


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
    canary_posts: pd.DataFrame | None = None,
    manifest: pd.DataFrame | None = None,
    text_edges: pd.DataFrame | None = None,
    traces: dict[str, pd.DataFrame] | None = None,
) -> RankResult:
    """Rank one window and persist every table. `networks` injects prebuilt
    `(networks, sizes, view)` in place of `coord2_run.snapshot_networks`, which
    is how the offline tests run without R2. `manifest` and `text_edges` inject
    just the window manifest and the pair-level text trace, for a caller that
    holds them in memory, and `traces` the per-trace (user_id, entity) rows that
    `networks` was built from; without them `daily_edges.shared_retweets` is
    null for co_retweet pairs and `daily_traces` is empty."""
    uri = uri or (lambda key: f"r2://{BUCKET}/{key}")
    stamp = run_stamp(now)
    run_id = f"{stamp:%Y%m%dT%H%M%SZ}"
    timings: dict[str, float] = {}

    # Planted posts join the window before any trace is built, so the canary is
    # carried by co-action like any other account rather than by a trace patched
    # in afterwards. The persisted text trace is GPU-built and cannot contain
    # them, so their text edges are added here on the same cut.
    planted = canary_posts if canary_posts is not None and not canary_posts.empty else None
    own_traces = traces is None
    trace_rows: dict[str, pd.DataFrame] = {} if traces is None else traces
    t = time.monotonic()
    nets, sizes, view = networks or coord2_run.snapshot_networks(
        con, w.id, text_threshold=text_threshold, text_overlap=text_overlap,
        min_entities=min_entities,
        transform_view=(lambda v: canary.inject_view(con, v, planted)) if planted is not None else None,
        manifest=manifest, text_edges=text_edges, trace_sink=trace_rows,
    )
    if planted is not None:
        extra_text = canary.text_edges(planted, threshold=text_threshold or 0.85,
                                       overlap=text_overlap)
        if not extra_text.empty:
            nets["text_similarity"] = pd.concat(
                [nets.get("text_similarity", pd.DataFrame(columns=extra_text.columns)), extra_text],
                ignore_index=True,
            )
        log.info("canary: %d planted post(s), %d text edge(s)", len(planted), len(extra_text))
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

    t = time.monotonic()
    edges = edges_frame(nets, trace_rows.get("co_retweet"))
    kept_ids = (
        rep.members.loc[
            rep.members["community"].isin(
                rep.communities.loc[rep.communities["kept"], "community"]),
            "user_id",
        ]
        if len(rep.communities) else pd.Series(dtype=str)
    )
    evidence = traces_frame(trace_rows, kept_ids)
    if planted is not None:
        edges = edges[~(edges["source"].map(canary.is_canary) | edges["target"].map(canary.is_canary))]
        evidence = evidence[~evidence["user_id"].map(canary.is_canary)]
    if own_traces:
        trace_rows.clear()
    timings["edges_s"] = time.monotonic() - t

    tag = {"run_id": run_id, "window_id": w.id, "window_end": w.end.isoformat(),
           "window_days": w.days}
    tables = {
        "accounts": listed.assign(**tag),
        "communities": rep.communities.assign(**tag),
        "members": rep.members[["user_id", "community", "within_score", "centrality"]].assign(**tag),
        "edges": edges.assign(**tag),
    }
    if not evidence.empty:
        tables["traces"] = evidence.assign(**tag)

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
    result = RankResult(tables=tables, summary=summary, trace_sizes=sizes)

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
    embed: Callable[[window.Window], dict] | None = None,
) -> dict:
    """[Embed] -> window -> text trace (only if missing) -> rank. Each stage is
    a callable so the Modal app can put it on the right hardware and the tests
    can fake it.

    `embed` runs before the window is pinned, because the manifest lists the
    vectors that exist when it is built and a re-run reads that listing back:
    vectors written after it would be invisible to the trace.
    """
    w = window.Window.ending(end, days)
    t0 = time.monotonic()
    embedded = embed(w) if embed is not None else None
    embed_s = round(time.monotonic() - t0, 1)
    t0 = time.monotonic()
    manifest = (ensure or (lambda win: window.ensure(win, con=con)))(w)
    posts = manifest[manifest["prefix"] == "posts"]
    out: dict = {
        "window_id": w.id,
        **({"embed": embedded, "embed_s": embed_s} if embed is not None else {}),
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

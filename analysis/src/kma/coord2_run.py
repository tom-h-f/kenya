"""A8: run the v2 detector over the Kenya corpus and report who it surfaces.

    uv run kma-coord2-run --snapshot 2026-09-05-promotion-off --days 14

Always reads a PINNED snapshot rather than the live glob, so a run is
reproducible from its snapshot id alone and cannot silently widen as the
collector writes.

## The threshold problem, stated rather than papered over

On the benchmark the centrality cut is a percentile selected on labelled
validation data. Kenya has no labels, so no such selection is possible, and the
operating point becomes a TRIAGE BUDGET decision: how many accounts is a human
willing to look at. `--top` expresses that honestly. A fixed 1e-2 would be worse
than arbitrary - eigenvector centrality is L2-normalised across nodes, so on a
million-author graph almost nothing clears it.

Nothing here is a verdict. It is a ranked list for a reader.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from datetime import datetime, timezone

from kma import bench, coord2
from kma.db import BUCKET, connect

log = logging.getLogger(__name__)

# The bipartite traces. Text similarity is deliberately separate: it needs
# embeddings and is quadratic, so it is opt-in rather than part of the default
# pass.
BIPARTITE_TRACES = {
    "co_retweet": coord2.co_retweet_traces,
    "co_url": coord2.co_url_traces,
    "hashtag_sequence": coord2.hashtag_sequence_traces,
    "fast_retweet": coord2.fast_retweet_traces,
}


@dataclass
class RunResult:
    networks: dict[str, pd.DataFrame]
    scores: pd.DataFrame
    trace_sizes: pd.DataFrame


def build_networks(
    con: duckdb.DuckDBPyConnection,
    view: str,
    *,
    traces: dict | None = None,
    min_entities: int = coord2.MIN_ENTITIES_PER_USER,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Every bipartite trace, from corpus rows to a filtered similarity network.

    `min_entities` is the activity floor, and it is exposed here because the
    2026-09-14 depth test made it the parameter that matters most: at 2, an
    account with two posts touching viral objects has a one-hot co-action vector
    whose cosine against every other toucher is 1.0 whatever TF-IDF does, and
    deepening such accounts removed 390 of 390 from the top 500. Sweeping it is
    how the floor gets set by measurement rather than by being the smallest
    number that is not 1.
    """
    wanted = traces or BIPARTITE_TRACES
    networks: dict[str, pd.DataFrame] = {}
    sizes = []

    for name, extractor in wanted.items():
        percentile = coord2.EDGE_PERCENTILE[name]
        rows = extractor(con, view)
        if rows.empty:
            log.info("%s: no trace rows", name)
            sizes.append({"trace": name, "trace_rows": 0, "users": 0, "edges": 0})
            continue
        edges = coord2.similarity_network(
            rows, percentile=percentile, min_entities=min_entities
        )
        networks[name] = edges
        users = len(set(edges["source"]) | set(edges["target"])) if not edges.empty else 0
        sizes.append(
            {"trace": name, "trace_rows": len(rows), "users": users, "edges": len(edges)}
        )
        log.info("%s: %d trace rows -> %d edges over %d users", name, len(rows), len(edges), users)

    return networks, pd.DataFrame(sizes)


def textsim_key(
    snapshot: str, threshold: float, platform: str = "x", overlap: float | None = None,
    bucket: str | None = None,
) -> str:
    """Where a text-similarity trace for this snapshot and cut lives in R2.

    Threshold is in the key because it decides most of the ranking: at 0.85 the
    trace carries 409,084 edges, at 0.70 it would carry millions. Two runs at
    different cuts are different traces, not versions of one. The word-overlap
    floor is in it for the same reason - at 0.85 it keeps 106,061 of those
    edges - and a trace built without one keeps the original path.
    """
    floor = "" if overlap is None else f"/overlap={overlap:.2f}"
    # A bucketed trace is (pair, bucket) rather than (pair): a different object,
    # so it gets its own key rather than overwriting the one every whole-snapshot
    # consumer reads.
    timed = "" if bucket is None else f"/bucket={bucket}"
    return (
        f"coord2/platform={platform}/kind=textsim"
        f"/snapshot={snapshot}/threshold={threshold:.2f}{floor}{timed}/edges.parquet"
    )


def apply_text_floor(
    con: duckdb.DuckDBPyConnection,
    edges: pd.DataFrame,
    view: str,
    minimum: int = coord2.MIN_ENTITIES_PER_USER,
) -> pd.DataFrame:
    """Drop text-similarity edges touching users with fewer than `minimum`
    text-eligible posts.

    `coord2.MIN_ENTITIES_PER_USER` is enforced inside `similarity_network`,
    which covers only the four bipartite traces. Text similarity takes a
    different path and had no equivalent, so a single near-duplicate post could
    link a one-post account into a large clique - the same pathology the floor
    exists to stop, re-entering by another door.

    Measured on snapshot 2026-09-05-promotion-off: 81 of the top 500 accounts
    had exactly one post, 110 had two or fewer, and that group averaged 0.223
    Kenya share against 0.416 for the rest. They were both the least active and
    the least relevant accounts in the ranking.
    """
    if edges.empty or minimum <= 1:
        return edges
    counts = con.sql(
        f"""SELECT CAST(user_id AS VARCHAR) AS user_id, count(*) AS n
            FROM ({view}) WHERE NOT coalesce(is_retweet, false) AND text IS NOT NULL
            GROUP BY 1"""
    ).df()
    eligible = set(counts.loc[counts["n"] >= minimum, "user_id"])
    kept = edges[edges["source"].isin(eligible) & edges["target"].isin(eligible)]
    log.info(
        "text floor: %d of %d edges kept (%d users below %d posts)",
        len(kept), len(edges), len(counts) - len(eligible), minimum,
    )
    return kept.reset_index(drop=True)


def load_textsim(
    con: duckdb.DuckDBPyConnection, snapshot: str, threshold: float, overlap: float | None = None,
    bucket: str | None = None, since: str | None = None, until: str | None = None,
) -> pd.DataFrame:
    """Read the persisted text-similarity trace, or say exactly how to make it.

    With `bucket` this reads the timestamped trace and, given `since`/`until`,
    slices it to that window and re-aggregates to one row per pair - the shape
    every consumer expects. That is what lets a windowed run use this trace at
    all; the unbucketed object has no time axis to slice."""
    key = textsim_key(snapshot, threshold, overlap=overlap, bucket=bucket)
    try:
        edges = con.sql(f"SELECT * FROM read_parquet('r2://{BUCKET}/{key}')").df()
    except duckdb.Error as exc:
        raise RuntimeError(
            f"no text-similarity trace at {key}. It is a GPU pass, so it is built "
            f"separately and once per (snapshot, threshold, overlap):\n"
            f"    cd analysis && uv run --with modal modal run modal_textsim.py "
            f"--snapshot {snapshot} --threshold {threshold} --overlap {overlap or 0}\n"
            f"Then re-run this. Use --no-text to run without it, but note the "
            f"trace is over half the fused edges."
        ) from exc
    edges = edges.assign(source=edges["source"].astype(str), target=edges["target"].astype(str))
    if bucket is None:
        return edges
    stamps = pd.to_datetime(edges["bucket"], utc=True)
    if since:
        edges = edges[stamps >= pd.Timestamp(since, tz="UTC")]
    if until:
        edges = edges[pd.to_datetime(edges["bucket"], utc=True) < pd.Timestamp(until, tz="UTC")]
    return (
        edges.groupby(["source", "target"], as_index=False)
        .agg(weight=("weight", "mean"))
    )


def run(
    snapshot: str,
    *,
    days: int | None = None,
    top: int = 500,
    text_threshold: float | None = 0.85,
    text_overlap: float | None = coord2.TEXT_MIN_OVERLAP,
    min_entities: int = coord2.MIN_ENTITIES_PER_USER,
    since: str | None = None,
    until: str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> RunResult:
    """One v2 pass over a pinned snapshot: traces, fusion, centrality, relevance.

    `days=None` uses the whole snapshot, which is the default because the
    method needs it: a 3-day window fragmented into 207 components whose largest
    was 40 nodes, and centrality had nothing to rank.
    """
    con = con or connect()
    networks, sizes, view = snapshot_networks(
        con, snapshot, days=days, text_threshold=text_threshold, text_overlap=text_overlap,
        min_entities=min_entities, since=since, until=until,
    )
    scores = coord2.detect(networks, threshold=0.0)
    scores = scores.sort_values("centrality", ascending=False).head(top).reset_index(drop=True)
    scores = attach_relevance(con, scores, view)
    # Which community each ranked account belongs to. Global centrality alone
    # hands the top of the ranking to one co-retweet block (measured
    # 2026-09-12), which is why v2 reports by community - but every consumer of
    # `kind=scores` got the global order with no way to spread across blocks.
    # The collector's promotion cap and the leads diff both need this column.
    from kma import coord2_communities

    member = coord2_communities.communities(coord2.fuse(networks))
    scores["community"] = scores["user_id"].astype(str).map(
        {str(k): int(v) for k, v in member.items()}
    ).astype("Int64")
    return RunResult(networks=networks, scores=scores, trace_sizes=sizes)


def snapshot_networks(
    con: duckdb.DuckDBPyConnection,
    snapshot: str,
    *,
    days: int | None = None,
    text_threshold: float | None = 0.85,
    text_overlap: float | None = coord2.TEXT_MIN_OVERLAP,
    min_entities: int = coord2.MIN_ENTITIES_PER_USER,
    since: str | None = None,
    until: str | None = None,
    transform_view: Callable[[str], str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, str]:
    """Every similarity network for a pinned snapshot, plus the posts view it
    was built from - the half of `run` that callers ranking differently (the
    community report, the daily pipeline) share with it.

    `transform_view` rewrites the view after the window filters and before any
    trace is built, so what it adds is seen by every trace rather than by the
    one the caller remembered to patch. The canary uses it to union its planted
    posts in memory; nothing it adds is ever written."""
    manifest = bench.load(snapshot, con=con)
    source = bench.pinned_source(manifest, "posts")

    view = coord2.posts_view(source)
    if days:
        view = f"SELECT * FROM ({view}) WHERE created_at >= now() - INTERVAL {int(days)} DAY"
    # An explicit window, for detecting campaigns as events rather than as a
    # three-month average. `days` is relative to now and so cannot address a
    # window in the past; these can, which is what makes a per-window series
    # possible.
    if since:
        view = f"SELECT * FROM ({view}) WHERE created_at >= TIMESTAMPTZ '{since}'"
    if until:
        view = f"SELECT * FROM ({view}) WHERE created_at < TIMESTAMPTZ '{until}'"
    if transform_view is not None:
        view = transform_view(view)

    networks, sizes = build_networks(con, view, min_entities=min_entities)

    if text_threshold is not None:
        edges = apply_text_floor(con, load_textsim(con, snapshot, text_threshold, text_overlap), view)
        networks["text_similarity"] = edges
        sizes = pd.concat(
            [sizes, pd.DataFrame([{"trace": "text_similarity", "trace_rows": pd.NA,
                                   "users": len(set(edges["source"]) | set(edges["target"])),
                                   "edges": len(edges)}])],
            ignore_index=True,
        )
        log.info("text_similarity: %d edges at threshold %.2f", len(edges), text_threshold)

    if not networks:
        raise RuntimeError("no similarity networks were built; nothing to detect on")
    return networks, sizes, view


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Run the v2 detector over a pinned snapshot.")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--days", type=int, default=0, help="0 for the whole snapshot")
    ap.add_argument("--top", type=int, default=500, help="triage budget, not a verdict")
    ap.add_argument("--text-threshold", type=float, default=0.85)
    ap.add_argument("--text-overlap", type=float, default=coord2.TEXT_MIN_OVERLAP,
                    help="word-overlap floor the text trace was built with; 0 for none")
    ap.add_argument("--no-text", action="store_true", help="skip the text-similarity trace")
    ap.add_argument("--min-entities", type=int, default=coord2.MIN_ENTITIES_PER_USER,
                    help="activity floor; the 2026-09-14 depth test made this the "
                         "parameter that decides whether the ranking is real")
    ap.add_argument("--since", help="window start (ISO, UTC), for event detection")
    ap.add_argument("--until", help="window end (ISO, UTC)")
    ap.add_argument("--persist", action="store_true", help="write the run to R2")
    args = ap.parse_args()

    threshold = None if args.no_text else args.text_threshold
    overlap = args.text_overlap or None
    result = run(args.snapshot, days=args.days or None, top=args.top, text_threshold=threshold,
                 text_overlap=overlap, min_entities=args.min_entities,
                 since=args.since, until=args.until)

    print(result.trace_sizes.to_string(index=False))
    print()
    share = result.scores["kenya_share"]
    print(f"kenya_share: mean {share.mean():.3f} median {share.median():.3f} "
          f"| >=15%: {int((share >= 0.15).sum())}/{len(share)}")
    print()
    print(result.scores.head(25).to_string(index=False))

    if args.persist:
        con = connect()
        print("\npersisted:", persist(con, result.scores, snapshot=args.snapshot,
                                      trace_sizes=result.trace_sizes,
                                      text_threshold=threshold, text_overlap=overlap,
                                      min_entities=args.min_entities,
                                      since=args.since, until=args.until))


if __name__ == "__main__":
    main()


def persist(
    con: duckdb.DuckDBPyConnection,
    scores: pd.DataFrame,
    *,
    snapshot: str,
    trace_sizes: pd.DataFrame,
    text_threshold: float | None = None,
    text_overlap: float | None = None,
    min_entities: int = coord2.MIN_ENTITIES_PER_USER,
    since: str | None = None,
    until: str | None = None,
    platform: str = "x",
) -> str:
    """Write one v2 pass under its own R2 prefix.

    A SEPARATE prefix from `coordination/`, deliberately. v1 writes clusters and
    v2 writes accounts; sharing a prefix would let a reader union two different
    units of prediction and get a number that means nothing. The two methods
    surfaced near-disjoint populations (22 of 500 overlap, measured 2026-09-08),
    so the distinction is not academic.

    Every parameter that decides the output travels with it: the snapshot id,
    the per-trace edge counts, and the text-similarity threshold, which is OUR
    choice rather than the paper's and determines most of the ranking.
    """
    now = datetime.now(timezone.utc)
    buf = scores.copy()
    buf["snapshot"] = snapshot
    buf["computed_at"] = now
    buf["text_threshold"] = text_threshold
    buf["text_overlap"] = text_overlap
    buf["min_entities"] = min_entities
    # A windowed run is a different object from a whole-snapshot one and must
    # not be comparable to it by accident.
    buf["window_since"] = since
    buf["window_until"] = until
    for _, row in trace_sizes.iterrows():
        buf[f"edges_{row['trace']}"] = int(row["edges"])

    key = (
        f"coord2/platform={platform}/kind=scores"
        f"/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.parquet"
    )
    con.register("_coord2_buf", buf)
    try:
        con.execute(
            f"COPY _coord2_buf TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)"
        )
    finally:
        con.unregister("_coord2_buf")
    log.info("wrote %s (%d accounts)", key, len(buf))
    return key


def attach_relevance(
    con: duckdb.DuckDBPyConnection, scores: pd.DataFrame, view: str, use_model: bool = True
) -> pd.DataFrame:
    """Kenya share per surfaced account, from `kma.relevance`.

    `use_model=False` puts it back on `measure.domain_bucket` alone, which is
    what every figure recorded before 2026-09-13 was computed with.

    Detection says accounts act together; it cannot say whether they act
    together about Kenya. v1's strongest evidence tier was only 8.3%
    Kenya-referencing, which is what discredited it, so no v2 output should be
    read without this column beside it.
    """
    from kma import measure, relevance

    ids = scores["user_id"].astype(str).tolist()
    con.register("_ids", pd.DataFrame({"user_id": ids}))
    posts = con.sql(
        f"SELECT p.post_id, p.user_id, p.text FROM ({view}) p "
        "JOIN _ids i ON CAST(p.user_id AS VARCHAR) = i.user_id"
    ).df()
    con.unregister("_ids")
    if posts.empty:
        return scores.assign(kenya_share=np.nan, n_posts=0)

    # The learned gate where a post has been scored, the keyword one where it
    # has not: recall 0.655 -> 1.000 on the human labels, and every Kenya share
    # in v2's output rests on this column.
    posts["bucket"] = (
        relevance.buckets(con, posts) if use_model
        else posts["text"].map(measure.domain_bucket)
    )
    agg = posts.groupby("user_id").agg(
        n_posts=("bucket", "size"),
        kenya_share=("bucket", lambda b: float((b == "kenya").mean())),
    )
    agg.index = agg.index.astype(str)
    return scores.assign(
        n_posts=scores["user_id"].astype(str).map(agg["n_posts"]).fillna(0).astype(int),
        kenya_share=scores["user_id"].astype(str).map(agg["kenya_share"]),
    )

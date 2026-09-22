"""Build the text-similarity trace for one snapshot and persist it to R2.

The body of `modal_textsim.py`, moved here so the daily pipeline
(`modal_v2_daily.py`) builds the trace by the same code path as a hand run
rather than a copy of it. Where the GPU comes from is the caller's business:
pass `coord2.gpu_cosine_pairs(cut)`'s factory as `similarity_for`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import duckdb
import numpy as np
import pandas as pd

from kma import bench, coord2
from kma.coord2_run import textsim_key
from kma.db import BUCKET

log = logging.getLogger(__name__)


def build(
    con: duckdb.DuckDBPyConnection,
    snapshot: str,
    *,
    limit: int = 0,
    percentile: float = 96.0,
    chunk: int = 1024,
    threshold: float = 0.0,
    overlap: float = -1.0,
    similarity_for: Callable[[float], coord2.PairSimilarity] | None = None,
    persist: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Text-eligible posts, their latest embeddings, and the thresholded
    user-pair network. Returns the edges and a summary; writes the edges to
    `coord2_run.textsim_key` unless `persist=False`.

    `overlap < 0` means `coord2.TEXT_MIN_OVERLAP`, `0` means no floor.
    """
    manifest = bench.load(snapshot, con=con)
    posts = bench.pinned_source(manifest, "posts")
    embeddings = bench.pinned_source(manifest, "embeddings")

    view = coord2.posts_view(posts)
    rows = coord2.text_rows(con, view)
    if limit:
        rows = rows.head(limit)
    log.info("text-eligible rows: %d", len(rows))

    con.register("_rows", rows[["post_id"]])
    try:
        vecs = con.sql(
            f"""SELECT e.platform_post_id AS post_id, e.embedding
                FROM {embeddings} e JOIN _rows r ON r.post_id = e.platform_post_id
                QUALIFY row_number() OVER (PARTITION BY e.platform_post_id ORDER BY e.embedded_at DESC) = 1"""
        ).df()
    finally:
        con.unregister("_rows")
    log.info("rows with an embedding: %d", len(vecs))

    joined = rows.merge(vecs, on="post_id", how="inner").reset_index(drop=True)
    coverage = len(joined) / max(len(rows), 1)
    if coverage < 0.5:
        # Measured 14% on the first daily window: the trace is then a sample of
        # the corpus, not the corpus, and a reader of its edges needs to know.
        log.warning("only %.0f%% of text-eligible posts have an embedding", coverage * 100)
    if joined.empty:
        raise RuntimeError(f"snapshot {snapshot!r}: no text-eligible post has an embedding")
    matrix = np.vstack(joined["embedding"].to_numpy())
    log.info("matrix %s", matrix.shape)

    # Estimate the cut on a sample with the exact CPU path, THEN push it into
    # the GPU pass. Estimating with a thresholded sampler would bias it.
    window = float(coord2.TEXT_WINDOW_DAYS) * 86400.0
    times = coord2.epoch_seconds(joined["created_at"])
    if threshold > 0:
        # The percentile reading does not survive contact with a real corpus: a
        # 96th percentile over ALL in-window pairs makes 4% of every pair an
        # edge. Measured 2026-09-07 on 16,478 posts, it resolved to 0.5198 and
        # produced 3,116,957 edges. An absolute cut is what the reference
        # implementation ships (`--tweet_sim_threshold`, default 0.7).
        cut = threshold
    else:
        cut = coord2.pair_similarity_percentile(
            matrix.astype("float64"), times, percentile, window_seconds=window, max_rows=4000
        )
    log.info("threshold = %.4f", cut)

    floor = coord2.TEXT_MIN_OVERLAP if overlap < 0 else overlap
    extra = {"similarity": similarity_for(float(cut))} if similarity_for else {}
    edges = coord2.text_similarity_network(
        joined[["user_id", "created_at", "clean"]],
        matrix,
        threshold=float(cut),
        chunk=chunk,
        min_overlap=floor or None,
        **extra,
    )
    log.info("edges: %d (word-overlap floor %s)", len(edges), floor or "off")

    r2_key = textsim_key(snapshot, float(cut), overlap=floor or None)
    if persist:
        con.register("_ts", edges)
        try:
            con.execute(f"COPY _ts TO 'r2://{BUCKET}/{r2_key}' (FORMAT parquet, COMPRESSION zstd)")
        finally:
            con.unregister("_ts")
        log.info("wrote r2://%s/%s", BUCKET, r2_key)

    users = len(set(edges["source"]) | set(edges["target"])) if len(edges) else 0
    return edges, {
        "snapshot": snapshot,
        "eligible": len(rows),
        "rows": len(joined),
        "embedding_coverage": round(coverage, 4),
        "threshold": float(cut),
        "overlap": floor,
        "edges": len(edges),
        "users": users,
        "r2_key": r2_key,
    }

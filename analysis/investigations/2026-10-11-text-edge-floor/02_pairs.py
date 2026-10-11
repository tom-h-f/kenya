"""Post-level text pairs for one window, at a chosen embedding coverage.

    cd analysis && uv run python investigations/2026-10-10-text-trace-value/02_pairs.py \
        --embeddings manifest|current --tag <name> [--window daily-2026-10-09-30d]

The production trace is user-level: one row per (account pair, day), with the
post pairs thrown away. Q3 needs the post pairs, so this repeats
`textsim_run.build` from the exported rows, keeping them. The method is the
production one: every cross-author pair of text-eligible posts at cosine
>= 0.85 (`v2_daily.TEXT_THRESHOLD`) whose cleaned texts share >= 0.10 char
4-grams (`coord2.TEXT_MIN_OVERLAP`), then the user-level edge is the mean
similarity of the qualifying post pairs per (pair, day of the LATER post).

`--embeddings manifest` reads the vectors the window manifest pins (`dt <= end`),
which is what production saw and is the reproduction gate against the persisted
trace. `current` reads every embeddings object on R2 now, which carries the
backfill. Output: out/<window>/pairs_<tag>/{pairs,edges,summary}.

Read-only on R2.
"""

import argparse
import json
import logging
import resource
import sys
import time

import numpy as np
import pandas as pd
import torch

from kma import bench, coord2
from kma.db import BUCKET, connect

from common import WINDOW, wdir

log = logging.getLogger("pairs")
CUT = 0.85
FLOOR = coord2.TEXT_MIN_OVERLAP


def embedding_paths(window: str, which: str, manifest: pd.DataFrame) -> list[str]:
    if which == "manifest":
        return sorted(manifest.loc[manifest["prefix"] == "embeddings", "path"].tolist())
    client = bench._r2_client()
    keys = [o["key"] for o in bench.list_objects(client, BUCKET, "embeddings")
            if o["key"].endswith(".parquet")]
    return sorted(f"r2://{BUCKET}/{k}" for k in keys)


def load_vectors(con, paths: list[str], wanted: set[str], step: int = 40) -> tuple[list[str], np.ndarray]:
    """Latest embedding per wanted post id, float32, a few objects at a time."""
    best: dict[str, tuple[object, np.ndarray]] = {}
    t = time.monotonic()
    for start in range(0, len(paths), step):
        chunk = paths[start : start + step]
        listed = ", ".join(f"'{p}'" for p in chunk)
        df = con.sql(
            f"SELECT platform_post_id AS post_id, embedded_at, embedding "
            f"FROM read_parquet([{listed}], union_by_name=true)"
        ).df()
        df = df[df["post_id"].isin(wanted)]
        for pid, at, vec in zip(df["post_id"], df["embedded_at"], df["embedding"]):
            prev = best.get(pid)
            if prev is None or at > prev[0]:
                best[pid] = (at, np.asarray(vec, dtype=np.float32))
        log.info("embeddings: %d/%d objects, %d wanted vectors (%.0fs)",
                 min(start + step, len(paths)), len(paths), len(best), time.monotonic() - t)
    ids = list(best)
    return ids, np.vstack([best[i][1] for i in ids]) if ids else np.zeros((0, 768), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", default=WINDOW)
    ap.add_argument("--embeddings", choices=["manifest", "current"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    started = time.monotonic()
    base = wdir(args.window)
    out = base / f"pairs_{args.tag}"
    out.mkdir(exist_ok=True)
    rows = pd.read_parquet(base / "rows.parquet")
    manifest = pd.read_parquet(base / "manifest.parquet")
    con = connect()
    con.execute("SET enable_progress_bar = false")

    cache = out / "vectors.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        ids, matrix = list(z["ids"]), z["matrix"]
    else:
        paths = embedding_paths(args.window, args.embeddings, manifest)
        ids, matrix = load_vectors(con, paths, set(rows["post_id"]))
        np.savez(cache, ids=np.array(ids, dtype=object), matrix=matrix)
    vec = pd.DataFrame({"post_id": ids, "k": np.arange(len(ids))})
    joined = rows.merge(vec, on="post_id", how="inner").reset_index(drop=True)
    matrix = matrix[joined["k"].to_numpy()]
    coverage = len(joined) / len(rows)
    log.info("eligible %d, embedded %d (%.1f%%)", len(rows), len(joined), coverage * 100)

    order = np.argsort(joined["created_at"].to_numpy(), kind="stable")
    joined = joined.iloc[order].reset_index(drop=True)
    matrix = matrix[order]
    times = coord2.epoch_seconds(joined["created_at"])
    users = joined["user_id"].astype(str).to_numpy()
    clean = joined["clean"].fillna("").to_numpy()
    grams: dict[int, frozenset] = {}

    def shared(a: int, b: int) -> float:
        x = grams.setdefault(a, coord2._grams(clean[a]) if clean[a] else frozenset())
        y = grams.setdefault(b, coord2._grams(clean[b]) if clean[b] else frozenset())
        return len(x & y) / len(x | y) if x and y else 0.0

    search = coord2.gpu_cosine_pairs(CUT, device=args.device)
    window_s = float(coord2.TEXT_WINDOW_DAYS) * 86400.0
    parts, raw_pairs = [], 0
    t = time.monotonic()
    for n, (i, j, sim) in enumerate(search(matrix, times, window_seconds=window_s, chunk=1024)):
        keep = users[i] != users[j]
        i, j, sim = i[keep], j[keep], sim[keep]
        raw_pairs += len(i)
        ov = np.fromiter((shared(int(a), int(b)) for a, b in zip(i, j)), dtype=float, count=len(i))
        parts.append(pd.DataFrame({"i": i, "j": j, "sim": sim, "overlap": ov}))
        if n % 50 == 0:
            log.info("block %d, raw cross-author pairs %d (%.0fs)", n, raw_pairs, time.monotonic() - t)
    pairs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        {"i": [], "j": [], "sim": [], "overlap": []})
    search_s = time.monotonic() - t

    pairs = pairs.assign(
        post_a=joined["post_id"].to_numpy()[pairs["i"]], post_b=joined["post_id"].to_numpy()[pairs["j"]],
        user_a=users[pairs["i"]], user_b=users[pairs["j"]],
        t_a=joined["created_at"].to_numpy()[pairs["i"]], t_b=joined["created_at"].to_numpy()[pairs["j"]],
    ).drop(columns=["i", "j"])
    kept = pairs[pairs["overlap"] >= FLOOR].copy()
    lo = np.where(kept["user_a"] <= kept["user_b"], kept["user_a"], kept["user_b"])
    hi = np.where(kept["user_a"] <= kept["user_b"], kept["user_b"], kept["user_a"])
    later = np.maximum(kept["t_a"].to_numpy(), kept["t_b"].to_numpy())
    kept = kept.assign(source=lo, target=hi, bucket=pd.to_datetime(later, utc=True).floor("D"))
    edges = (kept.groupby(["source", "target", "bucket"], as_index=False)
             .agg(weight=("sim", "mean")))

    pairs.to_parquet(out / "pairs_all.parquet")
    kept.to_parquet(out / "pairs.parquet")
    edges.to_parquet(out / "edges_bucket.parquet")
    plain = edges.groupby(["source", "target"], as_index=False).agg(weight=("weight", "mean"))
    plain.to_parquet(out / "edges.parquet")
    summary = {
        "window": args.window, "embeddings": args.embeddings, "eligible": len(rows),
        "embedded": len(joined), "coverage": round(coverage, 4),
        "cross_author_pairs_at_cut": int(len(pairs)), "pairs_after_overlap_floor": int(len(kept)),
        "bucket_edges": int(len(edges)), "user_edges": int(len(plain)),
        "users": int(len(set(edges["source"]) | set(edges["target"]))),
        "pair_search_s": round(search_s, 1), "device": args.device,
        "torch": torch.__version__,
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                             / (1024**3 if sys.platform == "darwin" else 1024**2), 2),
        "wall_s": round(time.monotonic() - started, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("%s", json.dumps(summary))


if __name__ == "__main__":
    main()

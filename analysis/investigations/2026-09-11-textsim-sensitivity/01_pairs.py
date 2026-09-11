"""Text-similarity threshold sweep on GPU: one exhaustive pass, exact edges per threshold.

    PYTHONPATH=<repo>/analysis/src python 01_pairs.py out/<export-dir> \\
        --min-threshold 0.70 --thresholds 0.70 0.75 0.80 0.85 0.90 0.95 --verify 0.85

`coord2.text_similarity_network` links two users when any pair of their posts
clears the cut, and weights the edge by the mean similarity over the pairs that
did. So the edges at a higher cut are NOT a filter of the edges at a lower one.

This records every post pair at or above the lowest cut once, then replays the
recorded pairs through the unchanged function for each threshold - the
aggregation stays the project's own. `--verify T` recomputes threshold T
directly on the GPU and asserts the replay produced the identical edge set.

Pairs go to disk block by block (`pairs/*.npz`), since at a low cut there can be
more of them than fit comfortably in memory.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kma import coord2


def capturing(inner, shard_dir: Path):
    shard_dir.mkdir(parents=True, exist_ok=True)
    for old in shard_dir.glob("*.npz"):
        old.unlink()
    stats = {"shards": 0, "pairs": 0}

    def pairs(vectors, times=None, *, window_seconds=None, chunk=512):
        for i, j, sim in inner(vectors, times, window_seconds=window_seconds, chunk=chunk):
            np.savez(
                shard_dir / f"{stats['shards']:07d}.npz",
                i=i.astype(np.int32), j=j.astype(np.int32), sim=sim.astype(np.float32),
            )
            stats["shards"] += 1
            stats["pairs"] += len(i)
            yield i, j, sim

    return pairs, stats


def replaying(shard_dir: Path, threshold: float):
    def pairs(vectors, times=None, *, window_seconds=None, chunk=512):
        for path in sorted(shard_dir.glob("*.npz")):
            with np.load(path) as shard:
                keep = shard["sim"] >= threshold
                if keep.any():
                    yield shard["i"][keep], shard["j"][keep], shard["sim"][keep]

    return pairs


def network(rows, matrix, threshold, similarity, chunk):
    return coord2.text_similarity_network(
        rows[["user_id", "created_at"]], matrix, threshold=threshold, similarity=similarity, chunk=chunk
    )


def canonical(edges: pd.DataFrame) -> pd.DataFrame:
    return edges.sort_values(["source", "target"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--min-threshold", type=float, default=0.70)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    ap.add_argument("--verify", type=float, default=None, help="recompute this threshold directly and compare")
    ap.add_argument("--chunk", type=int, default=256, help="rows per GPU block; the memory knob")
    args = ap.parse_args()

    if min(args.thresholds) < args.min_threshold:
        raise SystemExit("every threshold must be at or above --min-threshold")

    rows = pd.read_parquet(args.export_dir / "rows.parquet")
    matrix = np.load(args.export_dir / "matrix.npy")
    out = args.export_dir / f"sweep_min{args.min_threshold:.2f}"
    shards = out / "pairs"
    print(f"rows {len(rows):,}  matrix {matrix.shape}  chunk {args.chunk}  device {torch.cuda.get_device_name(0)}")

    torch.cuda.reset_peak_memory_stats()
    capture, stats = capturing(coord2.gpu_cosine_pairs(args.min_threshold), shards)
    start = time.perf_counter()
    lowest = network(rows, matrix, args.min_threshold, capture, args.chunk)
    elapsed = time.perf_counter() - start
    print(
        f"GPU pass at {args.min_threshold:.2f}: {stats['pairs']:,} post pairs in {stats['shards']:,} shards, "
        f"{elapsed:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
    )

    edges = {}
    for threshold in sorted(set(args.thresholds)):
        start = time.perf_counter()
        edges[threshold] = lowest if threshold == args.min_threshold else network(
            rows, matrix, threshold, replaying(shards, threshold), args.chunk
        )
        path = out / f"edges_t{threshold:.2f}.parquet"
        edges[threshold].to_parquet(path)
        users = len(set(edges[threshold]["source"]) | set(edges[threshold]["target"]))
        print(f"  t={threshold:.2f}: {len(edges[threshold]):,} edges over {users:,} users "
              f"({time.perf_counter() - start:.0f}s) -> {path.name}")

    if args.verify is not None:
        start = time.perf_counter()
        direct = canonical(network(rows, matrix, args.verify, coord2.gpu_cosine_pairs(args.verify), args.chunk))
        replayed = canonical(edges[args.verify])
        same_pairs = direct[["source", "target"]].equals(replayed[["source", "target"]])
        same_weight = same_pairs and np.allclose(direct["weight"], replayed["weight"], rtol=1e-6, atol=0)
        verdict = "IDENTICAL" if same_pairs and same_weight else "MISMATCH"
        print(f"verify t={args.verify:.2f}: direct {len(direct):,} vs replayed {len(replayed):,} edges -> "
              f"{verdict} ({time.perf_counter() - start:.0f}s)")
        if verdict != "IDENTICAL":
            raise SystemExit(1)


if __name__ == "__main__":
    main()

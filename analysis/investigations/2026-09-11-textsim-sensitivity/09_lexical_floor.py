"""The text trace with a word-overlap floor: cosine >= cut AND char-4gram Jaccard >= floor.

    PYTHONPATH=<repo>/analysis/src python 09_lexical_floor.py out/<export-dir> \\
        --shards out/<export-dir>/sweep_min0.75/pairs --cut 0.85 --floor 0.10 \\
        [--texts texts_recleaned.parquet --out-name production_check]

`06_pair_audit.py` found that most post pairs the 0.85 cut admits share almost
no words, and `07_encoder_check.py` that word overlap separates the reader's
same_message pairs from the rest far better than any encoder's cosine does:
on 300 labelled pairs a char-4gram Jaccard floor of 0.10 keeps 92% of matches
and admits none of the unrelated pairs. So this keeps the encoder as the
candidate generator and adds the floor as a second test on each pair.

Replays the post pairs `01_pairs.py` recorded, drops those under either test,
and hands the rest to the unchanged `coord2.text_similarity_network`, so the
aggregation into user edges stays the project's own. Writes
`<out-name>/edges_t<cut>.parquet`, which `02_score.py` reads.

`--texts` supplies re-cleaned text, row-aligned with `rows.parquet`. With an
`eligible` column it reproduces the production trace after `coord2.clean_text`
started stripping mentions: pairs touching a post that no longer makes
`MIN_TEXT_WORDS` are dropped. Rows are never removed, because the recorded pair
indices are positions in the time-sorted order of ALL rows; and dropping a
post changes nothing about the pairs between the others, so this is exactly
the pass production runs over the smaller row set.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kma import coord2


def grams(text: str, n: int = 4) -> frozenset:
    return frozenset(hash(text[k : k + n]) for k in range(max(len(text) - n + 1, 1)))


def floored(shard_dir: Path, cut: float, floor: float, clean: np.ndarray, eligible: np.ndarray, stats: dict):
    cache: dict[int, frozenset] = {}

    def gram_set(row: int) -> frozenset:
        if row not in cache:
            cache[row] = grams(clean[row]) if clean[row] else frozenset()
        return cache[row]

    def jaccard(a: int, b: int) -> float:
        x, y = gram_set(a), gram_set(b)
        return len(x & y) / len(x | y) if x and y else 0.0

    def pairs(vectors, times=None, *, window_seconds=None, chunk=512):
        for path in sorted(shard_dir.glob("*.npz")):
            with np.load(path) as shard:
                keep = (shard["sim"] >= cut) & eligible[shard["i"]] & eligible[shard["j"]]
                i, j, sim = shard["i"][keep], shard["j"][keep], shard["sim"][keep]
            stats["at_cut"] += len(i)
            overlap = np.fromiter((jaccard(a, b) for a, b in zip(i, j)), dtype=np.float32, count=len(i))
            passed = overlap >= floor
            stats["kept"] += int(passed.sum())
            if passed.any():
                yield i[passed], j[passed], sim[passed]

    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--shards", type=Path, required=True, help="pairs/ recorded by 01_pairs at or below --cut")
    ap.add_argument("--cut", type=float, default=0.85)
    ap.add_argument("--floor", type=float, default=0.10)
    ap.add_argument("--texts", type=Path, help="row-aligned texts to use instead of texts.parquet")
    ap.add_argument("--out-name", default=None, help="output folder; default sweep_lex<floor>")
    args = ap.parse_args()

    rows = pd.read_parquet(args.export_dir / "rows.parquet")
    texts = pd.read_parquet(args.texts or args.export_dir / "texts.parquet")
    if not (texts["post_id"].astype(str).to_numpy() == rows["post_id"].astype(str).to_numpy()).all():
        raise SystemExit("texts are not row-aligned with rows.parquet - re-run 00_export")

    # The recorded i, j index posts in the order text_similarity_network
    # iterates them - sorted by created_at, stably - not in row order. Texts
    # have to be put in that same order or every pair compares the wrong posts.
    order = np.argsort(rows["created_at"].to_numpy(), kind="stable")
    clean = texts["clean"].fillna("").to_numpy()[order]
    eligible = (texts["eligible"].to_numpy() if "eligible" in texts else np.ones(len(texts), bool))[order]
    if not eligible.all():
        print(f"{int((~eligible).sum()):,} of {len(eligible):,} posts no longer text-eligible")

    stats = {"at_cut": 0, "kept": 0}
    start = time.perf_counter()
    edges = coord2.text_similarity_network(
        # The replayed pairs never read the vectors, so none are loaded.
        rows[["user_id", "created_at"]], np.zeros((len(rows), 1), dtype=np.float32), threshold=args.cut,
        similarity=floored(args.shards, args.cut, args.floor, clean, eligible, stats),
    )
    out = args.export_dir / (args.out_name or f"sweep_lex{args.floor:.2f}")
    out.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(out / f"edges_t{args.cut:.2f}.parquet")
    users = len(set(edges["source"]) | set(edges["target"]))
    print(f"cut {args.cut}, floor {args.floor}: {stats['kept']:,} of {stats['at_cut']:,} post pairs kept "
          f"({stats['kept'] / max(stats['at_cut'], 1):.1%}); {len(edges):,} edges over {users:,} users "
          f"in {time.perf_counter() - start:.0f}s -> {out}")


if __name__ == "__main__":
    main()

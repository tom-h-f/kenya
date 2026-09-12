"""Encode the exported backlog on the GPU box with production's model and settings.

    python 02_encode.py out/ [--limit 2000]

Production (`kma.semantic.embed_new`) encodes the raw post text with
paraphrase-multilingual-mpnet-base-v2, normalised, in fp32 on CPU. This does
the same on CUDA in fp32 - not fp16, because these vectors are written into the
production prefix and have to be interchangeable with the ones already there.

First re-encodes `check.parquet` and compares with the persisted vectors; it
refuses to encode the backlog if the worst cosine falls below `--min-cosine`.
Needs no R2 credentials.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def encode(model, texts: list[str], batch: int) -> np.ndarray:
    order = np.argsort([len(t) for t in texts])[::-1]
    vectors = model.encode([texts[k] for k in order], batch_size=batch, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
    out = np.empty_like(vectors, dtype=np.float32)
    out[order] = vectors
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="first N pending posts only, for a timing run")
    ap.add_argument("--min-cosine", type=float, default=0.999)
    args = ap.parse_args()

    model = SentenceTransformer(MODEL, device="cuda")
    check = pd.read_parquet(args.out_dir / "check.parquet")
    got = encode(model, check["text"].fillna("").tolist(), args.batch)
    stored = np.vstack(check["embedding"].to_numpy()).astype(np.float32)
    stored /= np.linalg.norm(stored, axis=1, keepdims=True)
    cos = (got * stored).sum(1)
    print(f"check on {len(check):,} embedded posts: cosine min {cos.min():.6f}, "
          f"p01 {np.quantile(cos, 0.01):.6f}, median {np.median(cos):.6f}")
    if cos.min() < args.min_cosine:
        raise SystemExit(f"GPU vectors do not reproduce production's (min cosine {cos.min():.6f}); nothing encoded")

    pending = pd.read_parquet(args.out_dir / "pending.parquet")
    if args.limit:
        pending = pending.head(args.limit)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    vectors = encode(model, pending["text"].fillna("").tolist(), args.batch)
    seconds = time.perf_counter() - start
    scope = f"_limit{args.limit}" if args.limit else ""
    np.save(args.out_dir / f"vectors{scope}.npy", vectors)
    pending[["platform_post_id"]].to_parquet(args.out_dir / f"vector_ids{scope}.parquet")
    print(f"encoded {len(pending):,} posts in {seconds:.0f}s ({len(pending) / seconds:,.0f}/s), "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()

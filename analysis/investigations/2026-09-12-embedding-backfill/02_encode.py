"""Encode the exported backlog on the GPU box with production's model and settings.

    python 02_encode.py out/ [--limit 2000]

Production (`kma.semantic.embed_new`) encodes the raw post text with
paraphrase-multilingual-mpnet-base-v2, normalised, in fp32 on CPU. This does
the same on CUDA in fp32 - not fp16, because these vectors are written into the
production prefix and have to be interchangeable with the ones already there.

First re-encodes `check.parquet` and compares with the persisted vectors; it
refuses to encode the backlog if the worst cosine falls below `--min-cosine`.
Needs no R2 credentials.

Encodes in chunks and saves each one, for the same reason `03_write.py` records
its uploads: a run that dies at 90% must cost one chunk rather than the whole
pass. Two runs were lost this way - the mac's low-memory watchdog killed a
batch-256 pass, and the retry was abandoned mid-flight - and neither left
anything behind.
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
    ap.add_argument("--chunk", type=int, default=10_000, help="posts per saved part")
    args = ap.parse_args()

    # CUDA on the GPU box, MPS on the mac, CPU anywhere else: the vectors go
    # into the production prefix either way, and the check below is what proves
    # the device did not change them.
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    model = SentenceTransformer(MODEL, device=device)
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
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    scope = f"_limit{args.limit}" if args.limit else ""
    texts = pending["text"].fillna("").tolist()
    parts_dir = args.out_dir / f"parts{scope}"
    parts_dir.mkdir(exist_ok=True)

    start = time.perf_counter()
    encoded = 0
    paths = []
    for k, lo in enumerate(range(0, len(texts), args.chunk)):
        path = parts_dir / f"{k:04d}.npy"
        paths.append(path)
        if path.exists():
            print(f"part {k}: already encoded, skipping", flush=True)
            continue
        chunk_start = time.perf_counter()
        got = encode(model, texts[lo : lo + args.chunk], args.batch)
        # Write to a temp name and rename, so a kill mid-write cannot leave a
        # truncated part that the next run would skip as done.
        # `.tmp.npy`, not `.npy.tmp`: np.save appends `.npy` to any name that
        # does not already end in it, so the temp file would land somewhere the
        # rename could not find.
        tmp = path.with_name(f"{path.stem}.tmp.npy")
        np.save(tmp, got)
        tmp.rename(path)
        encoded += len(got)
        print(
            f"part {k}: {len(got):,} posts in {time.perf_counter() - chunk_start:.0f}s "
            f"-> {path.name}",
            flush=True,
        )
    vectors = np.concatenate([np.load(p) for p in paths]) if paths else np.empty((0, 768))
    seconds = time.perf_counter() - start
    np.save(args.out_dir / f"vectors{scope}.npy", vectors)
    pending[["platform_post_id"]].to_parquet(args.out_dir / f"vector_ids{scope}.parquet")
    peak = f", peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if device == "cuda" else ""
    rate = f"{encoded / seconds:,.0f}/s" if encoded and seconds else "resumed"
    print(
        f"encoded {encoded:,} of {len(pending):,} posts in {seconds:.0f}s ({rate}){peak}; "
        f"vectors{scope}.npy holds {len(vectors):,}"
    )


if __name__ == "__main__":
    main()

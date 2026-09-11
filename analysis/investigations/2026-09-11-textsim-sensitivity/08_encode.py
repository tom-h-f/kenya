"""Re-encode an export's posts with another encoder, as a new export 01_pairs can read.

    python 08_encode.py out/<export-dir> --model BAAI/bge-m3 [--limit 2000]

Writes `out/<export-dir>__<model-slug>/` holding the same `rows.parquet` (same
rows, same order) and a `matrix.npy` of that encoder's unit vectors, so
`01_pairs.py` and `02_score.py` run on it unchanged and every difference in the
result is the encoder's. Runs on the GPU box: it needs `texts.parquet` from
`00_export.py --texts-only`, and no R2 credentials.

Encodes the raw post text, as `07_encoder_check.py` scored it. Batches are
sorted by length so padding does not dominate; vectors are written back in row
order.
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

PREFIX = {"intfloat/multilingual-e5-large": "query: "}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only, for a timing run")
    args = ap.parse_args()

    rows = pd.read_parquet(args.export_dir / "rows.parquet")
    texts = pd.read_parquet(args.export_dir / "texts.parquet")
    if not (texts["post_id"].astype(str).to_numpy() == rows["post_id"].astype(str).to_numpy()).all():
        raise SystemExit("texts.parquet is not row-aligned with rows.parquet - re-run 00_export")
    if args.limit:
        rows, texts = rows.head(args.limit), texts.head(args.limit)

    model = SentenceTransformer(args.model, device="cuda", model_kwargs={"dtype": torch.float16})
    model.max_seq_length = args.max_seq
    inputs = [PREFIX.get(args.model, "") + t for t in texts["text"].fillna("")]
    order = np.argsort([len(t) for t in inputs])[::-1]

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    encoded = model.encode(
        [inputs[k] for k in order], batch_size=args.batch, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )
    seconds = time.perf_counter() - start
    matrix = np.empty_like(encoded, dtype=np.float32)
    matrix[order] = encoded

    slug = args.model.split("/")[-1]
    scope = f"__limit{args.limit}" if args.limit else ""
    out = args.export_dir.parent / f"{args.export_dir.name}__{slug}{scope}"
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / "rows.parquet")
    np.save(out / "matrix.npy", matrix)
    if not args.limit:
        shutil.copy(args.export_dir / "texts.parquet", out / "texts.parquet")

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"{args.model}: {len(rows):,} posts in {seconds:.0f}s ({len(rows) / seconds:,.0f}/s), "
          f"matrix {matrix.shape}, peak {peak:.2f} GiB -> {out}")


if __name__ == "__main__":
    main()

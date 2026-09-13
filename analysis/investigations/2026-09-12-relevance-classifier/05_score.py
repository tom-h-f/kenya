"""Score the exported corpus with the trained relevance classifier.

    python 05_score.py data/ [--limit 20000]

Writes `scores.parquet` (platform_post_id, p_kenya) next to the input. Runs on
the GPU box and needs no R2 credentials. Batches are sorted by length so
padding does not dominate, and scores are written back in input order.

The probability is kept rather than a boolean: the 0.5 cut is not tuned, and a
persisted probability lets the threshold move without re-scoring 1M posts.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 128


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("--model", default=None, help="defaults to <data>/model")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    posts = pd.read_parquet(args.data / "corpus.parquet")
    if args.limit:
        posts = posts.head(args.limit)
    texts = posts["text"].fillna("").astype(str).tolist()

    path = str(args.model or args.data / "model")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).cuda().eval()

    order = np.argsort([len(t) for t in texts])[::-1]
    probs = np.empty(len(texts), dtype=np.float32)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for s in range(0, len(order), args.batch):
        index = order[s : s + args.batch]
        encoded = tokenizer([texts[k] for k in index], truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**{k: v.cuda() for k, v in encoded.items()}).logits.float()
        probs[index] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        if s % (args.batch * 200) == 0:
            print(f"  {s + len(index):,}/{len(texts):,} ({time.perf_counter() - start:.0f}s)", flush=True)

    seconds = time.perf_counter() - start
    scope = f"_limit{args.limit}" if args.limit else ""
    posts[["platform_post_id"]].assign(p_kenya=probs).to_parquet(args.data / f"scores{scope}.parquet")
    print(f"scored {len(texts):,} posts in {seconds:.0f}s ({len(texts) / seconds:,.0f}/s), "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB; "
          f"p>=0.5 on {float((probs >= 0.5).mean()):.3f} of them")


if __name__ == "__main__":
    main()

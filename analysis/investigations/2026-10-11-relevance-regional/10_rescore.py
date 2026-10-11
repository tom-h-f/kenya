"""Score the whole exported corpus with a published model, on mike-pc, resumably.

    python 10_rescore.py corpus.parquet --model new_model --out scores_new/ [--chunk 250000] [--limit N]

Writes `part_0000.parquet`, ... (platform_post_id, p_kenya) into --out; parts
already there are skipped, so an interrupted pass resumes. The text is prepared
exactly as `kma.relevance.score_texts` does when serving (`relevance_prep`, read
from the model directory). No R2 credentials needed: the parts come back to the
Mac, which writes them to the `relevance/` prefix with `11_write_scores.py`.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import relevance_prep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(args.corpus, columns=["platform_post_id", "text"])
    ids = table.column("platform_post_id").to_pylist()
    texts = table.column("text").to_pylist()
    if args.limit:
        ids, texts = ids[: args.limit], texts[: args.limit]
    variant, handles = relevance_prep.load_for_model(args.model)
    print(f"{len(ids):,} posts, prep {variant}, {len(handles)} handles", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model)).cuda().eval()

    torch.cuda.reset_peak_memory_stats()
    for part, s in enumerate(range(0, len(ids), args.chunk)):
        target = args.out / f"part_{part:04d}.parquet"
        if target.exists():
            continue
        start = time.perf_counter()
        chunk = [relevance_prep.prepare(t, variant, handles) for t in texts[s : s + args.chunk]]
        order = np.argsort([len(t) for t in chunk])[::-1]
        probs = np.empty(len(chunk), dtype=np.float32)
        for b in range(0, len(order), args.batch):
            index = order[b : b + args.batch]
            enc = tokenizer([chunk[k] for k in index], truncation=True, max_length=128, padding=True,
                            return_tensors="pt")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**{k: v.cuda() for k, v in enc.items()}).logits.float()
            probs[index] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        tmp = target.with_suffix(".tmp")
        pd.DataFrame({"platform_post_id": ids[s : s + args.chunk], "p_kenya": probs}).to_parquet(tmp)
        tmp.rename(target)
        n = len(chunk)
        print(f"part {part}: {n:,} in {time.perf_counter() - start:.0f}s ({n / (time.perf_counter() - start):,.0f}/s), "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, p>=0.5 {float((probs >= 0.5).mean()):.3f}",
              flush=True)
    print("RESCORE_DONE", flush=True)


if __name__ == "__main__":
    main()

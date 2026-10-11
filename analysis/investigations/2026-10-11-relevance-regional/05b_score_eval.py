"""Score the eval sets with a saved model directory: the old model, or a finished run.

    python 05b_score_eval.py data/ --model old_model --name old
    python 05b_score_eval.py data/ --model data/runs/handles_s0/model --name handles_s0_saved

Applies the preparation recorded beside the model (`prep.json`, `strip` when absent),
the same way `kma.relevance.score_texts` does when serving.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import relevance_prep


def score(model, tokenizer, texts: list[str], batch: int = 128) -> np.ndarray:
    order = np.argsort([len(t) for t in texts])[::-1]
    out = np.empty(len(texts), dtype=np.float32)
    for s in range(0, len(order), batch):
        index = order[s : s + batch]
        enc = tokenizer([texts[k] for k in index], truncation=True, max_length=128, padding=True, return_tensors="pt")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**{k: v.cuda() for k, v in enc.items()}).logits.float()
        out[index] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    variant, handles = relevance_prep.load_for_model(args.model)
    print(f"prep {variant}, {len(handles)} handles", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model)).cuda().eval()
    out = args.data / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.data.glob("eval_*.parquet")):
        frame = pd.read_parquet(path)
        texts = [relevance_prep.prepare(t, variant, handles) for t in frame["text"].astype(str)]
        frame["p_kenya"] = score(model, tokenizer, texts)
        frame.to_parquet(out / f"preds_{path.stem.removeprefix('eval_')}.parquet")
        print(f"  {path.stem}: {len(frame)}", flush=True)


if __name__ == "__main__":
    main()

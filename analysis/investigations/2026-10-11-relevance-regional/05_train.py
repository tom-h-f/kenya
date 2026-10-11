"""Fine-tune the relevance classifier on mike-pc, with regional hard negatives, and score every eval set.

    python 05_train.py data/ --variant strip --seed 0 --name strip_s0
    python 05_train.py data/ --variant handles --seed 0 --name handles_s0 --handles data/handles.txt

`data/` holds `train.parquet` (post_id, text, label kenya|offdomain, source) and
any number of `eval_<name>.parquet` (post_id, text, ...). Eval sets are scored
with the finished model and written as `runs/<name>/preds_<name>.parquet`, so
the comparison against the old model is made by one script on one machine.

The recipe is 2026-09-12's: afro-xlmr-large, fp32 weights, bf16 autocast,
8-bit AdamW, batch 32, gradient checkpointing. The text a model sees is
`relevance_prep.prepare`, the same function serving uses.
"""

import argparse
import json
import time
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import relevance_prep

BASE = "Davlan/afro-xlmr-large"
MAX_LEN = 128


def batches(tokenizer, texts, labels, batch, shuffle, seed=0):
    order = np.random.default_rng(seed).permutation(len(texts)) if shuffle else np.arange(len(texts))
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        yield {k: v.cuda() for k, v in enc.items()}, torch.tensor([labels[i] for i in idx]).cuda()


@torch.no_grad()
def predict(model, tokenizer, texts, batch=128) -> np.ndarray:
    model.eval()
    order = np.argsort([len(t) for t in texts])[::-1]
    out = np.empty(len(texts), dtype=np.float32)
    for s in range(0, len(order), batch):
        index = order[s : s + batch]
        enc = tokenizer([texts[k] for k in index], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**{k: v.cuda() for k, v in enc.items()}).logits.float()
        out[index] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
    return out


def f1_of(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("--variant", choices=relevance_prep.VARIANTS, default="strip")
    ap.add_argument("--handles", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="keep the weights (2.1 GB)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    handles = relevance_prep.load_handles(args.handles) if args.handles else frozenset()
    prep = lambda texts: [relevance_prep.prepare(t, args.variant, handles) for t in texts]
    labelled = pd.read_parquet(args.data / "train.parquet")
    labelled = labelled[labelled["label"].isin(("kenya", "offdomain"))].reset_index(drop=True)
    if args.limit:
        labelled = labelled.sample(args.limit, random_state=0).reset_index(drop=True)
    rng = np.random.default_rng(0)
    is_val = np.zeros(len(labelled), dtype=bool)
    is_val[rng.permutation(len(labelled))[: max(1, len(labelled) // 10)]] = True
    val, train = labelled[is_val], labelled[~is_val]
    y = lambda f: (f["label"] == "kenya").astype(int).tolist()
    print(f"[{args.name}] train {len(train):,} ({sum(y(train)):,} kenya), val {len(val):,}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2, dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable()
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    train_text, val_text = prep(train["text"].tolist()), prep(val["text"].tolist())

    torch.cuda.reset_peak_memory_stats()
    best, best_state, start, history = -1.0, None, time.perf_counter(), []
    for epoch in range(args.epochs):
        model.train()
        for enc, labels in batches(tokenizer, train_text, y(train), args.batch, True, seed=args.seed * 100 + epoch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
        p, r, f1 = f1_of(np.array(y(val), dtype=bool), predict(model, tokenizer, val_text) >= 0.5)
        peak = torch.cuda.max_memory_allocated() / 2**30
        history.append({"epoch": epoch + 1, "val_precision": p, "val_recall": r, "val_f1": f1, "peak_gib": peak})
        print(f"[{args.name}] epoch {epoch + 1}: val P {p:.3f} R {r:.3f} F1 {f1:.3f} "
              f"({time.perf_counter() - start:.0f}s, peak {peak:.2f} GiB)", flush=True)
        if f1 > best:
            best, best_state = f1, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)

    out = args.data / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.data.glob("eval_*.parquet")):
        frame = pd.read_parquet(path)
        frame["p_kenya"] = predict(model, tokenizer, prep(frame["text"].astype(str).tolist()))
        frame.to_parquet(out / f"preds_{path.stem.removeprefix('eval_')}.parquet")
    (out / "run.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                              "history": history, "train_rows": len(train),
                                              "seconds": time.perf_counter() - start}, indent=1))
    if args.save:
        model.save_pretrained(out / "model")
        tokenizer.save_pretrained(out / "model")
        (out / "model" / relevance_prep.PREP_FILE).write_text(json.dumps({"variant": args.variant}))
        if handles:
            (out / "model" / relevance_prep.HANDLES_FILE).write_text("\n".join(sorted(handles)))
    print(f"[{args.name}] done in {time.perf_counter() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()

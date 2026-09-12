"""Fine-tune a kenya-vs-offdomain classifier on mike-pc, then score it on the B2 labels.

    python 03_train.py data/ [--epochs 3] [--limit 500]

`data/` holds `labelled.parquet` (from 02_label.py), `measure_human.csv` (Tom's
100) and `measure_sample.csv` (the model's 300). Neither B2 file is in the
training data - 01_sample.py excluded them.

Binary, `unclear` dropped, because that is how the gate is scored
(`measure_eval.score`): precision and recall of the kenya call over posts
someone could place. Base model `Davlan/afro-xlmr-large` (public, so no token
on this box). Recipe from the 2026-09-10 measurement on this card: fp32
weights, bf16 autocast, 8-bit AdamW, batch 32 with gradient checkpointing -
5.3 GiB, under the ~7 GiB point where Windows starts spilling to shared memory.

Reports, on the same posts: the keyword gate, this model, and "gate OR model"
(the gate's precision is 0.971, so its kenya calls can stand and the model only
has to find what it misses). Corpus-weighted recall uses the stratum shares
each sample carries, as `measure_eval.score` does.
"""

import argparse
import json
import time
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

BASE = "Davlan/afro-xlmr-large"
MAX_LEN = 128


def batches(tokenizer, texts, labels, batch, shuffle, seed=0):
    order = np.random.default_rng(seed).permutation(len(texts)) if shuffle else np.arange(len(texts))
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt")
        yield {k: v.cuda() for k, v in enc.items()}, torch.tensor([labels[i] for i in idx]).cuda()


@torch.no_grad()
def predict(model, tokenizer, texts, batch=128) -> np.ndarray:
    model.eval()
    probs = []
    for enc, _ in batches(tokenizer, texts, [0] * len(texts), batch, shuffle=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits.float()
        probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def scores(frame: pd.DataFrame, predicted: np.ndarray, weighted: bool) -> dict:
    actual = (frame["label"] == "kenya").to_numpy()
    w = frame["stratum_share"].to_numpy() / frame.groupby("bucket")["bucket"].transform("size").to_numpy() \
        if weighted else np.ones(len(frame))
    tp, fp, fn = w[predicted & actual].sum(), w[predicted & ~actual].sum(), w[~predicted & actual].sum()
    return {"precision": round(float(tp / (tp + fp)), 3) if tp + fp else None,
            "recall": round(float(tp / (tp + fn)), 3) if tp + fn else None,
            "n": int(len(frame))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="train on N rows only, for a timing run")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    torch.manual_seed(0)

    labelled = pd.read_parquet(args.data / "labelled.parquet")
    labelled = labelled[labelled["label"] != "unclear"].sample(frac=1.0, random_state=0).reset_index(drop=True)
    if args.limit:
        labelled = labelled.head(args.limit)
    val = labelled.iloc[: max(1, len(labelled) // 10)]
    train = labelled.iloc[len(val):]
    y = lambda f: (f["label"] == "kenya").astype(int).tolist()
    print(f"train {len(train):,} ({int(sum(y(train))):,} kenya), val {len(val):,}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2, dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable()
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    torch.cuda.reset_peak_memory_stats()
    best, best_state, start = -1.0, None, time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        for enc, labels in batches(tokenizer, train["text"].tolist(), y(train), args.batch, True, seed=epoch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
        p = predict(model, tokenizer, val["text"].tolist()) >= args.threshold
        v = scores(val.assign(bucket="all", stratum_share=1.0), p, weighted=False)
        f1 = 2 * v["precision"] * v["recall"] / (v["precision"] + v["recall"]) if v["precision"] and v["recall"] else 0
        print(f"epoch {epoch + 1}: val precision {v['precision']} recall {v['recall']} F1 {f1:.3f} "
              f"({time.perf_counter() - start:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB)",
              flush=True)
        if f1 > best:
            best, best_state = f1, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)

    out = args.data / ("model" + (f"_limit{args.limit}" if args.limit else ""))
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    report = {}
    # Both corpus-weighted by the gate's buckets, as the recorded B2 figures are
    # (human recall 0.655 weighted; unweighted it reads 0.892).
    for name, path, weighted in (("human 100", "measure_human.csv", True), ("model 300", "measure_sample.csv", True)):
        frame = pd.read_csv(args.data / path)
        if "bucket" not in frame:
            frame = frame.merge(pd.read_csv(args.data / "measure_sample.csv")[["post_id", "bucket", "stratum_share"]],
                                on="post_id")
        frame = frame[frame["label"].isin(("kenya", "offdomain"))].reset_index(drop=True)
        # The gate's own call, recorded on both B2 files when they were drawn.
        gate = frame["bucket"].eq("kenya").to_numpy()
        learned = predict(model, tokenizer, frame["text"].astype(str).tolist()) >= args.threshold
        report[name] = {"gate": scores(frame, gate, weighted), "model": scores(frame, learned, weighted),
                        "gate or model": scores(frame, gate | learned, weighted)}
    print(json.dumps(report, indent=1))
    (out / "b2_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

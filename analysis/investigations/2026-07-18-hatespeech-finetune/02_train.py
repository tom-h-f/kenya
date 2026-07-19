# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
#     "scikit-learn>=1.4",
#     "torch>=2.4",
#     "transformers>=4.48",
#     "accelerate>=1.0",
# ]
# ///
"""Fine-tune a 3-class hate/offensive/neither classifier.

Defaults to a sample smoke run (1k rows, 1 epoch, model saved to
out/model-sample); pass --full for the real run (out/model). Class-weighted
cross-entropy for the 11:3:1 imbalance; best epoch by val macro-F1.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from _common import ID2LABEL, LABEL2ID, OUT, SEED, device, load_split

DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base"


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len: int, weighted: bool = False):
        enc = tokenizer(
            df["text"].tolist(),
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        self.encodings = enc
        self.labels = df["label"].tolist()
        self.weights = df["agreement"].tolist() if weighted else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = self.labels[i]
        if self.weights is not None:
            item["sample_weight"] = self.weights[i]
        return item


class PaddingCollator:
    def __init__(self, tokenizer):
        from transformers import DataCollatorWithPadding

        self.pad = DataCollatorWithPadding(tokenizer)

    def __call__(self, feats):
        weights = (
            [f.pop("sample_weight") for f in feats]
            if "sample_weight" in feats[0]
            else None
        )
        batch = self.pad(feats)
        if weights is not None:
            batch["sample_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        sample_weight = inputs.pop("sample_weight", None)
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights.to(outputs.logits.device),
            reduction="none",
        )
        if sample_weight is not None:
            loss = loss * sample_weight
        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "hate_f1": f1_score(labels, preds, labels=[2], average="macro"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default=None, help="suffix for out/model-<tag>")
    ap.add_argument("--agreement-min", type=float, default=0.0)
    ap.add_argument("--weight-by-agreement", action="store_true")
    ap.add_argument("--extra-data", default=None, help="parquet appended to train")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    args = ap.parse_args()

    train, val = load_split("train"), load_split("val")
    if args.agreement_min > 0:
        before = len(train)
        train = train[train["agreement"] >= args.agreement_min]
        print(f"agreement >= {args.agreement_min}: {before} -> {len(train)} train rows")
    if args.extra_data:
        import pandas as pd

        extra = pd.read_parquet(args.extra_data)
        train = pd.concat(
            [train, extra[["text", "label", "agreement"]]], ignore_index=True
        )
        print(f"+{len(extra)} extra rows from {args.extra_data} -> {len(train)}")
    full_steps = math.ceil(len(train) / args.batch_size) * args.epochs
    if not args.full:
        import pandas as pd

        train = pd.concat(
            g.sample(min(len(g), 400), random_state=SEED)
            for _, g in train.groupby("label")
        )
        val = val.sample(300, random_state=SEED)
        args.epochs = 1
    if args.tag:
        out_dir = OUT / f"model-{args.tag}"
    else:
        out_dir = OUT / ("model" if args.full else "model-sample")

    dev = device()
    print(f"device={dev} model={args.model} train={len(train)} val={len(val)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    counts = train["label"].value_counts().sort_index()
    weights = torch.tensor(
        (len(train) / (3 * counts)).values, dtype=torch.float32
    )
    print(f"class weights: {weights.tolist()}")

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=SEED,
        fp16=dev == "cuda",
        report_to=[],
    )
    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=TweetDataset(
            train, tokenizer, args.max_len, weighted=args.weight_by_agreement
        ),
        eval_dataset=TweetDataset(val, tokenizer, args.max_len),
        data_collator=PaddingCollator(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume or None)
    elapsed = time.time() - t0

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    final = trainer.evaluate()
    print(f"val macro-F1: {final['eval_macro_f1']:.4f} ({elapsed:.0f}s)")

    if not args.full:
        steps = math.ceil(len(train) / args.batch_size) * args.epochs
        per_step = elapsed / steps
        print(
            f"~{per_step:.2f}s/step -> projected --full "
            f"({full_steps} steps, {ap.get_default('epochs')} epochs): "
            f"~{per_step * full_steps / 60:.0f} min"
        )

    (out_dir / "train_log.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "device": dev,
                "seconds": elapsed,
                "final_eval": final,
                "log_history": trainer.state.log_history,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

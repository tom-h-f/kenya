# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
#     "torch>=2.4",
#     "transformers>=4.56",
#     "accelerate>=1.0",
# ]
# ///
"""Domain-adaptive pretraining: continue MLM on the collected-posts corpus.

Defaults to a sample smoke run (1k texts, 20 steps, out/dapt-sample); pass
--full for the real run (out/dapt-afro-xlmr). Checkpoints every --save-steps
to survive Colab disconnects; re-running auto-resumes. Stock-model holdout
MLM loss is recorded first so the DAPT gain is measurable.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from _common import OUT, device

DEFAULT_MODEL = "Davlan/afro-xlmr-large"


class MlmDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len: int):
        self.enc = tokenizer(texts, truncation=True, max_length=max_len, padding=False)

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


class TimingProbe(TrainerCallback):
    def __init__(self, probe: int, abort_hours: float | None):
        self.probe = probe
        self.abort_hours = abort_hours
        self.t0 = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            self.t0 = time.time()
        elif state.global_step == self.probe and self.t0:
            per = (time.time() - self.t0) / (self.probe - 1)
            hrs = per * state.max_steps / 3600
            print(
                f"[probe] {per:.2f}s/optim-step x {state.max_steps} steps "
                f"-> projected {hrs:.1f}h"
            )
            if self.abort_hours and hrs > self.abort_hours:
                print(
                    f"[probe] STOPPING: projection > {self.abort_hours}h. "
                    "Re-run with --epochs 1 (still worthwhile) or wait for A100."
                )
                control.should_training_stop = True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--corpus", default=str(OUT / "dapt_corpus.parquet"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing checkpoints, start over")
    ap.add_argument("--abort-over-hours", type=float, default=8.0)
    ap.add_argument("--grad-checkpoint", action="store_true")
    args = ap.parse_args()

    out_dir = OUT / ("dapt-afro-xlmr" if args.full else "dapt-sample")
    ckpt_dir = out_dir / "checkpoints"

    df = pd.read_parquet(args.corpus)
    texts = df["text"].sample(frac=1.0, random_state=args.seed).tolist()
    if not args.full:
        texts = texts[:1000]
        args.save_steps = 10
    n_hold = max(20, int(len(texts) * 0.02))
    hold, train = texts[:n_hold], texts[n_hold:]

    dev = device()
    print(f"device={dev} model={args.model} train={len(train)} holdout={len(hold)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model, dtype=torch.float32)

    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        max_steps=20 if not args.full else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        prediction_loss_only=True,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.grad_checkpoint,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        logging_steps=50 if args.full else 5,
        seed=args.seed,
        fp16=dev == "cuda",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=MlmDataset(train, tokenizer, args.max_len),
        eval_dataset=MlmDataset(hold, tokenizer, args.max_len),
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm_probability=0.15
        ),
        processing_class=tokenizer,
        callbacks=[
            TimingProbe(100 if args.full else 10, args.abort_over_hours)
        ],
    )

    baseline_path = out_dir / "baseline.json"
    if not baseline_path.exists():
        set_seed(args.seed)
        base = trainer.evaluate()
        baseline = {
            "eval_loss": base["eval_loss"],
            "ppl": float(torch.exp(torch.tensor(base["eval_loss"]))),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(baseline, indent=2))
        print(f"stock baseline: loss {baseline['eval_loss']:.4f} "
              f"ppl {baseline['ppl']:.1f}")
    else:
        baseline = json.loads(baseline_path.read_text())
        print(f"stock baseline (cached): loss {baseline['eval_loss']:.4f}")

    resume = not args.fresh and (
        args.resume or any(ckpt_dir.glob("checkpoint-*"))
    )
    if resume:
        print("resuming from checkpoint")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume or None)
    elapsed = time.time() - t0

    set_seed(args.seed)
    final = trainer.evaluate()
    final_ppl = float(torch.exp(torch.tensor(final["eval_loss"])))
    gain = baseline["eval_loss"] - final["eval_loss"]
    print(
        f"MLM holdout: stock {baseline['eval_loss']:.4f} -> "
        f"dapt {final['eval_loss']:.4f} (gain {gain:.4f}, ppl {final_ppl:.1f})"
    )
    if gain < 0.10:
        print("WARNING: DAPT gain < 0.10 - gate NOT met, model may not be "
              "worth using over stock. Say so loudly in findings.md.")

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "dapt_log.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "device": dev,
                "seconds": elapsed,
                "baseline": baseline,
                "final_eval_loss": final["eval_loss"],
                "final_ppl": final_ppl,
                "gain": gain,
                "log_history": trainer.state.log_history,
            },
            indent=2,
            default=str,
        )
    )
    print(f"saved to {out_dir} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()

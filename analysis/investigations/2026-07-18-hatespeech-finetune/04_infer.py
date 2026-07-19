# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
#     "torch>=2.4",
#     "transformers>=4.48",
# ]
# ///
"""Inference helper for the fine-tuned classifier.

Library: `predict(texts) -> DataFrame(label, label_id, p_neither, p_offensive,
p_hate)`. CLI: score a CSV column, e.g.
`uv run 04_infer.py --csv posts.csv --column text --out scored.csv`.
This is the seam for later kma pipeline wiring.
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch

from _common import LABELS, OUT, device


def predict(
    texts: list[str],
    model_dir: str | None = None,
    batch_size: int = 64,
    max_len: int = 128,
) -> pd.DataFrame:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = model_dir or str(OUT / "model")
    dev = device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(dev).eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i : i + batch_size],
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(dev)
            probs.append(torch.softmax(model(**enc).logits, dim=-1).cpu())
    p = torch.cat(probs).numpy()

    label_id = p.argmax(axis=1)
    return pd.DataFrame(
        {
            "label": [LABELS[i] for i in label_id],
            "label_id": label_id,
            "p_neither": p[:, 0],
            "p_offensive": p[:, 1],
            "p_hate": p[:, 2],
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--column", default="text")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model-dir", default=str(OUT / "model"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    scored = predict(df[args.column].astype(str).tolist(), model_dir=args.model_dir)
    result = pd.concat([df.reset_index(drop=True), scored], axis=1)

    out = args.out or args.csv.rsplit(".", 1)[0] + "_scored.csv"
    result.to_csv(out, index=False)
    print(result["label"].value_counts().to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

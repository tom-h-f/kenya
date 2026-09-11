"""Which encoder separates same-message pairs from the rest?

    cd analysis && uv run python investigations/2026-09-11-textsim-sensitivity/07_encoder_check.py \\
        investigations/2026-09-11-textsim-sensitivity/out/2026-09-05-promotion-off

Scores every blind-labelled pair from `06_pair_audit.py --judge` under each
candidate encoder and reports how well its cosine ranks same_message above the
rest (AUC, bootstrap 95% interval), and how many non-matching pairs it still
admits at the cut that keeps 90% of the matches.

Every pair here was first selected by the current encoder at >= its cut, so
this measures precision among what the trace already admits. It says nothing
about matches the current encoder misses; an encoder that wins here still has
to be re-run over the corpus before it can replace the trace.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kma import coord2

CANDIDATES = {
    "paraphrase-multilingual-mpnet (current)": ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", ""),
    "stsb-xlm-r-multilingual (paper)": ("sentence-transformers/stsb-xlm-r-multilingual", ""),
    "LaBSE": ("sentence-transformers/LaBSE", ""),
    "multilingual-e5-large": ("intfloat/multilingual-e5-large", "query: "),
    "bge-m3": ("BAAI/bge-m3", ""),
}
BOOTSTRAP = 2000


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def auc_interval(scores: np.ndarray, positive: np.ndarray, rng) -> tuple[float, float]:
    draws = []
    for _ in range(BOOTSTRAP):
        k = rng.integers(0, len(scores), len(scores))
        draws.append(auc(scores[k][positive[k]], scores[k][~positive[k]]))
    return tuple(np.nanquantile(draws, [0.025, 0.975]))


def admitted_at_recall(scores: np.ndarray, positive: np.ndarray, recall: float = 0.9) -> float:
    cut = np.quantile(scores[positive], 1 - recall)
    return float((scores[~positive] >= cut).mean())


def labelled_pairs(snapshot_dir: Path) -> pd.DataFrame:
    files = sorted(snapshot_dir.glob("pair_audit_judged*.parquet"))
    frame = pd.concat([pd.read_parquet(f).assign(source=f.stem) for f in files], ignore_index=True)
    frame = frame.dropna(subset=["label", "text_a", "text_b"])
    key = frame[["a", "b"]].apply(lambda r: tuple(sorted(r)), axis=1)
    return frame.loc[~key.duplicated()].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("snapshot_dir", type=Path)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    pairs = labelled_pairs(args.snapshot_dir)
    positive = (pairs.label == "same_message").to_numpy()
    unrelated = (pairs.label == "unrelated").to_numpy()
    print(f"{len(pairs)} labelled pairs from {sorted(pairs.source.unique())}")
    print(pairs.label.value_counts().to_string(), "\n")

    unique = pd.unique(np.r_[pairs.text_a, pairs.text_b])
    position = {t: k for k, t in enumerate(unique)}
    a = pairs.text_a.map(position).to_numpy()
    b = pairs.text_b.map(position).to_numpy()

    scores = {
        "char-4gram Jaccard (lexical)": pairs.gram_j.to_numpy(),
        "token Jaccard (lexical)": pairs.tok_j.to_numpy(),
    }
    for name, (model_id, prefix) in CANDIDATES.items():
        model = SentenceTransformer(model_id)
        vectors = model.encode([prefix + t for t in unique], batch_size=32, normalize_embeddings=True)
        scores[name] = (vectors[a] * vectors[b]).sum(1)

    rng = np.random.default_rng(0)
    rows = []
    for name, s in scores.items():
        low, high = auc_interval(s, positive, rng)
        rows.append({
            "scorer": name,
            "AUC match vs rest": round(auc(s[positive], s[~positive]), 3),
            "95% interval": f"{low:.2f}-{high:.2f}",
            "AUC match vs unrelated": round(auc(s[positive], s[unrelated]), 3),
            "non-matches admitted at 90% recall": round(admitted_at_recall(s, positive), 3),
        })
    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    pairs.assign(**{f"score:{k}": v for k, v in scores.items()}).to_parquet(
        args.snapshot_dir / "encoder_check_scores.parquet"
    )


if __name__ == "__main__":
    main()

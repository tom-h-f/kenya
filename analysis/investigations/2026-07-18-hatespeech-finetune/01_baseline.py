# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2", "pyarrow>=18", "scikit-learn>=1.4"]
# ///
"""TF-IDF + logistic regression floor. Any transformer must beat this macro-F1."""

from __future__ import annotations

import json
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline

from _common import LABELS, OUT, SEED, load_split


def main() -> None:
    train, test = load_split("train"), load_split("test")

    t0 = time.time()
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED
        ),
    )
    pipe.fit(train["text"], train["label"])
    pred = pipe.predict(test["text"])
    elapsed = time.time() - t0

    macro = f1_score(test["label"], pred, average="macro")
    print(f"trained + scored in {elapsed:.1f}s")
    print(classification_report(test["label"], pred, target_names=LABELS, digits=3))
    print(f"baseline macro-F1: {macro:.4f}")

    (OUT / "01_baseline.json").write_text(
        json.dumps({"macro_f1": macro, "seconds": elapsed}, indent=2)
    )


if __name__ == "__main__":
    main()

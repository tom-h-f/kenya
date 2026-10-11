"""Keep the labelled sets in R2, outside the public repo.

    cd analysis && uv run python investigations/2026-10-11-relevance-regional/13_persist_training_data.py

The 3,000 labels of 2026-09-12 were only ever on disk (their `out/` is untracked)
and had to be recovered from a checkout; these are the posts and labels for this
retrain, under `training/relevance/2026-10-11/`, never read by a pipeline.
"""

from pathlib import Path

from kma.bench import _r2_client
from kma.db import BUCKET

OUT = Path(__file__).parent / "out"
PREFIX = "training/relevance/2026-10-11"
FILES = [OUT / "data" / "train.parquet", OUT / "data" / "eval_regional.parquet", OUT / "data" / "eval_human88.parquet",
         OUT / "data" / "eval_b2human.parquet", OUT / "data" / "eval_b2model.parquet", OUT / "data" / "eval_tz200.parquet",
         OUT / "labelled_train.parquet", OUT / "labelled_holdout.parquet", OUT / "old" / "labelled.parquet",
         OUT / "kenyan_handles.txt"]

client = _r2_client()
for path in FILES:
    key = f"{PREFIX}/{path.name if path.parent.name != 'old' else 'opus3000_' + path.name}"
    client.upload_file(str(path), BUCKET, key)
    print("uploaded", key, path.stat().st_size)

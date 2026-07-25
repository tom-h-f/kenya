#!/bin/bash
# Seed-1337 mix sweep: Opus extra-repeat in {1,2,3}. Repeat 5 already done as r3-opus-s1337.
# After this, 3-seed only the Pareto candidates.
set -e
cd "$(dirname "$0")"
RUN="uv run"
command -v uv >/dev/null 2>&1 || RUN="python"

evalset() {
  tag=$1; dir=$2
  for s in challenge_opus-v4 test_unanimous gold; do
    if [ -f "out/eval-$tag-${s}_metrics.json" ]; then
      echo "=== skip eval $tag/$s ==="
      continue
    fi
    $RUN 03_eval.py --model-dir "$dir" --split "$s" --prefix "eval-$tag-$s"
  done
}

train_variant() {
  tag=$1; shift
  if [ -f "out/eval-$tag-test_unanimous_metrics.json" ]; then
    echo "=== skip $tag (already evaluated) ==="
    return
  fi
  resume=""
  if ls "out/model-$tag/checkpoints"/checkpoint-* >/dev/null 2>&1; then
    resume="--resume"
    echo "=== resuming $tag from checkpoint ==="
  fi
  echo "=== train $tag: $@ ==="
  $RUN 02_train.py --full --tag "$tag" $resume --batch-size 32 "$@"
  evalset "$tag" "out/model-$tag"
  rm -rf "out/model-$tag/checkpoints"
}

BASE="--model out/dapt-afro-xlmr --grad-accum 2 --warmup-ratio 0.06 --focal-gamma 2.0 --val-split val2026_opus-v4 --agreement-min 0.6 --epochs 5 --lr 1e-5 --seed 1337"

# AfriHate ×1, Opus ×R
for R in 1 2 3; do
  tag="r3-mix-r${R}-s1337"
  train_variant "$tag" $BASE \
    --extra-data out/afrihate_swa.parquet,out/train2026_opus-v4.parquet \
    --extra-repeat "1,$R"
done

# Optional arm: Opus ×3 + hard-flag oversample ×2 (error-driven)
if [ -f out/train2026_opus-v4-hardflags.parquet ]; then
  tag="r3-mix-r3hard-s1337"
  train_variant "$tag" $BASE \
    --extra-data out/afrihate_swa.parquet,out/train2026_opus-v4.parquet,out/train2026_opus-v4-hardflags.parquet \
    --extra-repeat "1,3,2"
fi

echo "=== mix sweep summary ==="
$RUN python - <<'PY'
import json
from pathlib import Path
rows = []
for path in sorted(Path("out").glob("eval-r3-mix-*_metrics.json")):
    m = json.loads(path.read_text())
    rows.append((path.name, m.get("macro_f1"), m.get("hate_f1")))
# include prior ×5 for comparison
for path in sorted(Path("out").glob("eval-r3-opus-s1337-*_metrics.json")):
    m = json.loads(path.read_text())
    rows.append((path.name, m.get("macro_f1"), m.get("hate_f1")))
for path in sorted(Path("out").glob("eval-r3-baseline-*_metrics.json")):
    m = json.loads(path.read_text())
    rows.append((path.name, m.get("macro_f1"), m.get("hate_f1")))
for name, macro, hate in rows:
    print(f"{name}: macro_f1={macro} hate_f1={hate}")
PY
echo "=== mix sweep done ==="

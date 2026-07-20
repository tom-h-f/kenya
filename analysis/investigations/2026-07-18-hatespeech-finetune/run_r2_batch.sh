#!/bin/bash
# Round 2 (2026-07-20): does Plan A's 1,662 labelled 2026 rows improve
# detection of the 2026 coded register?
#
# Three things get scored on the same three test sets:
#   r2-baseline  - the shipped d3-s1337, NOT retrained. The control. Without
#                  this row, "improvement" has no reference point.
#   r2-mixed     - from the DAPT encoder, all data, 2026 rows oversampled 5x
#                  (they are 3% of the corpus otherwise and get drowned).
#   r2-continue  - from d3-s1337, 2026 rows ONLY, low lr - the two-stage bet.
#
# Test sets: gold (283 random-control rows - the only prevalence-honest one),
# challenge (195 lexicon/NLI rows - coded-term recall, non-circular but not a
# random sample), test_unanimous (2013-era - regression check).
# Model selection is on val2026 for every variant: we are optimising for 2026,
# not for 2013.
set -e
cd "$(dirname "$0")"

RUN="uv run"
command -v uv >/dev/null 2>&1 || RUN="python"

evalset() {
  tag=$1; dir=$2
  for s in gold challenge test_unanimous; do
    if [ -f "out/eval-$tag-${s}_metrics.json" ]; then
      echo "=== skip eval $tag/$s (done) ==="
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

# control: existing shipped model, no training
evalset r2-baseline out/model-d3-s1337

COMMON="--grad-accum 2 --warmup-ratio 0.06 --focal-gamma 2.0 --val-split val2026"

train_variant r2-mixed --model out/dapt-afro-xlmr $COMMON \
  --agreement-min 0.6 \
  --extra-data out/afrihate_swa.parquet,out/train2026.parquet \
  --extra-repeat 1,5 \
  --epochs 5 --lr 1e-5

train_variant r2-continue --model out/model-d3-s1337 $COMMON \
  --no-base-train \
  --extra-data out/train2026.parquet \
  --epochs 3 --lr 5e-6

echo "=== round 2 done ==="
grep -H macro_f1 out/eval-r2-*_metrics.json || true

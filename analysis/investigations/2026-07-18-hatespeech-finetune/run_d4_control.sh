#!/bin/bash
# Plan D control (2026-07-19): the d-dapt -> d-dapt-focal comparison changed
# TWO things at once - it added the focal term AND dropped class weighting.
# This run isolates them: plain unweighted CE, no focal, everything else
# identical to d-dapt/d-dapt-focal.
#   lands near 0.70 -> the win was removing class weights, focal incidental
#   lands near 0.63 -> the focal term is doing the work
set -e
cd "$(dirname "$0")"

RUN="uv run"
command -v uv >/dev/null 2>&1 || RUN="python"

run() {
  tag=$1; shift
  if [ -f "out/eval-$tag-unan_metrics.json" ]; then
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
  $RUN 03_eval.py --model-dir "out/model-$tag" --prefix "eval-$tag-test"
  $RUN 03_eval.py --model-dir "out/model-$tag" --split test_unanimous \
    --prefix "eval-$tag-unan"
  rm -rf "out/model-$tag/checkpoints"
}

D_ARGS="--agreement-min 0.6 --extra-data out/afrihate_swa.parquet \
       --epochs 5 --lr 1e-5 --grad-accum 2 --warmup-ratio 0.06 \
       --model out/dapt-afro-xlmr"

run d-dapt-nowt $D_ARGS --no-class-weights

echo "=== control done ==="
grep -H macro_f1 out/eval-d-dapt-nowt-*_metrics.json || true

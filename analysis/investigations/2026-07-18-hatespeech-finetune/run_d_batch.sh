#!/bin/bash
# Plan D stage 1: afro-xlmr-large ablation ladder on the canonical recipe.
# batch 8 x grad-accum 8 (T4, 560M model). NEVER --resume a checkpoint made
# under a different batch/schedule; all d-* tags are fresh dirs so this is
# moot here. Checkpoints are deleted after eval (7GB each, Drive quota).
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
       --epochs 5 --lr 1e-5 --grad-accum 2 --warmup-ratio 0.06"

run d-base       --model Davlan/afro-xlmr-large $D_ARGS
run d-dapt       --model out/dapt-afro-xlmr     $D_ARGS
run d-dapt-ls    --model out/dapt-afro-xlmr     $D_ARGS --label-smoothing 0.05
run d-dapt-focal --model out/dapt-afro-xlmr     $D_ARGS --focal-gamma 2.0

echo "=== batch done ==="
grep -H macro_f1 out/eval-d*_metrics.json || true

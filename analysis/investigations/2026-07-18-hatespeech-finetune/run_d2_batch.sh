#!/bin/bash
# Plan D stage 2 (DECISION 2026-07-19): focal beat label-smoothing decisively
# (unan 0.7024 vs 0.6373), and DAPT beat stock (0.6315 vs 0.6091), so LLRD
# goes on top of d-dapt-focal. Same batch/schedule as stage 1.
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

run d-dapt-focal-llrd --model out/dapt-afro-xlmr $D_ARGS \
  --focal-gamma 2.0 --llrd 0.9

echo "=== stage 2 done ==="
grep -H macro_f1 out/eval-d-dapt-focal-llrd-*_metrics.json || true

#!/bin/bash
# Plan D stage 3 (DECISION 2026-07-19): winner = d-dapt-focal (unan 0.7024).
# LLRD was tried on top and LOST (0.6498), so it is not in the shipped recipe.
# Seed 42 = the existing d-dapt-focal run; only two more seeds are needed for
# mean +- sd. Hate support on the unanimous test is 75 rows, so this is what
# turns the ladder table into a quotable result.
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
       --model out/dapt-afro-xlmr --focal-gamma 2.0"

run d3-s1337 $D_ARGS --seed 1337
run d3-s2027 $D_ARGS --seed 2027

echo "=== stage 3 done ==="
grep -H macro_f1 out/eval-d3-*_metrics.json || true

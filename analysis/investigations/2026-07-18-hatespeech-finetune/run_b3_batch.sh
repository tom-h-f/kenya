#!/bin/bash
# B3 label-noise + C3 AfriHate ablation batch. Sequential; works on tac2
# (uv/MPS) and Colab (plain python/CUDA). Eval on full + unanimous test.
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
  $RUN 02_train.py --full --tag "$tag" $resume --batch-size 64 "$@"
  $RUN 03_eval.py --model-dir "out/model-$tag" --prefix "eval-$tag-test"
  $RUN 03_eval.py --model-dir "out/model-$tag" --split test_unanimous \
    --prefix "eval-$tag-unan"
}

run b3-all
run b3-agree60 --agreement-min 0.6
run b3-weighted --weight-by-agreement
run c3-afrihate --extra-data out/afrihate_swa.parquet
run c4-combo --agreement-min 0.6 --extra-data out/afrihate_swa.parquet

echo "=== batch done ==="
grep -H macro_f1 out/eval-*-*_metrics.json || true

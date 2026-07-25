#!/bin/bash
# Round 3 (2026-07-24): Opus v4 partial 1,500-label mixed recipe vs shipped model.
#
# Comparators:
#   r3-baseline     - shipped d3-s1337, re-eval only
#   r3-opus-s1337/8/9 - DAPT + 2013 base + AfriHate + Opus train2026 extras
#
# Eval sets:
#   challenge_opus-v4  - prior challenge IDs with Opus labels (primary 2026 signal)
#   test_unanimous     - 2013 regression
#   gold               - old Gemini gold labels (prevalence-style regression only;
#                        no Opus gold coverage in the partial set)
set -e
cd "$(dirname "$0")"

RUN="uv run"
command -v uv >/dev/null 2>&1 || RUN="python"

evalset() {
  tag=$1; dir=$2
  for s in challenge_opus-v4 test_unanimous gold; do
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

# control: existing shipped model
evalset r3-baseline out/model-d3-s1337

COMMON="--model out/dapt-afro-xlmr --grad-accum 2 --warmup-ratio 0.06 --focal-gamma 2.0 --val-split val2026_opus-v4 --agreement-min 0.6 --extra-data out/afrihate_swa.parquet,out/train2026_opus-v4.parquet --extra-repeat 1,5 --epochs 5 --lr 1e-5"

train_variant r3-opus-s1337 $COMMON --seed 1337
train_variant r3-opus-s1338 $COMMON --seed 1338
train_variant r3-opus-s1339 $COMMON --seed 1339

echo "=== round 3 opus done ==="
python - <<'PY'
import json
from pathlib import Path
rows = []
for path in sorted(Path("out").glob("eval-r3-*_metrics.json")):
    metrics = json.loads(path.read_text())
    rows.append((path.name, metrics.get("macro_f1"), metrics.get("hate_f1")))
for name, macro, hate in rows:
    print(f"{name}: macro_f1={macro} hate_f1={hate}")
PY

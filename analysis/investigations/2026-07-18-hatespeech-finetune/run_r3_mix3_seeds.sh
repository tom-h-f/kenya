#!/bin/bash
# 3-seed confirm for Pareto mix×3 (AfriHate×1, Opus×3).
# Seed 1337 already trained as r3-mix-r3-s1337; alias then train 1338/1339.
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

COMMON="--model out/dapt-afro-xlmr --grad-accum 2 --warmup-ratio 0.06 --focal-gamma 2.0 --val-split val2026_opus-v4 --agreement-min 0.6 --extra-data out/afrihate_swa.parquet,out/train2026_opus-v4.parquet --extra-repeat 1,3 --epochs 5 --lr 1e-5"

# Reuse seed-1337 artifacts from the mix sweep under the mix3 tag.
if [ -f out/eval-r3-mix-r3-s1337-test_unanimous_metrics.json ]; then
  echo "=== alias r3-mix-r3-s1337 -> r3-mix3-s1337 ==="
  for s in challenge_opus-v4 test_unanimous gold; do
    for ext in metrics.json errors.csv hate_thresholds.csv confusion.png; do
      src="out/eval-r3-mix-r3-s1337-${s}_$ext"
      dst="out/eval-r3-mix3-s1337-${s}_$ext"
      if [ -f "$src" ] && [ ! -f "$dst" ]; then
        cp -a "$src" "$dst"
      fi
    done
  done
  if [ -d out/model-r3-mix-r3-s1337 ] && [ ! -e out/model-r3-mix3-s1337 ]; then
    ln -sfn model-r3-mix-r3-s1337 out/model-r3-mix3-s1337
  fi
fi

train_variant r3-mix3-s1337 $COMMON --seed 1337
train_variant r3-mix3-s1338 $COMMON --seed 1338
train_variant r3-mix3-s1339 $COMMON --seed 1339

echo "=== mix×3 3-seed summary ==="
$RUN python - <<'PY'
import json
import statistics as st
from pathlib import Path

def load(tag, split):
    p = Path(f"out/eval-{tag}-{split}_metrics.json")
    return json.loads(p.read_text())["macro_f1"] if p.exists() else None

for split in ("challenge_opus-v4", "test_unanimous", "gold"):
    vals = [load(f"r3-mix3-s{s}", split) for s in (1337, 1338, 1339)]
    vals = [v for v in vals if v is not None]
    base = load("r3-baseline", split)
    if not vals:
        print(f"{split}: missing")
        continue
    m = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    delta = (m - base) if base is not None else None
    print(f"{split}: mean={m:.4f} sd={sd:.4f} n={len(vals)} baseline={base} delta={delta}")
PY
echo "=== mix×3 3-seed done ==="

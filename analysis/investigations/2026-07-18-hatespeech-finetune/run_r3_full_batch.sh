#!/bin/bash
# Round-3 full Opus v4 (2,440 labels): same Pareto recipe as mix×3 (AfriHate×1, Opus×3).
# Compare to promoted r3-mix3 (1,500-label) and shipped d3-s1337.
set -e
cd "$(dirname "$0")"
RUN="uv run"
command -v uv >/dev/null 2>&1 || RUN="python"

evalset() {
  tag=$1; dir=$2
  for s in challenge_opus-v4-full gold_opus-v4-full test_unanimous; do
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

# Re-eval promoted + shipped on the new full splits for fair compare.
for tag_dir in "r3-full-baseline:out/model-d3-s1337" "r3-full-mix3:out/model-r3-mix3-s1337"; do
  tag="${tag_dir%%:*}"
  dir="${tag_dir#*:}"
  if [ -d "$dir" ]; then
    evalset "$tag" "$dir"
  else
    echo "=== missing $dir; skip $tag ==="
  fi
done

COMMON="--model out/dapt-afro-xlmr --grad-accum 2 --warmup-ratio 0.06 --focal-gamma 2.0 --val-split val2026_opus-v4-full --agreement-min 0.6 --extra-data out/afrihate_swa.parquet,out/train2026_opus-v4-full.parquet --extra-repeat 1,3 --epochs 5 --lr 1e-5"

train_variant r3-full-s1337 $COMMON --seed 1337
train_variant r3-full-s1338 $COMMON --seed 1338
train_variant r3-full-s1339 $COMMON --seed 1339

echo "=== full Opus v4 summary ==="
$RUN python - <<'PY'
import json
import statistics as st
from pathlib import Path

def load(tag, split):
    p = Path(f"out/eval-{tag}-{split}_metrics.json")
    return json.loads(p.read_text())["macro_f1"] if p.exists() else None

splits = ("challenge_opus-v4-full", "gold_opus-v4-full", "test_unanimous")
tags = {
    "shipped-d3": "r3-full-baseline",
    "promoted-mix3-1500": "r3-full-mix3",
    "full-s1337": "r3-full-s1337",
    "full-s1338": "r3-full-s1338",
    "full-s1339": "r3-full-s1339",
}
print(f"{'tag':22s} {'challenge':>10} {'gold':>10} {'unan':>10}")
for name, tag in tags.items():
    vals = [load(tag, s) for s in splits]
    fmt = [f"{v:.4f}" if v is not None else "  -" for v in vals]
    print(f"{name:22s} {fmt[0]:>10} {fmt[1]:>10} {fmt[2]:>10}")

full_tags = ["r3-full-s1337", "r3-full-s1338", "r3-full-s1339"]
print("\nfull 3-seed mean±sd:")
for split in splits:
    vals = [load(t, split) for t in full_tags]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        continue
    base = load("r3-full-baseline", split)
    mix = load("r3-full-mix3", split)
    m, sd = st.mean(vals), st.stdev(vals)
    print(f"  {split}: {m:.4f}±{sd:.4f}  vs shipped={(m-base) if base else None}  vs mix3={(m-mix) if mix else None}")
PY
echo "=== full Opus v4 done ==="

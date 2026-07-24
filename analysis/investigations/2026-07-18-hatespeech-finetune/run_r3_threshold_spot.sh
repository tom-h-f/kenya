#!/bin/bash
# Full hate-threshold sweeps + 14 coded spotcheck for baseline and r3 Opus seeds.
set -e
cd "$(dirname "$0")"
RUN="uv run"
command -v uv >/dev/null 2>&1 || RUN="python"

# Prefer a train-schema parquet for load_split compatibility.
if [ ! -f out/spotcheck_coded14.parquet ]; then
  $RUN python - <<'PY'
import pandas as pd
from pathlib import Path
src = Path("out/spotcheck_coded14_opus.csv")
df = pd.read_parquet(src)
out = df.rename(columns={"label_id": "label"})[["post_id", "text", "label"]].copy()
if "agreement" not in out.columns:
    out["agreement"] = 1.0
out.to_parquet("out/spotcheck_coded14.parquet", index=False)
print("wrote out/spotcheck_coded14.parquet", len(out))
PY
fi

sweep() {
  tag=$1; dir=$2; split=$3
  prefix="sweep-$tag-$split"
  if [ -f "out/${prefix}_full_sweep.csv" ]; then
    echo "=== skip sweep $tag/$split ==="
    return
  fi
  echo "=== sweep $tag / $split ==="
  $RUN 11_hate_sweep.py --model-dir "$dir" --split "$split" --prefix "$prefix"
}

spot() {
  tag=$1; dir=$2
  out="out/spotcheck_coded14_$tag.csv"
  if [ -f "$out" ]; then
    echo "=== skip spotcheck $tag ==="
    return
  fi
  echo "=== spotcheck $tag ==="
  $RUN 04_infer.py --csv out/spotcheck_coded14_opus.csv --column text \
    --model-dir "$dir" --out "$out"
}

for split in val2026_opus-v4 challenge_opus-v4 test_unanimous; do
  sweep r3-baseline out/model-d3-s1337 "$split"
done
spot r3-baseline out/model-d3-s1337

for seed in 1337 1338 1339; do
  tag="r3-opus-s$seed"
  dir="out/model-$tag"
  for split in val2026_opus-v4 challenge_opus-v4 test_unanimous; do
    sweep "$tag" "$dir" "$split"
  done
  spot "$tag" "$dir"
done

echo "=== summarize operating points ==="
$RUN python - <<'PY'
from pathlib import Path
import pandas as pd

rows = []
for path in sorted(Path("out").glob("sweep-*_full_sweep.csv")):
    tag_split = path.name.removeprefix("sweep-").removesuffix("_full_sweep.csv")
    # tag may contain hyphens; split is last underscore-joined known name
    for split in ("val2026_opus-v4", "challenge_opus-v4", "test_unanimous"):
        if tag_split.endswith(split):
            tag = tag_split[: -(len(split) + 1)]
            break
    else:
        continue
    df = pd.read_csv(path)
    if df.empty:
        continue
    df = df.copy()
    df["f1"] = df.apply(
        lambda r: (2 * r.precision * r.recall / (r.precision + r.recall))
        if (r.precision + r.recall)
        else 0.0,
        axis=1,
    )
    best_f1 = df.loc[df["f1"].idxmax()]
    gate = df[df["recall"] >= 0.80]
    best_p = gate.loc[gate["precision"].idxmax()] if len(gate) else None
    # prevalence-ish: ~1.4% of N — approximate using flagged count if N known later
    rows.append({
        "tag": tag,
        "split": split,
        "best_f1_thr": float(best_f1.threshold),
        "best_f1": float(best_f1.f1),
        "best_f1_P": float(best_f1.precision),
        "best_f1_R": float(best_f1.recall),
        "best_f1_flagged": int(best_f1.flagged),
        "R80_thr": float(best_p.threshold) if best_p is not None else None,
        "R80_P": float(best_p.precision) if best_p is not None else None,
        "R80_flagged": int(best_p.flagged) if best_p is not None else None,
    })

summary = pd.DataFrame(rows)
summary.to_csv("out/r3_threshold_summary.csv", index=False)
print(summary.to_string(index=False))

# spotcheck p_hate means
print("\n=== coded-14 mean p_hate ===")
for path in sorted(Path("out").glob("spotcheck_coded14_*.csv")):
    if path.name.endswith("_opus.parquet"):
        continue
    df = pd.read_csv(path)
    if "p_hate" not in df.columns:
        print(path.name, "no p_hate")
        continue
    print(f"{path.name}: mean_p_hate={df.p_hate.mean():.4f} "
          f"argmax_hate={(df.label_id==2).sum() if 'label_id' in df else '?'} "
          f"n={len(df)}")
PY

echo "=== threshold + spotcheck done ==="

# Opus v4 Partial Relabel Training Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the available 1,500 Opus v4 labels into round-3 train/eval assets and measure whether they beat the shipped classifier under the settled mixed recipe.

**Architecture:** Merge automated + manual Opus labels into one incomplete but usable 2026 dataset. Keep prior split membership where possible. Train DAPT + 2013 base mixed with Opus 2026 extras (not Opus-only). Compare against shipped `d3-s1337` on challenge, 2013 regression, and any available gold overlap.

**Tech Stack:** Python/pandas/uv, existing `02_train.py` / `03_eval.py`, Modal A100 for full runs.

**Inventory (measured 2026-07-24):**
- Automated chunks: 1,100
- Manual outbox batches 000-003: 400
- Union: **1,500** unique IDs (no overlap)
- Classes: neither 676 / offensive 558 / hate 266
- Flags: ethnic_targeting 266 / coded 58 / dehumanisation 43 / violence 44
- Prior-split coverage inside the 1,500: train 1,305 / challenge **195 (all)** / gold **0**

**Decision:** 1,500 is enough to train. Do not wait for the remaining 940. Do not train flag heads yet (support still thin).

---

### Task 1: Freeze and merge the 1,500-label set

**Files:**
- Modify or extend: `22_import_manual_labels.py` (or add `23_opus_partial_merge.py`)
- Generate: `out/labels_2026_opus-v4-partial.parquet`
- Generate: `out/21_opus_partial_report.json`

**Step 1: Write failing tests for partial merge**

Require:
- union of `claude-opus-code/chunk_*.jsonl` + `opus_v4_full_manual/outbox/*.jsonl`
- reject duplicate conflicting labels for the same `post_id`
- preserve batch text/stratum by ID
- do **not** require all 2,440 rows or complete 25-row chunks
- emit single-labeller provenance: model `opus`, prompt v4, sources `automated|manual`

**Step 2: Implement merge + report**

Report must include class/flag/confidence counts, stratum coverage, prior-split coverage, and movement vs `labels_2026_full_final.parquet` on the 1,500 shared IDs.

**Step 3: Run and commit**

```bash
uv run 23_opus_partial_merge.py
uv run --with pytest pytest -q test_23_opus_partial_merge.py
```

Commit: `feat(analysis): merge Opus v4 partial labels`

---

### Task 2: Rebuild round-3 splits from the partial set

**Files:**
- Extend: `17_prep_round2.py` or add `24_prep_opus_v4.py`
- Generate: `out/{train2026,val2026,gold,challenge}_opus-v4.parquet`
- Generate: `out/24_opus_v4_splits.json`

**Step 1: Define split rules**

Reuse prior `split` from `labels_2026_full_final.parquet` for every labelled ID:
- all 195 challenge IDs in the 1,500 → `challenge_opus-v4`
- train IDs → carve `val2026_opus-v4` (default 250 if train allows stratification, else 20%)
- remainder → `train2026_opus-v4`
- gold: **empty or unavailable** for Opus labels; keep evaluating shipped/old gold only as a regression reference, and say so explicitly

Shape columns exactly as now: `post_id, text, label(int), agreement=1.0`.

**Step 2: Tests**

- no ID leakage across train/val/challenge
- labels map through `LABEL2ID`
- challenge n == 195
- train+val == 1,305

**Step 3: Commit**

`feat(analysis): prep Opus v4 train splits`

---

### Task 3: Local smoke train (sub-minute)

**Files:** none committed; local `out/model-r3-opus-smoke/`

**Step 1: Tiny smoke**

```bash
uv run 02_train.py --tag r3-opus-smoke --sample 200 \
  --model out/dapt-afro-xlmr \
  --extra-data out/train2026_opus-v4.parquet \
  --extra-repeat 5 \
  --no-class-weights \
  --val-split val2026_opus-v4 \
  --epochs 1 --batch-size 8
```

If DAPT weights are only on Modal/main checkout, point `--model` at the existing local/HF path used previously.

**Step 2: Stop if smoke fails**

Fix data/schema issues before any Modal spend.

Commit only code/test fixes if needed.

---

### Task 4: Full mixed recipe on Modal (3 seeds)

**Files:**
- Create: `run_r3_opus_batch.sh`
- Generate: `out/model-r3-opus-s{1337,1338,1339}/`
- Generate: eval metrics JSONs

**Recipe (settled; do not relitigate):**
- Start from DAPT afro-xlmr
- Mix 2013 base + AfriHate if previously used + Opus `train2026_opus-v4`
- Oversample 2026 extras (`--extra-repeat 5` matching round 2)
- Plain CE / no class weights (focal optional only to match shipped comparator; prefer `--no-class-weights` without claiming focal gains)
- Val on `val2026_opus-v4`
- Three seeds

**Comparators:**
1. `r2-baseline` / shipped `d3-s1337` re-eval
2. `r3-opus-mixed` three seeds

**Eval sets:**
- `challenge_opus-v4` (primary 2026 coded/hard set; labels are Opus, so this measures fit to Opus taxonomy)
- `test_unanimous` / 2013 regression
- old `gold` only as prevalence-style regression against **old** labels or model behaviour — do not call it Opus gold

Also score the 14 known coded posts if that helper still exists.

**Step: launch**

```bash
uv run modal run --detach modal_train.py --cmd "bash run_r3_opus_batch.sh" --spawn
```

Commit script before launch: `feat(analysis): add Opus v4 round-3 train recipe`

---

### Task 5: Decide and update STATE

**Files:**
- Modify: `STATE.md`, `findings-plan-a.md` or a short `findings-opus-v4.md`

Record:
- exact label count and sources (1,100 automated + 400 manual)
- prompt hash / model (`claude` / `opus`)
- single-labeller caveat
- no gold coverage in this partial set
- seed macro-F1 mean±sd vs shipped 0.688±0.021
- challenge and 2013 deltas
- go/no-go on replacing production weights
- explicitly defer remaining 940 labels and flag heads

Commit: `docs(analysis): record Opus v4 train verdict`

---

## Out of scope

- Finishing the other 940 posts before training
- Dual-labeller adjudication
- Flag-head training
- Prompt v5
- Treating the 120-row human calibration as an independent Opus accuracy gate (you already accepted Opus there)

## Stop-gates

1. After Task 1: confirm 1,500 unique clean rows and sane class movement.
2. After Task 3: smoke must train/eval without schema errors.
3. After Task 4: only promote if mixed three-seed result beats shipped by more than seed noise (~2.1pt unan macro-F1) **or** shows clear challenge/coded gain without 2013 collapse.

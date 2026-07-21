# Opus v4 Remaining Work

> Lean plan. No subagent review loops. Code first where needed, then paid Opus
> only behind stop-gates.

**Goal:** Finish the report provenance fix, then produce Opus v4 labels for the
2,440-post batch under an isolated tag.

**Already done:** prompt `label_v4.md`, runner validation/provenance, Opus
model ID `claude-opus-4-6-thinking`, reporting CLI (`21_opus_relabel.py`).

**Not done:** any Opus v4 labelling run.

**Working directory:**
`.worktrees/opus-v4-relabel/analysis/investigations/2026-07-18-hatespeech-finetune`

**Source batch (main checkout, ignored artifacts):**
`/Users/tom/Code/Misc/kenya-monitor-2027/analysis/investigations/2026-07-18-hatespeech-finetune/out/labels/full/batch.parquet`

**Baseline labels:**
`.../out/labels_2026_full_final.parquet`

**Labeller:** `claude-opus-4.6` → agy `claude-opus-4-6-thinking`

---

## Task R1: Close the report gap (no Opus spend)

**Files:** `21_opus_relabel.py`, `test_21_opus_relabel.py`

1. `compare-full` must verify baseline `post_id`, `text`, and `stratum` match
   the run's `batch.parquet` before comparing labels.
2. Calibration driver parquet must contain only `post_id`, `text`, `stratum`.
   Human labels stay in the separate calibration CSV used only at score time.
3. Tests for both; run `uv run --with pytest pytest -q test_21_opus_relabel.py`.
4. Commit: `fix(analysis): align comparison provenance`

Stop if tests fail.

---

## Task R2: Calibration (cheap Opus spend, ~120 posts)

1. Build input:
   ```bash
   uv run 21_opus_relabel.py make-calibration \
     --source out/blind_check_coded_calibration.csv
   ```
2. Label:
   ```bash
   uv run 13_label_drive.py --input opus_v4_calibration.parquet \
     --tag opus-v4-calibration --prompt-version v4 \
     --labellers claude-opus-4.6 --concurrency 2
   ```
3. Score:
   ```bash
   uv run 21_opus_relabel.py score-calibration \
     --tag opus-v4-calibration --labeller claude-opus-4.6
   ```
4. Human inspect mismatches involving hate, aliases (`Kasongo`/`Zakayo`),
   quotation, and condemnation. Fix prompt only if errors are rule-shaped;
   if prompt changes, new tag/revision and rerun R2.

**Stop-gate:** do not start R3 until calibration looks sane. No hard numeric
promotion gate here; this is a regression check against the 120-row human set.

Commit reports: `test(analysis): score Opus v4 calibration`

---

## Task R3: Stratified pilot (medium spend, ~100 posts)

```bash
SOURCE=/Users/tom/Code/Misc/kenya-monitor-2027/analysis/investigations/2026-07-18-hatespeech-finetune/out/labels/full/batch.parquet

uv run 13_label_drive.py --input "$SOURCE" --pilot 100 \
  --tag opus-v4-pilot --prompt-version v4 \
  --labellers claude-opus-4.6 --concurrency 2

uv run 14_label_merge.py --tag opus-v4-pilot --prompt-version v4 \
  --labellers claude-opus-4.6 --blind-check 30
```

Inspect: parked chunks (must be 0), class/flag distribution, blind sheet,
whether aliases and ethnic targeting are being confused.

**Stop-gate:** explicit yes before R4. Full run is ~25x this spend.

Commit: `test(analysis): validate Opus v4 pilot`

---

## Task R4: Full 2,440 relabel (expensive)

Only after R3 approval.

```bash
SOURCE=/Users/tom/Code/Misc/kenya-monitor-2027/analysis/investigations/2026-07-18-hatespeech-finetune/out/labels/full/batch.parquet
BASELINE=/Users/tom/Code/Misc/kenya-monitor-2027/analysis/investigations/2026-07-18-hatespeech-finetune/out/labels_2026_full_final.parquet

uv run 13_label_drive.py --input "$SOURCE" \
  --tag opus-v4-full --prompt-version v4 \
  --labellers claude-opus-4.6 --concurrency 2

# resume with the same command if interrupted

uv run 14_label_merge.py --tag opus-v4-full --prompt-version v4 \
  --labellers claude-opus-4.6 --blind-check 100

uv run 21_opus_relabel.py compare-full --tag opus-v4-full \
  --labeller claude-opus-4.6 --baseline "$BASELINE"
```

Expect: 98 chunks, 2,440 unique IDs, zero parked, single-labeller caveat in
merge report, movement JSON written.

Update `STATE.md` and `findings-plan-a.md` with model ID, prompt hash,
calibration summary, class/flag counts, movement, and reliability caveat.
Do not claim independent human gold.

Commit: `feat(analysis): relabel corpus with Opus v4`

---

## Rules

- Never overwrite tags `full`, `heldout-v3`, or prior Opus heldout outputs.
- One agent in this session does the work; no nested implementer/reviewer loops.
- Do not start R4 without an explicit go-ahead after R3.
- If spend or quality looks wrong at R2/R3, stop and reassess rather than
  "finishing the plan."

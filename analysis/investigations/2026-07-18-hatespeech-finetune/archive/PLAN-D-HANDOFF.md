# Plan D handoff: afro-xlmr-large Kenya hate-speech classifier

Self-contained brief for the executing agent. Read fully before coding.
Companion docs in this dir: `findings.md` (results so far), `plans-round2.md`
(the four workstreams - this file supersedes its Plan D section for
execution), `README.md` (dataset + run basics).

## Mission

Train the best-possible hate/offensive/neither classifier for tweets by
Kenyan citizens on politics/society/culture, using `Davlan/afro-xlmr-large`,
runnable on Google Colab (free T4 today; A100 path noted). Beat the current
best (c4-combo, twitter-xlm-roberta-base) by the promotion gates below, then
package for deployment.

## Context you must know

- Project: kenya-monitor-2027 - X/Twitter monitor for 2027 Kenyan election;
  ~109k collected posts in R2 (DuckDB access via `kma.db`, package at
  `analysis/src/kma/`). Current incitement triage = lexicon + zero-shot NLI
  (`kma/incitement.py`) - weak on coded speech; this classifier is its
  planned upgrade/complement.
- All work so far lives in THIS investigation dir, inside git worktree
  `hatespeech-finetune` (branch `worktree-hatespeech-finetune`). Scripts are
  standalone PEP 723 (`uv run <script>`); also run under plain `python` on
  Colab (Colab ships uv too - both work).
- **Current best (2026-07-19, the number to beat)**: `c4-combo` =
  twitter-xlm-roberta-base, 4 epochs, batch 64, class-weighted CE, on
  "agree60 + AfriHate" data. Unanimous-test macro-F1 **0.5972**, hate
  P 0.36 / R 0.81. Full-test macro-F1 0.5580.
- **Canonical data recipe (locked by ablation - do not relitigate)**:
  - our 48k (`out/train.parquet`) filtered to `agreement >= 0.6`
  - PLUS AfriHate Swahili (`out/afrihate_swa.parquet`, 18,266 rows)
  - class-weighted CE; NO agreement sample-weighting (measured worse)
- **Eval protocol (fixed - never train on these)**:
  - `out/test.parquet` (4,666 rows, includes noisy labels - secondary)
  - `out/test_unanimous.parquet` (2,264 rows - HEADLINE metric)
  - `out/val.parquet` for epoch selection (metric: macro-F1)
  - Expect the winner to trail on full test while winning on unanimous -
    that pattern is fine (see findings.md B3/C3 verdict).
- Splits are v2 (langid-cleaned, near-dup clusters forced into train). Never
  re-run `00_prep.py` with different settings mid-experiment - it would
  invalidate all comparisons.
- Class mapping everywhere: 0=neither, 1=offensive, 2=hate. (NOT the
  Davidson paper order.)

## Environment map

- This box (tac2): M4 Pro 24GB, MPS. Fine for smoke tests (`02_train.py`
  sample mode ~40s). Do NOT schedule multi-hour MPS runs - overnight
  processes here got killed repeatedly (cause unknown, likely sleep).
- Colab: Tom has FREE tier only (T4 16GB). A100 sections below are
  conditional on future upgrade.
- Google Drive syncs locally at `~/Drive/`; Colab sees it as
  `/content/drive/MyDrive/`. Existing working dir on both:
  `~/Drive/Colab/hatespeech-finetune/` == `MyDrive/Colab/hatespeech-finetune/`
  (contains scripts + out/ with all parquets and prior run artifacts).
  Hand files to/from Colab by copying to `~/Drive/Colab/` - no manual upload.
- `HF_TOKEN` env var on this box has Tom's HuggingFace token (AfriHate terms
  accepted). HF hub available for pushing models (private repo).
- Existing Colab pattern to copy: `colab_b3_batch.ipynb` + `run_b3_batch.sh`
  (idempotent: per-epoch checkpoints on Drive, auto `--resume`, skip-if-
  evaluated). Keep this property for every long run you author.

## Colab gotchas (all bitten already - respect them)

1. `drive.mount` fails if OAuth consent checkboxes not all ticked.
2. Colab exports `MPLBACKEND=module://matplotlib_inline...` - pop it from
   `os.environ` before importing matplotlib in any uv-run script
   (see `03_eval.py` header).
3. Free T4 disconnects on idle: every run must checkpoint to Drive each
   epoch and resume cleanly. Re-running the batch cell must be safe.
4. Do not resume a checkpoint across a batch-size/schedule change - wipe the
   variant's `checkpoints/` dir instead.
5. fp16 on CUDA (T4 has no bf16), fp32 on MPS. `02_train.py` handles this.

## Deliverables (in order)

### D0. Corpus export for DAPT
Script `08_export_corpus.py` (run on tac2 - needs R2 access via analysis
env): pull all posts' text via `kma.db.posts_source('x')` (dedupe by
platform_post_id, keep latest), light clean (strip URLs, keep @mentions
as-is, drop <10-char), dedupe exact + near-dup (reuse `00_prep.py` MinHash
helpers), write `out/dapt_corpus.parquet` (~100k texts) and copy to
`~/Drive/Colab/hatespeech-finetune/out/`. Record row count in the log.

### D1. DAPT - domain-adaptive pretraining
Script `09_dapt.py` (standalone, Colab-first):
- Continue MLM on `dapt_corpus.parquet` from `Davlan/afro-xlmr-large`
  (fallback if T4 16GB OOMs even at batch 8 + grad-accum: run DAPT on
  `afro-xlmr-base` and do the whole plan at base size - a working base-size
  DAPT model beats an untrained large one).
- mlm_probability 0.15, seq 128, effective batch 64 via grad accumulation,
  lr 5e-5, 2 epochs, fp16, per-epoch (or per-N-steps) Drive checkpoints +
  resume.
- T4 timing: MEASURE 100 steps first, print projection; if projected > ~8h,
  cut to 1 epoch (still worthwhile) or wait for A100.
- Gate: held-out MLM loss (hold out 2% of corpus) clearly below the stock
  model's. Save as `out/dapt-afro-xlmr/` + push to HF private repo.

### D2. Fine-tune on canonical recipe
Extend `02_train.py` (already has --tag/--agreement-min/--extra-data/
--resume/--batch-size) with:
- `--base MODEL_OR_PATH` already exists as `--model` - use it to point at
  the DAPT output.
- label smoothing flag (`--label-smoothing 0.05`), focal-loss flag
  (`--focal-gamma 2.0`, replaces class weights when set),
  layer-wise LR decay (`--llrd 0.9`).
- T4 fitting for large (560M): batch 8 x grad-accum 8 (effective 64),
  fp16, max_len 128, lr 1e-5, warmup 6%, 5 epochs, early stop patience 2.
Ablation ladder - ONE change at a time, in a `run_d_batch.sh` mirroring
`run_b3_batch.sh` (idempotent, eval both test sets per variant):
1. `d-base`: afro-xlmr-large stock + canonical recipe (no DAPT) - anchor
2. `d-dapt`: DAPT model + canonical recipe
3. `d-dapt-ls`: + label smoothing 0.05
4. `d-dapt-focal`: focal loss instead of class weights
5. `d-dapt-llrd`: + layer-wise decay (on best of 3/4)
Keep winner; if DAPT does not beat stock, say so loudly and continue with
stock (do not silently proceed).

### D3. Multi-seed confirmation
Best variant x 3 seeds. Report mean ± sd for unan macro-F1 and hate-F1.
Ship the median-val seed's checkpoint.

### D4. Evaluation + report
- `03_eval.py` (has --model-dir/--split/--prefix) on both test sets.
- Spot-check 2026 transfer exactly as round 1: run `04_infer.py` on
  `../2026-07-17-manipulation-sweep/out/10_flagged.csv` - compare with
  round-1 result (9/14 offensive, 0 hate, p_hate~0.01). Any movement of
  coded posts toward hate = the headline finding; no movement = expected
  until Plan A labels land, also report.
- Update `findings.md` with a Plan D section: table of all variants, both
  tests, threshold sweeps for the winner.
- **Promotion gates** (vs c4-combo 0.5972 unan / 0.5580 full):
  - unan macro-F1 >= 0.62 (i.e. > +2pt) AND
  - unan hate recall >= 0.80 held while precision not worse than 0.36 AND
  - no collapse on full test (>= 0.55)
  Below gates: still write everything up; the model does NOT replace
  c4-combo.

### D5. Package
- Push winner to private HF hub (`kenya-hatespeech-afroxlmr` or similar,
  token in HF_TOKEN); record repo id in findings.md.
- Copy final model dir to `~/Drive/Colab/hatespeech-finetune/out/model-d/`.
- Do NOT wire into kma pipeline (separate task, needs the D6 discussion in
  plans-round2.md - tf1 is a 3.8GB-RAM CPU box, quantisation decision
  pending).

## Explicitly out of scope for this handoff

- Plan A (LLM labelling) - runs separately; when its labels land, a later
  round adds them + the multi-task flag head. Do not block on it.
- Coded-term challenge set - needs Plan A/human input.
- kma/enrich integration, corpus-wide scoring, R2 persistence.

## Working agreements

- Sub-minute smoke run before every long run (sample mode exists).
- Measure before scaling: print per-step timing + projection, abort if
  projection insane. Never assume T4 numbers.
- Every long-running Colab cell idempotent (resume + skip-finished).
- Update `findings.md` as results land, not at the end. Numbers without a
  seed/dataset/split label are worthless - always annotate.
- Keep caveman comms style with Tom; commit only when he asks.
- If MPS/local training tempts you: don't (see Environment map).

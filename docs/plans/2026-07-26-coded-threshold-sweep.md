# Coded-suspect threshold sweep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Retune `kma.measure.coded_suspect` NLI / `fp_risk` gates on a **fixed banked eval set** (no new labels) to raise recall on true coded menace without re-admitting political-slogan FPs like `Fukuza Ruto`.

**Architecture:** Build a frozen eval table from banked challenge hard-misses, coded-14 spotcheck, and committed measure fixtures. Grid-search menace thresholds and political-criticism margins per `fp_risk` tier. Pick the Pareto config that maximizes coded recall on positives while keeping FP rate on known negatives at or below baseline. Write chosen constants into `kma.measure` and lock them with regression tests.

**Tech Stack:** Python, pandas, `kma.measure`, `kma.incitement.scan_text`, pytest; optional DuckDB only to pull live NLI for banked post IDs

**Prerequisite:** `kma.measure` on the implementation branch (`feat/hate-measurement` or merged). Plan 1 (persist) is independent but should use the **post-sweep** constants when refreshing.

**Constraints:** No new human/Opus labels. No 3-class retrain. Eval labels come only from already-banked artifacts.

---

### Task 1: Assemble frozen eval pack

**Files:**
- Create: `analysis/investigations/2026-07-18-hatespeech-finetune/25_coded_eval_pack.py`
- Create: `analysis/investigations/2026-07-18-hatespeech-finetune/out/coded_eval_pack.parquet` (or csv)
- Reference inputs (read-only):
  - `out/r3_hard_misses_challenge.csv`
  - `out/spotcheck_coded14_opus.csv` / `out/spotcheck_coded14_opus.parquet` if present
  - `analysis/tests/fixtures/measure_fixtures.csv` (after measure branch merge)

**Step 1: Define label column `y_coded` (programmatic, documented)**

| source | rule for `y_coded` |
|---|---|
| Fixtures `coded_suspect` expected True | positive |
| Fixtures expected False | negative |
| Hard-miss rows with flags containing `coded_language` or true∈{hate} with ethnic coded rationale | positive (use existing `true` + `flags` / rationale heuristics documented in script) |
| Hard-miss offensive→neither with only mild insult, no coded flags | negative |
| Coded-14 Opus labels: treat `offensive`+coded flags / documented coded posts as positive; `neither` as negative | as labeled |

Script must print counts and refuse to invent labels.

**Step 2: For each row, ensure columns:** `post_id`, `text`, `y_coded`, `source`, plus NLI scores if available (join live `incitement/` by id; if missing, leave null and let `coded_suspect` behave as today).

**Step 3: Write pack; commit script + small committed CSV sample (not necessarily full parquet if gitignored under `out/`)**

```bash
cd analysis && uv run python investigations/2026-07-18-hatespeech-finetune/25_coded_eval_pack.py
git add investigations/2026-07-18-hatespeech-finetune/25_coded_eval_pack.py
# commit a slim fixtures-derived eval csv under tests/fixtures if out/ is gitignored
git commit -m "feat(measure): coded eval pack builder"
```

---

### Task 2: Threshold sweep script

**Files:**
- Create: `analysis/investigations/2026-07-18-hatespeech-finetune/26_coded_threshold_sweep.py`
- Create: `out/coded_threshold_sweep.csv` (local artifact)

**Step 1: Parameter grid**

For each `fp_risk` in `{low, medium, high}`:

- `menace_min`: e.g. low `[0.35, 0.40, 0.45, 0.50, 0.55]`; medium `[0.45..0.65]`; high `[0.55..0.75]`
- `pol_margin`: e.g. `[-0.10, -0.05, 0.0, 0.05, 0.10, 0.15]` (high tier prefers ≥0.10)

Reuse current defaults as baseline row.

**Step 2: For each config, evaluate**

Metrics on the frozen pack:

- recall = TP / (TP+FN) on `y_coded==True`
- precision = TP / (TP+FP)
- FP count on known negatives (especially fixture `kenya_fukuza_ruto`, `sports_nyoka`)
- Hard constraint: `kenya_fukuza_ruto` and `sports_nyoka` must stay **negative** for a config to be admissible

**Step 3: Print Pareto table; write CSV**

**Step 4: Commit script**

```bash
git commit -m "feat(measure): coded threshold sweep script"
```

---

### Task 3: Choose and lock constants

**Files:**
- Modify: `analysis/src/kma/measure.py` (`_CODED_MENACE_MIN`, `_CODED_POL_MARGIN`)
- Modify: `analysis/tests/test_measure.py` / `tests/fixtures/measure_fixtures.csv` if expectations shift
- Modify: investigation `STATE.md` or a short `out/coded_threshold_choice.md` note

**Step 1: Selection rule (fixed)**

Among admissible configs (Fukuza + sports nyoka still False):

1. Maximize recall on positives.
2. Break ties by higher precision.
3. Break further ties by closer to current defaults (smaller L1 on thresholds).

**Step 2: Patch `measure.py` constants; update unit tests/fixtures to match**

**Step 3:**

```bash
cd analysis && uv run pytest tests/test_measure.py tests/test_measure_fixtures.py -q
```

**Step 4: Commit**

```bash
git commit -m "feat(measure): lock coded_suspect thresholds from sweep"
```

---

### Task 4: Re-validate on live lexicon slice (read-only)

**Step 1:** Run attach_measurement_columns on live corpus (or post-plan-1 hatespeech columns). Record:

- `coded_suspect` count
- top terms among coded
- confirm high-engagement `Fukuza Ruto` class remains mostly not coded

**Step 2:** Write numbers into `out/coded_threshold_choice.md` (may be gitignored; paste summary into commit message body only if needed, else keep local).

**Step 3:** If plan 1 already shipped, re-run `python -m kma.hatespeech --refresh-measure` so persisted columns pick up new gates.

---

### Done when

- Sweep script + eval pack builder are reproducible.
- New thresholds are in `measure.py` with tests green.
- Admissible constraint holds: political `fukuza` + sports `nyoka` fixtures stay non-coded.
- Live coded count is reported (order-of-magnitude check vs pre-sweep).

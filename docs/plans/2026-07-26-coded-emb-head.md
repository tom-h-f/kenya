# Weakly-supervised coded head (embeddings) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train a small binary coded-speech classifier on **existing multilingual embeddings** using only programmatic weak labels (lexicon ∩ NLI vs clean neither), producing a `p_coded` score for automated measurement alongside regex∩NLI `coded_suspect`.

**Architecture:** Mine weak positives/negatives from the live corpus with deterministic rules (no new human/Opus labels). Fit a sklearn model (logistic regression or HistGradientBoosting) on frozen embedding vectors from R2 `embeddings/`. Persist the model artifact under `analysis/` (joblib) and optionally write `p_coded` / `coded_model_flag` onto `hatespeech/` rows during refresh. Eval **only** on banked packs from plan 2 / fixtures / coded-14 / hard-misses.

**Tech Stack:** Python, sklearn, joblib, DuckDB/R2 embeddings + posts + incitement, `kma.measure` for weak-label rules, pytest

**Prerequisite:** `kma.measure` available. Prefer completing plan 2 (threshold lock) first so weak-positive mining uses the same NLI gates. Plan 1 persist can add `p_coded` columns in a follow-up refresh.

**Constraints:** No new labels. Do not fine-tune afro-xlmr in this plan. Do not overwrite 3-class `label` / `hate_flag` semantics.

---

### Task 1: Weak-label mining function

**Files:**
- Create: `analysis/src/kma/coded_model.py`
- Test: `analysis/tests/test_coded_model.py`

**Step 1: Failing tests for label rules**

```python
from kma.coded_model import weak_label

def test_weak_positive_madoadoa_high_nli():
    assert weak_label(
        text="kutoa madoadoa Kenya. Tuwe safi kama pamba",
        dehumanisation=0.9, violence_call=0.8, othering=0.5, political_criticism=0.3,
    ) == 1

def test_weak_negative_clean():
    assert weak_label(
        text="Lower the cost of living in Kenya before 2027.",
        dehumanisation=0.05, violence_call=0.05, othering=0.05, political_criticism=0.9,
    ) == 0

def test_weak_unlabeled_fukuza_borderline():
    # high-fp lexicon without clearing coded_suspect -> exclude from train (None)
    assert weak_label(
        text="Fukuza Ruto #Wantam",
        dehumanisation=0.2, violence_call=0.15, othering=0.1, political_criticism=0.85,
    ) is None
```

**Step 2: Implement `weak_label(...) -> 0 | 1 | None`**

- Positive (1): `measure.coded_suspect(...)` is True (post-plan-2 gates).
- Negative (0): no lexicon hits AND `p_neither`-style confidence unavailable here → use NLI: `max(menace) < 0.25` AND `political_criticism >= 0.5` AND `domain_bucket != offdomain` (or text has Kenya markers / ambiguous).
- Else `None` (excluded from training).

**Step 3: Tests pass; commit**

```bash
git commit -m "feat(coded_model): weak_label mining rules"
```

---

### Task 2: Build train matrix from R2

**Files:**
- Create: `analysis/investigations/2026-07-18-hatespeech-finetune/27_coded_emb_dataset.py`

**Step 1: Script loads** latest posts ⨝ embeddings ⟕ incitement; applies `weak_label`; downsamples negatives to a fixed ratio (e.g. 5:1 or 10:1 neg:pos); writes `out/coded_emb_train.parquet` with `platform_post_id`, `y`, `embedding`.

**Step 2: Hold out banked eval IDs** (fixtures, coded-14, hard-miss post ids) from training explicitly.

**Step 3: Print class counts; commit script**

```bash
git commit -m "feat(coded_model): embedding dataset builder"
```

---

### Task 3: Train sklearn model + banked eval

**Files:**
- Create: `analysis/investigations/2026-07-18-hatespeech-finetune/28_coded_emb_train.py`
- Create: `analysis/src/kma/artifacts/coded_emb_logreg.joblib` (or `analysis/models/` if preferred and not gitignored — if artifacts are large/gitignored, store on Modal volume / R2 and document path in STATE)

**Step 1: Train**

- `StandardScaler` + `LogisticRegression(max_iter=1000, class_weight="balanced")` as default (simple, calibrated-ish).
- Optional ablation: `HistGradientBoostingClassifier` — keep whichever wins banked macro/PR.

**Step 2: Eval on frozen pack (plan 2)**

Report AUROC / AP / recall@precision≥0.5 (or similar) on banked positives/negatives. Compare to regex∩NLI `coded_suspect` baseline on the same pack.

**Promote model only if:** banked AP ≥ baseline `coded_suspect` AP **or** recall@fixed FP budget beats baseline, and fixture Fukuza/sports stay below decision threshold.

**Step 3: Commit training script + small metrics JSON under `out/` or `docs` summary**

```bash
git commit -m "feat(coded_model): train embedding logreg"
```

---

### Task 4: Inference API in kma

**Files:**
- Modify: `analysis/src/kma/coded_model.py` (`predict_proba`, `load_model`)
- Test: load fixture embeddings or random unit vectors with a tiny fitted model in-test (fit on synthetic data in test to avoid shipping large binaries if needed)

**Step 1:** `predict_p_coded(embeddings: np.ndarray) -> np.ndarray`

**Step 2:** Decision threshold `CODED_MODEL_THRESHOLD` from val sweep (default 0.5; tune on banked pack).

**Step 3: Commit**

```bash
git commit -m "feat(coded_model): inference helpers"
```

---

### Task 5: Optional persist `p_coded` on hatespeech/

**Files:**
- Modify: `analysis/src/kma/hatespeech.py` (`refresh_measure` / `score_new` path)
- Modify: `docs/analysis/data-model.md`

**Step 1:** When embeddings exist for the batch, add `p_coded` float and `coded_model_flag` bool (`p_coded >= thr`).

**Step 2:** Operational OR for automation:

`flagged = hate_flag | coded_suspect | coded_model_flag`  
(document that `coded_suspect` remains the interpretable lexicon∩NLI series; model is auxiliary).

**Step 3:** Refresh subset; verify counts; commit

```bash
git commit -m "feat(hatespeech): persist p_coded from emb model"
```

---

### Task 6: Notebook series

**Files:**
- Modify: `analysis/notebooks/hatespeech.py`

Add a third thin series or callout: daily `coded_model_flag` rate vs `coded_suspect` on Kenya scope (measurement only).

**Commit:** `feat(hatespeech): chart p_coded series`

---

### Done when

- Weak labels + train/eval scripts reproducible without new human labels.
- Model beats or matches lexicon∩NLI on banked eval under the promotion rule.
- Inference path exists in kma; fixture FPs remain suppressed at the chosen threshold.
- If persisted: `p_coded` documented in data-model and available on latest hatespeech rows.

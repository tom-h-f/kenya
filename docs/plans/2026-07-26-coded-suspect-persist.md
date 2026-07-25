# Operationalize coded_suspect on hatespeech/ Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Persist Kenya-scope measurement fields (`domain`, `coded_suspect`, and related) as columns on the existing R2 `hatespeech/` prefix so automated consumers can OR them with `hate_flag` without a human queue.

**Architecture:** Keep the afro-xlmr 3-class scores unchanged. At write time (and via a CPU refresh backfill), join post text + latest `incitement/` NLI, run `kma.measure`, and append measurement columns onto each `hatespeech/` row. New enrich scores and a one-shot refresh cover the corpus. Notebooks read the persisted columns when present, falling back to live `attach_measurement_columns` only if columns are missing (old rows mid-rollout).

**Tech Stack:** Python, DuckDB/R2 parquet, `kma.measure`, `kma.hatespeech`, `kma.enrich`, pyarrow

**Prerequisite:** Merge `feat/hate-measurement` (or equivalent) so `analysis/src/kma/measure.py` and its tests exist on the branch you implement from.

**Constraints:** No new human/Opus labels. No analyst queue. Do not change `HATE_THRESHOLD` (0.28) or model weights in this plan.

---

### Task 1: Document the extended hatespeech schema

**Files:**
- Modify: `docs/analysis/data-model.md` (hatespeech/ section)

**Step 1: Extend the hatespeech schema docs**

Add columns (all written with every new/refreshed row):

| column | type | meaning |
|---|---|---|
| `domain` | string | `kenya` / `offdomain` / `ambiguous` |
| `in_kenya_scope` | bool | `domain != offdomain` |
| `lexicon_hits` | list\<string\> | live `scan_text` hits at write time |
| `coded_suspect` | bool | `measure.coded_suspect(...)` using joined NLI |
| `explicit_toxic` | bool | Kenya-scoped `(label != neither) \| hate_flag` |
| `flagged` | bool | `hate_flag \| coded_suspect` (Kenya-aware operational OR; `coded_suspect` already false when NLI missing) |

Note: older parquet runs lack these columns; readers must `union_by_name` and treat null measure cols as “needs refresh”.

**Step 2: Commit**

```bash
git add docs/analysis/data-model.md
git commit -m "docs: hatespeech measure columns schema"
```

---

### Task 2: Helper to build measure columns for a scored batch

**Files:**
- Modify: `analysis/src/kma/hatespeech.py`
- Test: `analysis/tests/test_hatespeech_measure.py` (create)

**Step 1: Write failing tests**

```python
# analysis/tests/test_hatespeech_measure.py
import pandas as pd
from kma.hatespeech import attach_persisted_measure

def test_attach_persisted_measure_madoadoa():
    posts = pd.DataFrame({
        "platform_post_id": ["1"],
        "text": ["next ni kutoa madoadoa Kenya. Tuwe safi kama pamba"],
        "label": ["neither"],
        "hate_flag": [False],
        "dehumanisation_score": [0.9],
        "violence_call_score": [0.8],
        "othering_score": [0.5],
        "political_criticism_score": [0.3],
    })
    out = attach_persisted_measure(posts)
    assert bool(out.loc[0, "coded_suspect"]) is True
    assert out.loc[0, "domain"] == "kenya"
    assert bool(out.loc[0, "flagged"]) is True

def test_attach_persisted_measure_offdomain_not_explicit():
    posts = pd.DataFrame({
        "platform_post_id": ["2"],
        "text": ["Remigrate Zohran Mamdani from America now"],
        "label": ["hate"],
        "hate_flag": [True],
        "dehumanisation_score": [0.1],
        "violence_call_score": [0.1],
        "othering_score": [0.1],
        "political_criticism_score": [0.1],
    })
    out = attach_persisted_measure(posts)
    assert out.loc[0, "domain"] == "offdomain"
    assert bool(out.loc[0, "explicit_toxic"]) is False
    assert bool(out.loc[0, "coded_suspect"]) is False
```

**Step 2: Run tests (expect fail)**

```bash
cd analysis && uv run pytest tests/test_hatespeech_measure.py -q
```

**Step 3: Implement `attach_persisted_measure`**

In `hatespeech.py`, wrap `measure.attach_measurement_columns` and add:

```python
def attach_persisted_measure(df: pd.DataFrame) -> pd.DataFrame:
    out = measure.attach_measurement_columns(df)
    out["flagged"] = out["hate_flag"].fillna(False) | out["coded_suspect"].fillna(False)
    return out
```

Ensure NLI columns may be absent (all-null) → `coded_suspect` False.

**Step 4: Tests pass; commit**

```bash
git add analysis/src/kma/hatespeech.py analysis/tests/test_hatespeech_measure.py
git commit -m "feat(hatespeech): attach persisted measure cols"
```

---

### Task 3: Write measure columns in `score_new`

**Files:**
- Modify: `analysis/src/kma/hatespeech.py` (`score_new`, `_pending`)
- Modify: `analysis/src/kma/db.py` only if a join helper is useful (optional)

**Step 1: Pending posts include text; after `_predict`, join latest incitement NLI**

For the batch of `platform_post_id`s, SQL LEFT JOIN `latest_incitement` (or inline qualify) to pull `dehumanisation_score`, `violence_call_score`, `othering_score`, `political_criticism_score`.

**Step 2: Build arrow table with existing + measure columns**

After `attach_persisted_measure`, persist:

- existing: `label`, `p_*`, `hate_flag`, `model`, `scored_at`
- new: `domain`, `in_kenya_scope`, `lexicon_hits` (from `lexicon_hits_live`), `coded_suspect`, `explicit_toxic`, `flagged`

Store `lexicon_hits` as list\<string\> (same pattern as `incitement/`).

**Step 3: Unit-test table column set with a mocked predict (no GPU)** if feasible; otherwise rely on Task 2 + a small integration skip-if-no-env test.

**Step 4: Commit**

```bash
git commit -m "feat(hatespeech): persist measure cols on score_new"
```

---

### Task 4: CPU refresh/backfill for existing hatespeech rows

**Files:**
- Modify: `analysis/src/kma/hatespeech.py` (add `refresh_measure`)
- Modify: CLI in same module (`--refresh-measure`)
- Optional: `analysis/modal_backfill.py` note or flag if that is the GPU backfill entrypoint (measure refresh is CPU)

**Step 1: Implement `refresh_measure(con, platform="x", limit=None) -> int`**

Algorithm:
1. Load latest posts ⨝ latest hatespeech (model cols) ⟕ latest incitement NLI.
2. `attach_persisted_measure`.
3. Write a new `hatespeech/.../run=*.parquet` with **full** rows (model probs copied + new measure cols), `scored_at=now` so `latest_*` picks them up.
4. Bound with `limit` for dry runs.

**Step 2: CLI**

```bash
cd analysis && uv run python -m kma.hatespeech --refresh-measure --limit 500
# full corpus when ready:
uv run python -m kma.hatespeech --refresh-measure
```

**Step 3: Dry-run on 500 rows against R2; assert new columns present in the written schema (read back one run or query latest).**

**Step 4: Commit**

```bash
git commit -m "feat(hatespeech): refresh_measure backfill"
```

---

### Task 5: Wire notebook / consumers to persisted `flagged`

**Files:**
- Modify: `analysis/notebooks/hatespeech.py` (prefer persisted cols; fallback to live measure)
- Modify: `docs/analysis/data-model.md` quirks if needed

**Step 1:** In the load cell, SELECT the new columns when present. If `coded_suspect` is null for a row, run `attach_measurement_columns` for those rows only (or whole frame during rollout).

**Step 2:** Headline stats use persisted `explicit_toxic` / `coded_suspect` / `flagged` on Kenya scope.

**Step 3: Commit**

```bash
git commit -m "feat(hatespeech): notebook uses persisted measure cols"
```

---

### Task 6: Full refresh + verification

**Step 1:** Run full `--refresh-measure` (CPU; long R2 write).

**Step 2:** Query latest hatespeech:

```sql
-- via uv run python / duckdb
-- expect: coded_suspect True count ~tens; offdomain hate_flag still stored
-- but explicit_toxic False for offdomain; flagged = hate_flag OR coded_suspect
```

Compare order-of-magnitude to fixtures summary: `coded_suspect_kenya` ~30, `hate_flag_offdomain` ~40 (those remain `hate_flag` True but `explicit_toxic` False).

**Step 3:** `cd analysis && uv run pytest tests/test_measure.py tests/test_hatespeech_measure.py tests/test_incitement.py -q`

**Step 4: Commit** any doc/threshold notes; do not change gates here (that is plan 2).

---

### Done when

- New `score_new` rows always include measure columns.
- Full corpus refreshed once.
- Notebook rates match prior live-measure behavior within small drift (NLI join coverage).
- `flagged` is available for any automated consumer without a human queue.

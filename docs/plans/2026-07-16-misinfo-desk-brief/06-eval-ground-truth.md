# 06 — Eval & ground truth gate

> **For Claude:** Final regression gate before calling the rebuild done.

**Goal:** Extend eval so main triage + thin lane expectations are explicit; known limitations are not silent failures.

---

### Task 1: Expectations for tiers

**Files:**
- Modify: `analysis/src/kma/eval.py`
- Modify: `analysis/run_eval.py`
- Test: `analysis/tests/test_ground_truth.py`

**Behaviour:**
- Cases may declare `lane: main | thin | either`
- `surface` required for SACCO-style in main
- Motorcade-style: `thin` surface **or** `known-limitation` with note
- Report prints lane + tier

**AC:**
- [ ] `uv run python run_eval.py` exits 0 under updated expectations
- [ ] Known limitations listed in report, not treated as silent pass/fail confusion
- [ ] Short “how to add a case” note in this file (below) or `docs/analysis/`

---

### How to add a ground-truth case

1. Add a case dict in `kma/eval.py` with: name, expected lane, expect (`surface` | `known-limitation`), keyword/entity probes, note.
2. Add/adjust unit test in `tests/test_ground_truth.py` if logic is pure.
3. Run `uv run python run_eval.py` against live R2 when validating recall (optional in CI).

## Verify

```bash
cd analysis && uv run pytest tests/test_ground_truth.py -v
cd analysis && uv run python run_eval.py
```

(Live `run_eval.py` may require R2 credentials; unit tests must pass offline.)

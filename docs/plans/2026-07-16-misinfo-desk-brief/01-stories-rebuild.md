# 01 — Stories rebuild

> **For Claude:** Use TDD. Commit when this plan’s AC pass.

**Goal:** Make claims the reliable spine: stable `story_id`, main triage lane unchanged in spirit, plus a thin high-gap lane for small uncorroborated claims.

**Architecture:** Extend `kma.stories` without breaking existing scorecard consumers. Thin lane is a separate tier, not a backdoor into high suspicion.

**Tech stack:** DuckDB, Polars, existing embeddings + trusted-source corroboration.

---

### Task 1: Stable story_id

**Files:**
- Modify: `analysis/src/kma/stories.py`
- Test: `analysis/tests/test_stories.py`

**Steps:**
1. Add deterministic `story_id` (hash of sorted member post ids, or min post id + size) on candidate/scorecard rows.
2. Test: same member set → same id; different set → different id.
3. Commit: `feat(stories): add stable story_id`

**AC:**
- [ ] `story_id` present on candidate and scorecard frames
- [ ] Deterministic across runs for identical member sets

---

### Task 2: Thin high-gap lane

**Files:**
- Modify: `analysis/src/kma/stories.py`
- Test: `analysis/tests/test_stories.py`

**Behaviour:**
- Main lane: keep `DEFAULT_MIN_SIZE` (3) + existing scorecard weights.
- Thin lane: allow `min_size_thin=2` (or configurable), require maximal/near-maximal corroboration gap + entity novelty signal; set `tier="thin_evidence"`.
- Thin stories must **not** sort into “high suspicion” solely on gap; scorecard may still compute but UI/eval treat tier separately.

**Steps:**
1. Failing tests for thin inclusion and non-elevation.
2. Implement `candidate_stories(..., include_thin=True)` or `candidate_stories_thin` + merge helper returning `tier`.
3. Pass tests; commit: `feat(stories): add thin high-gap evidence lane`

**AC:**
- [ ] Unit tests: thin-lane inclusion when gap high and authors≥2
- [ ] Unit tests: thin stories not auto-elevated to high suspicion without amp/coord signals
- [ ] Main-lane defaults unchanged for existing callers unless opt-in

---

### Task 3: Persist tier + story_id

**Files:**
- Modify: `analysis/src/kma/stories.py` (`persist_stories`)
- Test: `analysis/tests/test_stories.py`

**AC:**
- [ ] Persisted parquet schema includes `story_id`, `tier`
- [ ] Tests cover schema columns (mock write or column assert on frame before write)

---

### Task 4: Ground-truth expectations (partial)

**Files:**
- Modify: `analysis/src/kma/eval.py` (thin-lane hook; full gate in 06)
- Test: `analysis/tests/test_ground_truth.py`

**AC:**
- [ ] SACCO-style case still expected in main triage
- [ ] Motorcade-style case expected in thin lane **or** marked known-limitation with note
- [ ] `uv run pytest analysis/tests/test_stories.py analysis/tests/test_ground_truth.py` pass

## Verify

```bash
cd analysis && uv run pytest tests/test_stories.py tests/test_ground_truth.py -v
```

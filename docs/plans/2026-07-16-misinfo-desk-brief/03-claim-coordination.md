# 03 — Claim-scoped coordination

> **For Claude:** Use TDD. Depends on 01 (story members). Full multiplex stays in research notebook.

**Goal:** Answer “who co-amplified *this claim*?” without dumping global Leiden clusters on the journalist.

**Architecture:** Thin wrappers over `kma.coordination` that filter validated edges / cluster membership to story authors ∪ amplifiers.

---

### Task 1: Account set for a story

**Files:**
- Modify: `analysis/src/kma/coordination.py` or create `claim_coordination` helpers in `kma/stories.py` / `kma/framing.py` — prefer `kma/coordination.py` functions:
  - `story_account_set(con, story) -> set[str]`
  - `claim_coordination(con, accounts | story, ...) -> edges, clusters, summary`

**AC:**
- [ ] Empty amplifier set → empty edges/clusters, no crash
- [ ] All returned edge endpoints ⊆ account set
- [ ] Summary includes channel mix, size, overlap with global clusters when available
- [ ] Language remains triage (no auto “inauthentic campaign” label)

---

### Task 2: Tests

**Files:**
- Modify/create: `analysis/tests/test_coordination.py`

**AC:**
- [ ] Filter invariant tested with synthetic edges
- [ ] Empty input tested
- [ ] `uv run pytest tests/test_coordination.py -v` pass

## Verify

```bash
cd analysis && uv run pytest tests/test_coordination.py -v
```

## Note

Wave B (co-hashtag / co-URL / co-mention) remains data-gated; do not block this plan on Wave B coverage.

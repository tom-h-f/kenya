# 02 — Claim-anchored framing

> **For Claude:** Use TDD. Depends on 01 (`story_id` + member posts).

**Goal:** For each story, answer “what narrative neighborhood is this claim in, and how is mood shifting?” without treating corpus-wide topics as the claim itself.

**Architecture:** New helpers in `kma/framing.py` (preferred) or `kma/stories.py`. Reuse `assign_topics` / labels; do not invent language slices from X `lang`.

---

### Task 1: story → topic mapping

**Files:**
- Create: `analysis/src/kma/framing.py`
- Test: `analysis/tests/test_framing.py`

**API (minimum):**
```python
def story_topics(con, stories, topics_df) -> pl.DataFrame
# columns: story_id, topic_id, topic_terms, overlap_n / sim
```

**AC:**
- [ ] Returns per-story nearest topic(s)
- [ ] Empty/noise topic (-1) handled without crash
- [ ] Tests with fixture frames

---

### Task 2: Local keywords + sentiment timeline

**API:**
```python
def story_framing(con, story_id | stories, *, days=None) -> dict | pl.DataFrame
# topic_ids, topic_terms, top_keywords, sentiment_timeline
```

**AC:**
- [ ] `top_keywords` from claim neighborhood (c-TF-IDF or reuse story keywords)
- [ ] `sentiment_timeline` buckets by time over claim window using `labels/`
- [ ] Empty neighborhood → empty keywords + explicit empty timeline (no exception)
- [ ] Primary slices do not use X `lang`

---

### Task 3: Optional live stance passthrough

Reuse `kma.classify.stance` for investigator-chosen targets; document as live/not persisted.

**AC:**
- [ ] Notebook/docs note stance is live and target-parameterized
- [ ] No new persistence required

## Verify

```bash
cd analysis && uv run pytest tests/test_framing.py -v
```

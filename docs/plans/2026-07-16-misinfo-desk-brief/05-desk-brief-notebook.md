# 05 — Desk brief notebook

> **For Claude:** Depends on 01–04 APIs. Extensive detail is intentional.

**Goal:** Ship `analysis/notebooks/desk_brief.py` — investigator-facing, extensive, composed from rebuilt signals.

---

### Task 1: Scaffold marimo notebook

**Files:**
- Create: `analysis/notebooks/desk_brief.py`
- Modify: `analysis/README.md` (how to open)

**Controls:**
- Time window (default 7d)
- Focus: all / main triage / thin_evidence / single `story_id`
- Optional filters: topic, region, community (aggregate)

**Sections:**
1. Circulating claims (main + thin lanes)
2. Corroboration desk (nearest trusted always shown)
3. Amplifiers and origin
4. Framing shifts (topic + temporal mood)
5. Claim-scoped coordination
6. Community and region lens (**disclaimer banner**)
7. What’s new vs prior `stories/` persist run
8. Per-story deep dive (exemplars, timelines, keywords, human follow-up questions — as much detail as needed)

**AC:**
- [ ] Notebook imports only public `kma.*` APIs from 01–04
- [ ] Every community section displays `TRIBE_DISCLAIMER`
- [ ] Markdown never auto-calls a claim false or an account a bot
- [ ] `analysis/README.md` documents `uv run marimo edit notebooks/desk_brief.py`
- [ ] `uv run marimo check notebooks/desk_brief.py` succeeds (or project-equivalent)

---

### Task 2: Docs pointer

**Files:**
- Modify: `docs/analysis/README.md` — Phase 5 row linking to this plan folder

**AC:**
- [ ] Phase 5 listed with status and link to `docs/plans/2026-07-16-misinfo-desk-brief/`

## Verify

```bash
cd analysis && uv run marimo check notebooks/desk_brief.py
```

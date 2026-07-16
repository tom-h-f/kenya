# Misinfo Desk Brief — Plan Index

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task.

**Goal:** Rebuild claim/topic/coordination/community signals for journalistic use, then ship an extensive investigator-facing marimo desk brief.

**Architecture:** Method labs (phase notebooks) stay. New APIs feed `analysis/notebooks/desk_brief.py`. Claims are the spine; topics/coordination/community are lenses on claims. No auto false/bot verdicts.

**Tech stack:** Python, DuckDB, Polars, marimo, existing `kma` package, R2 Parquet.

## Reading order

| # | File | Ships |
|---|---|---|
| 00 | [00-design.md](00-design.md) | Locked product/method decisions |
| 01 | [01-stories-rebuild.md](01-stories-rebuild.md) | Claim spine + thin high-gap lane |
| 02 | [02-claim-framing.md](02-claim-framing.md) | Story↔topic + temporal framing |
| 03 | [03-claim-coordination.md](03-claim-coordination.md) | Claim-scoped CIB slices |
| 04 | [04-community-region.md](04-community-region.md) | Aggregate geo/community for claims |
| 05 | [05-desk-brief-notebook.md](05-desk-brief-notebook.md) | Extensive `desk_brief.py` |
| 06 | [06-eval-ground-truth.md](06-eval-ground-truth.md) | Eval harness + regression gate |

## Dependencies

```
00 → 01 → 02 → 04 → 05 → 06
         ↘ 03 ↗
```

## Global acceptance criteria

- [ ] Gap ≠ false; coordination ≠ malice; bot score ≠ bot; capture-is-sample caveats appear wherever those signals surface
- [ ] Community/tribe findings are aggregate-only with `TRIBE_DISCLAIMER` always visible
- [ ] Phase 1–4 research notebooks still run; desk brief is additive
- [ ] `uv run pytest` in `analysis/` green after each numbered plan
- [ ] Commits use conventional commit style; one logical chunk per commit

## Out of scope

- Wave B coordination channels blocked on data density (document only)
- RSS/news-site corroboration beyond trusted X handles
- Replacing phase notebooks
- Auto-publishing briefs outside the notebook

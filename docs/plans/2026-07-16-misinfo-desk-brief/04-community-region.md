# 04 — Community & region lens

> **For Claude:** Use TDD. Community is in scope and instructive; care is mandatory.

**Goal:** Aggregate region + community breakdowns for a claim’s author/amplifier set, with coverage stats and disclaimer.

**Architecture:** Extend `kma.deltas` with claim-scoped aggregations. Never emit per-author community in desk exports.

---

### Task 1: Claim-scoped slice API

**Files:**
- Modify: `analysis/src/kma/deltas.py`
- Test: `analysis/tests/test_deltas.py` (create if missing)

**API:**
```python
def slice_claim(
    con,
    author_handles: list[str] | pl.Series,
    dimension: Literal["region", "community"],
) -> pl.DataFrame
# aggregate volume/sentiment + coverage_pct + disclaimer
```

**Behaviour:**
- Always attach / return `TRIBE_DISCLAIMER` when `dimension=="community"` (and surface it for region cells that sit beside community).
- If mappable location coverage below threshold (e.g. <20%), return flag `insufficient_location_signal=True` and avoid sharp percentage headlines.

**AC:**
- [ ] Aggregate tables only (no per-author community column)
- [ ] Disclaimer present on community results (column or paired constant; notebook will assert display)
- [ ] Unmapped locations counted; low coverage → insufficient signal flag
- [ ] Tests cover unmapped + disclaimer presence

---

### Task 2: Wire to story account sets

**AC:**
- [ ] Helper accepts story member/amplifier handles from 01/03
- [ ] Unit test with mixed mapped/unmapped handles

## Verify

```bash
cd analysis && uv run pytest tests/test_deltas.py -v
```

## Policy (copy into notebook)

`TRIBE_DISCLAIMER` must appear on every community cell. Findings are aggregate-only and wrong for individuals / mixed urban / diaspora users.

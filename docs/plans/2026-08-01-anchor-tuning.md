# Anchor tuning for hate-seeking search

## Context

Kenya anchoring exists because the platform `lang` field is useless for Swahili
(~0% of Swahili posts tagged `sw`), so `term AND (Kenya OR Ruto OR ...)` is the
only way to scope an everyday Swahili word to Kenyan political discourse. The
accepted cost was stated up front: anchoring only finds coded posts that *also*
name a Kenyan target, and coded speech exists to avoid naming targets.

Two weeks of live sweeps (`collection_runs/`, the rendered-query audit trail) say
the cost is larger than intended, and that the tier assignment is backwards.

| tier | anchor set | terms | posts | posts/window |
|---|---|---|---|---|
| `low` | none | 4 | 426 | **3.6** |
| `high` | `wide` (12) | 3 | 48 | 0.5 |
| `medium` | `core` (6) | 9 | 20 | **0.1** |
| targets-of-hate | `wide` (12) + menace | 8 | **0** | 0.0 |

**92% of all yield came from the four unanchored terms.**

## The bug: tiers are inverted

Anchoring is `term AND (A OR B OR ...)`. Adding alternatives to the OR group makes
it **easier** to satisfy, i.e. *less* restrictive. The current mapping
(`hate_seek.ANCHOR_TIER`) is:

```python
{"low": None, "medium": "core", "high": "wide"}   # core=6 anchors, wide=12
```

So `high` false-positive terms - `mende` (cockroach), `nyoka` (snake), `fukuza`
(chase away), the everyday words that most need scoping - get the **loosest**
filter, while `medium` terms get the tightest. That is the opposite of the
documented intent ("`high` -> `wide` (12): everyday Swahili; unanchored they
return a region-wide noise firehose").

The yield numbers are consistent with the inversion (`medium` 0.1/window vs
`high` 0.5/window), though they are confounded by base rate: `mende`/`nyoka` are
common words, `kwekwe`/`madimoni` are rare. The confound is exactly why the fix
must be measured rather than assumed.

## Why this needs an experiment, not just a swap

Yield alone cannot tell us whether anchoring is helping. A term returning 0
posts might be correctly filtered (no Kenyan usage this fortnight) or wrongly
filtered (Kenyan usage that never names Ruto/IEBC/Kenya in the same post). The
`collection_runs/` manifest records the rendered query but not the counterfactual.

The decisive measurement is an A/B on the *same terms in the same windows*:
anchored vs unanchored, comparing both volume and **precision** (what share of
returned posts are actually Kenya-scoped, which `measure.domain_bucket` already
answers on the scored corpus).

## Plan

### 1. Fix the inversion (small, do first)

Swap the mapping so scoping tightens with risk:

```python
ANCHOR_TIER = {"low": None, "medium": "wide", "high": "core"}
```

`core` (6 anchors) is the tighter filter and should go to the high-fp terms.
Update the table in `docs/collection/hate-seeking.md`, which currently documents
the inverted logic as if it were correct. Add a test asserting that the anchor
set assigned to `high` is a **subset** of the one assigned to `medium`, so the
relationship cannot silently invert again.

### 2. Run the A/B (the actual decision)

New investigation `analysis/investigations/2026-08-01-anchor-ab/`:

- Take the 16-term register. For each term, issue the same day-windows twice -
  once anchored, once bare - via `monitor hate-seek --dry-run` rendering plus a
  bounded live pass. Write both to `hate_search` with a distinguishing
  `source` (`ab_anchored` / `ab_bare`) so the manifest separates them.
- Join the returned posts to `hatespeech/` and compute, per term and arm:
  - volume
  - **Kenya-scope precision**: share with `domain != 'offdomain'`
  - toxicity yield: share `flagged`
- Budget: 16 terms x 2 arms x 3 windows at `--limit 20` is ~1,000 posts, one
  off-peak pass. Cheap.

**Decision rule, fixed in advance** so the result cannot be rationalised:
keep anchoring for a term only where the bare arm's Kenya-scope precision drops
below 0.7. Otherwise drop to unanchored. Terms where both arms return < 5 posts
are inconclusive and stay as-is pending more data.

### 3. Reconsider targets-of-hate

Eight ethnonym x menace queries returned **zero posts across two weeks**. The
conjunction requires an ethnonym AND a menace term AND a Kenya anchor in one
post, which is close to requiring an explicit, self-labelling threat. Options,
in order of preference:

1. **Drop the anchor** (the ethnonym already establishes Kenya scope for
   `Kikuyu`/`Kalenjin`/`Luhya`; less so for `Somali`/`Nubian`).
2. **Widen the menace list** beyond the current 6 - it is a subset of the
   lexicon, and the highest-signal coded terms (`madoadoa`, `watajua hawajui`)
   are already searched bare, so the conjunction adds nothing for them.
3. **Retire the pass** if 1 and 2 still yield nothing. Zero-yield queries still
   cost pool budget and still carry the corpus-bias risk that put them in a
   quarantined partition.

Whatever changes, `hate_target_search` stays a separate partition excluded from
prevalence - that reasoning is unaffected.

### 4. Revisit the anchor vocabulary

`core`/`wide` are regime and opposition figures plus cities. Coded incitement
often names neither. Worth testing a third set built from high-frequency Kenyan
political tokens mined from the corpus rather than hand-picked - but only after
the A/B establishes whether anchoring is worth keeping at all.

## Files

- `collector/src/kenya_monitor/hate_seek.py` - `ANCHOR_TIER` (step 1)
- `collector/config/hate_terms.yaml` - anchor sets, `menace`, `ethnonyms` (step 3)
- `collector/tests/test_hate_seek.py` - subset assertion (step 1)
- `docs/collection/hate-seeking.md` - the anchor-policy table documents the
  inverted logic and must be corrected
- `analysis/investigations/2026-08-01-anchor-ab/` - new (step 2)

## Verification

- Step 1: `monitor hate-seek --dry-run` shows `mende` rendering with the 6-anchor
  set and `kwekwe` with the 12-anchor set; the subset test fails if reversed.
- Step 2: the A/B table, per term and arm, with the decision rule applied.
- Step 3: re-run the ethnonym pass and confirm non-zero yield, or retire it.
- Prevalence must not move: `hate_target_search` and `hate_search` stay in
  `TARGETED_TYPES`, so `latest_posts(scope="baseline")` is unaffected either way.

## Open question

Whether "anchored recall" is even the right target. The design's own premise is
that term search only needs to find a *foothold account*, after which
account-scoped expansion pulls that account at full recall. If that holds, a
high-precision/low-recall anchored query is fine and the 92%-from-unanchored
figure is not a problem to solve. The A/B's precision column is what tells us
which regime we are in - worth deciding explicitly before tuning further.

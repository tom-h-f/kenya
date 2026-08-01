# Census tuning: what we measured, what is settled, what is not

How object selection for the retweeter census was diagnosed and changed on
2026-08-01, and the three questions still open. Written because most of the
numbers involved are counter-intuitive and several of them contradict what the
code and older docs assert.

---

## 1. Why any of this matters: the chain from collection to a cluster

```
census an object  ->  engagement rows  ->  co_retweet traces  ->  hub cap
    ->  account x account projection  ->  significance test  ->  cluster
```

The **hub cap** is the step that makes census selection matter. From
`coordination.validated_edges`: objects acted on by more than the cap carry no
coordination signal, because *pairs sharing only a mega-viral tweet are organic*,
and one such hub dilutes the aggregate null until real clusters vanish.

The consequence is easy to miss: **an account whose only traces are on hub objects
disappears from the projection entirely.** `run_channel` rebuilds the trace table
after the cap and recomputes activity from it, so such an account cannot become a
cluster member by any path. Collecting it was wasted effort.

That makes "accounts discovered" an actively misleading metric. The number that
matters is **amplifiers surviving the hub cap**.

---

## 2. What was wrong

### 2a. The hub cap had silently stopped filtering

`hub_cap = max(50, 5% of accounts)`. The percentage term was meant to lift the cap
on small corpora, but it scales *with* the corpus:

| | |
|---|---|
| amplifying accounts | 64,002 |
| computed cap | **3,200** |
| busiest object | **1,171 amplifiers** |

The cap was three times larger than the largest object. It excluded nothing, and
had been excluding nothing for some time as the corpus grew.

Measured effect on the 14-day window: **1.9% of objects generated 99.3% of all
36.8M projected pairs.** That both OOM'd a 3.7Gi host and did precisely what the
cap exists to prevent.

Fixed by bounding it above (`HUB_CAP_MAX = 100`). Result was *better detection*,
not just lower memory:

| | vacuous cap (3,200) | bounded (100) |
|---|---|---|
| clusters | 1,078 | 298 |
| corroborated across both channels | 12 (47 accounts) | **14 (60 accounts)** |
| corroboration rate | 1.1% | **4.7%** |

~780 spurious single-channel clusters disappeared *and* more real clusters
survived the significance test.

### 2b. The census was selecting exclusively the useless band

Selection ranked by `repost_count DESC`, which is the intuitive choice and the
wrong one.

| | |
|---|---|
| objects censused | 422 |
| of those, hubs (>100 amplifiers) | **420 (99.5%)** |
| median censused degree | 295 |
| max | 1,192 |
| amplifier accounts collected | 128,556 |
| **surviving the hub cap** | **11,393 (8.9%)** |

Toxic-object coverage was worse than the headline 0.4% suggested. Broken down by
repost band, the coordination-useful population is small and had **zero** coverage:

| repost band | toxic objects | censused |
|---|---|---|
| 0 | 2,194 | 0 (uncensusable) |
| 1-2 | 367 | 0 |
| **3-100 (the useful band)** | **368** | **0** |
| >100 | 164 | 31 |

The denominator was never 8,453. It was **368**, and effective useful coverage was
**0%**.

### 2c. The economics invert once measured

`retweeters()` pages at 100, so cost scales with degree:

| object | requests | accounts | usable for coordination |
|---|---|---|---|
| hub (>100) | ~3 | ~300 | **0** |
| mid-band (3-100) | 1 | ~27 | **all 27** |

Banding is simultaneously **cheaper per object** and **strictly more useful**.
There is no trade-off being made.

### 2d. Two blockers behind the toxic path

- `hot_toxic_objects` hardcoded `lookback_days=2`. An object created three days
  ago and scored today was invisible to the census *forever* — the enrich backlog
  could never be worked at any cap. Now 14 days, matching `MAX_AGE_DAYS`,
  `HATE_LOOKBACK_DAYS` and `COORD_LOOKBACK_DAYS`.
- The hate steps used `cycle % N`, and `cycle` resets to 0 on every restart.
  Frequent redeploys meant they never ran — **zero executions across a whole
  container lifetime**. Now wall-clock scheduled; observed 9 executions after
  the fix.

### 2e. Selection was not TTL-aware

Found after banding shipped. Selection is deterministic (`repost_count DESC`), so
it re-picked the same already-censused objects every pass and `_due` discarded
them:

| | |
|---|---|
| objects selected per pass | 495 |
| actually fetched | **10-35** |
| uncensused in-band objects available | ~2,900 |

~95% of the selection budget was being spent re-picking completed work. Fixed by
excluding recently-censused objects **during selection**, using R2 as the source
of truth rather than the local JSON state, so it self-heals if state is lost.

---

## 3. Result

| metric | before | after |
|---|---|---|
| median censused degree | 295 | **90** |
| objects above the hub cap | 99.5% | **34.2%** |
| objects yielding usable pairs | 0.5% | **66%** |
| amplifiers surviving the cap | 11,393 | **17,349** |
| survival rate | 8.9% | **12.3%** |
| usable trace rows | 24,119 | **46,987** |
| toxic objects reachable per pass | ~70 | **245** |

---

## 4. The three open questions

All three share a root: everything above measures **inputs**. None of it
establishes that cluster *detection* improved.

### Q1 — Is a 12.3% survival rate sufficient?

**Plainly:** only about an eighth of the accounts we collect can ever be
considered as network members. The rest are known to exist but have no usable
behavioural evidence. That fraction went up, but nobody has checked whether it is
*enough*.

Why it is not answerable from what we have:
- The corpus carries a large historical tail of hub-only traces. Banding changes
  what we collect from here on; it cannot repair what is already stored. So the
  ratio understates the quality of *new* collection.
- Cluster detection is non-linear. Doubling usable traces does not double
  validated edges — the significance test may need a density threshold that we
  are either already past or nowhere near, and the ratio alone does not say which.

**Experiment.** Run `coordination_run --persist` after several TTL-aware passes
and compare against the recorded baseline: **65,570 co_retweet edges, 1,898
co_reply, 298 clusters, 14 corroborated (60 accounts)**. The decisive figure is
corroborated clusters, not total edges — total edges rose under the *broken* cap
too, which is exactly how the problem hid.

**What would falsify the work:** corroborated clusters flat or down while usable
traces rose. That would mean the census is not the binding constraint and the
effort belongs elsewhere.

### Q2 — Is `SNOWBALL_BAND_MAX = 100` the right cutoff?

**Plainly:** we only census posts with 3-100 retweets. Above 100 the maths throws
the result away. But **34.2% of what we fetch still comes back above 100**,
because a post keeps being retweeted between the moment we record its count and
the moment we fetch its retweeters.

Two candidate explanations, not yet separated:
1. Genuine growth between snapshot and census.
2. A selection artefact — `hot_objects` bands on `max(repost_count)` over
   *retweet rows*, which may understate the original's true count.

The measured ratio of censused degree to recorded `repost_count` is **0.97**,
which argues *against* systematic growth and therefore *for* explanation 2. That
also disproves a "snapshot lag" theory advanced earlier in this work — a
correction was nearly shipped for a lag that does not exist.

**Experiment.** For a sample of selected objects, record `repost_count` at
selection, the original post's own `repost_count`, and the censused degree.
Separating (1) from (2) determines whether the fix is a lower band max or a
corrected selection column. **Do not lower `SNOWBALL_RETWEETERS_LIMIT`** —
truncating the fetch would disguise a hub as a legitimate mid-degree object and
inject it into the projection, which is worse than the wasted request.

### Q3 — Does coverage plateau?

**Plainly:** every object is re-censusable after 12 hours. Once we have worked
through what is currently available, the rate of new-account discovery stops
depending on our settings and starts depending on how fast new content appears.

Current supply, 2-day window: **3,188 in-band candidates, 303 censused.** So there
is headroom now. What is unknown is the steady state once that gap closes — at
which point `SNOWBALL_REFRESH_HOURS`, not `SNOWBALL_TOP_RETWEETED`, becomes the
governing parameter.

This matters because growth has been reported as though it continues. It may not.

**Experiment.** Track uncensused in-band candidates per pass for a week. If it
trends to zero, coverage is saturated and the levers change: widen the band
downward (the 1-2 repost tier is 367 toxic objects, unpairable individually but
not worthless in aggregate), shorten the TTL, or accept saturation as success.

---

## 5. The metric of record

**`nohub_amp` — amplifier accounts surviving the hub cap.** Currently 17,349 of
141,306.

Not "accounts discovered". That number doubled while contributing almost nothing,
which is what made it convincing.

---

## 6. A note on how these were found

Four separate parameter changes were made without first checking what consumed
their output: the hub cap, the census cap, the banding/TTL interaction, and a
DuckDB memory limit applied to a workload that was simply too large.

**Every one produced a number that went up.** Accounts discovered doubled.
Objects censused doubled. Both were real measurements of the wrong thing.

The check that would have caught all four is the same: before changing a
parameter, trace what consumes its output and confirm the consumer can use more of
it. A green test suite did not help — `test_hot_objects_selection_and_missing_refs`
asserted that the *most-amplified* object wins, encoding the bug, and passed
throughout.

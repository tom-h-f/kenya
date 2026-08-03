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
disappears from the projection entirely.** `validated_edges` rebuilds the trace
table after the cap (`{t}_nohub`) and re-derives both the projection and
`activity()` from it, so such an account cannot become a cluster member by any
path. Collecting it was wasted effort.

That makes "accounts discovered" an actively misleading metric. Surviving the
cap is the first hurdle — but not the last, and not the metric of record
either. See §5.

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
| median censused degree | 295 | **73** |
| objects above the hub cap | 99.5% | **6.4%** [^1] |
| toxic objects reachable per pass | ~70 | **245** |

Account-level totals are deliberately not in this table. They were computed by
hand, by two different methods, days apart — which is the failure this document
is about. They now come from the pipeline's own counters; see §5.

[^1]: An earlier version of this table said **34.2%**. That figure was measured
over a window that straddled the banding commit (`eb4bc8f`, 2026-08-01 16:19),
so it averaged the old behaviour with the new. Split by census hour, censuses
before 16:00 were ~100% hubs at median degree 300-900; from 17:00 onward, 0-2
hubs per hour at median degree 34-80 and 6.4% above the cap. The lesson is
§6's, applied to §3: **never measure across a parameter change without
splitting on the change time.** `census_runs/` now carries `code_version` and a
timestamp on every pass so this cannot recur silently.

---

## 4. The three open questions

All three share a root: everything above measures **inputs**. None of it
establishes that cluster *detection* improved.

### Q1 — Is the survival rate sufficient?

**Plainly:** only a small fraction of the accounts we collect can ever be
considered as network members. The rest are known to exist but have no usable
behavioural evidence. That fraction went up, but nobody has checked whether it
is *enough*.

The fraction is also smaller than this document first claimed. It was stated as
12.3% (survivors / collected); the honest figure is **8.2%** (`pairable` /
`accounts_all`, 14,898 of 182,391 on 2026-08-02), because surviving the cap is
not the same as being able to appear in an edge — §5.

Why it is not answerable from what we have:
- The corpus carries a large historical tail of hub-only traces. Banding changes
  what we collect from here on; it cannot repair what is already stored. So the
  ratio understates the quality of *new* collection.
- Cluster detection is non-linear. Doubling usable traces does not double
  validated edges — the significance test may need a density threshold that we
  are either already past or nowhere near, and the ratio alone does not say which.

A third reason it was not answerable: **the ratio was measured against the wrong
denominator.** See §5 — the population that matters is `pairable`, not
`nohub_amp`, and it is 3.5x smaller.

**Experiment (now automatic).** Every `coordination_run --persist` writes its
counters to `coordination/kind=run_metrics`, so the comparison is a query over
the series rather than a hand-recorded baseline:

```sql
SELECT computed_at, code_version, channel, pairable, nohub_amp,
       n_clusters, n_corroborated_clusters, n_corroborated_accounts
FROM coordination_metrics ORDER BY computed_at;
```

The baseline to beat is the hand-recorded **65,570 co_retweet edges, 1,898
co_reply, 298 clusters, 14 corroborated (60 accounts)**. First job is to confirm
the series reproduces it; if it does not, the counters are wrong and nothing
else follows.

The decisive figure is corroborated clusters, not total edges — total edges rose
under the *broken* cap too, which is exactly how the problem hid.

**What would falsify the work:** corroborated clusters flat or down while
`pairable` rose. That would mean the census is not the binding constraint and
the effort belongs elsewhere.

### Q2 — Is `SNOWBALL_BAND_MAX = 100` the right cutoff? — **RESOLVED, no change**

**Yes.** The question rested on "34.2% of what we fetch still comes back above
100", and that number was an artefact of the measurement window (see the
footnote in §3). Post-deploy the rate is **6.4%**.

Both candidate explanations were tested against the retained snapshots and both
are refuted:

1. *Growth between snapshot and census.* Median (`repost_count` now /
   `repost_count` at selection) = **1.00** across 717 objects. There is no
   systematic growth. This also disposes of the "snapshot lag" theory advanced
   earlier — a correction was nearly shipped for a lag that does not exist.
2. *A selection artefact — banding on `max(repost_count)` over retweet rows
   understating the original's count.* The two candidate columns agree at a
   median ratio of 1.00. Decisively: of the 945 objects censused on 2026-08-01,
   the 289 that came back as hubs read ~8,000 on **both** columns
   (median 8,813 on retweet rows, 8,071 on the original's own row). No column
   misled selection; those objects were censused before banding existed.

Of the 427 objects that were genuinely in band at selection time, 88 (20.6%)
exceeded 100 by census time, at median degree 82 — a modest overshoot with no
single cause worth chasing.

**Do not lower `SNOWBALL_RETWEETERS_LIMIT`** — still correct, still load
bearing. Truncating the fetch would disguise a hub as a legitimate mid-degree
object and inject it into the projection, which is worse than the wasted
request.

**Standing guard:** `n_over_band_max` in `census_runs/`, with a warning above
15%. The healthy state (6.4%) clears it comfortably; a regression to the
pre-banding state (99.5%) trips it on the first pass.

### Q3 — Does coverage plateau?

**Plainly:** every object is re-censusable after 12 hours. Once we have worked
through what is currently available, the rate of new-account discovery stops
depending on our settings and starts depending on how fast new content appears.

**Saturation is imminent, not hypothetical.** Measured 2026-08-02 by running the
selector against live R2:

| | |
|---|---|
| in-band candidates, 2-day window | 3,496 |
| of those, selectable (not censused inside the 12h TTL) | **742** |
| per-pass budget (`SNOWBALL_TOP_RETWEETED`) | 250 |
| densest object still selectable | **63** (band max is 100) |

Two independent signs the band is nearly worked through: the selectable backlog
is under three passes' worth, and the upper half of the band is already empty —
nothing between 63 and 100 remains uncensused.

So `SNOWBALL_REFRESH_HOURS` is about to become the governing parameter, and
growth should stop being reported as though it continues.

**Experiment (now automatic).** `candidates_uncensused` is written to
`census_runs/` on every pass. Track it for a week rather than sampling it by
hand. If it flattens near zero the levers change: widen the band downward (the
1-2 repost tier is 367 toxic objects, unpairable individually but not worthless
in aggregate), shorten the TTL, or accept saturation as success. Which one is a
decision for the data, not for now.

---

## 5. The metric of record

**`pairable` — accounts holding at least `min_repetition` (2) non-hub objects.**

Not "accounts discovered": that number doubled while contributing almost
nothing, which is what made it convincing.

And **not `nohub_amp`** either, which is what this document originally
nominated. Surviving the hub cap is necessary but not sufficient.
`projected_edges` requires `min_repetition = 2` distinct shared objects, so an
account whose non-hub traces number exactly one cannot appear in any edge, in
any cluster, ever — no matter what else is collected.

`co_retweet`, from the pipeline's own counters, 2026-08-02:

| | |
|---|---|
| accounts with any trace (`accounts_all`) | 182,391 |
| surviving the hub cap (`nohub_amp`) | 52,871 |
| **holding >= 2 non-hub objects (`pairable`)** | **14,898** |
| survivors that can never form an edge | 37,973 |

`nohub_amp` overstates the usable population by **3.5x**. That ratio held at
3.55x on an independent hand calculation over the 14-day window the day before,
on a smaller corpus — it is a structural property of the degree distribution,
not an artefact of one snapshot. `nohub_amp` is kept in
`coordination/kind=run_metrics` only to show the size of the gap.

This is the same error §6 is about, committed by this document: a number that
went up, measuring the wrong thing. The check that catches it is the same one —
trace what consumes the output. Here the consumer is `min_repetition`.

The **outcome** measure remains corroborated clusters (`n_channels >= 2`), now
computed and persisted per run rather than derived by hand.

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

---

## 7. Found while measuring: the engagement arm ignores the coordination window

Not one of the three questions, and more consequential than two of them.

`traces()` scopes the post-derived arm of `co_retweet` to `COORD_LOOKBACK_DAYS`
(14) via `_latest_posts_cte`. `_engagement_traces` had **no time filter at
all** — it read the whole `engagements/` prefix, forever. Measured 2026-08-01:

| | |
|---|---|
| engagement traces total | 191,824 |
| on objects created more than 14 days ago | 34,682 (114 objects) |
| on objects with no post row at all, so age unknown | 10,629 (121 objects) |
| **outside the window `COORD_LOOKBACK_DAYS` claims to enforce** | **45,311 (23.6%)** |

Three harms, all silent:

1. `COORD_LOOKBACK_DAYS` did not do what its own comment says it does.
2. Stale hub-era census rows inflate object degrees. An object pushed over the
   hub cap by traces from three weeks ago takes **every account whose only
   traces were on it** out of the projection entirely (§1).
3. Edges could form out of months-old activity, contradicting the stated
   rationale — "a ring that was active months ago is not the current network".

**Fix.** The engagement arm semi-joins to `lp`, the same windowed relation the
post arm already uses, so the window is inherited rather than restated and the
two cannot drift. Objects with no post row are dropped: a retweet cannot precede
its object, which makes the object's `created_at` a sound bound, whereas census
`collected_at` bounds only when we *observed* the incidence — a year-old retweet
censused this morning would pass that test. The drop is a lag, not a loss:
those ids are exactly what `hot_objects`' hydration arm exists to fetch, and
they re-enter on their merits once it does.

---

## 8. How to read the two series

| question | prefix | helper |
|---|---|---|
| what did the census have to work with, and what did it do | `census_runs/` | `kma.db.census_runs` |
| what did detection make of it | `coordination/kind=run_metrics` | `kma.db.coordination_metrics` |

Both are **series, not state** — the helpers return every run, oldest first,
rather than the latest. That is deliberate: every question here is a difference
between runs.

**The one rule.** Never aggregate across a parameter change without splitting on
the change time. That single mistake produced §3's wrong 34.2% and very nearly
caused a correction to be shipped for a growth effect that does not exist. Both
prefixes carry `code_version` and a timestamp on every row so the split is
mechanical:

```sql
SELECT code_version, count(*) AS passes,
       median(candidates_uncensused) AS backlog,
       median(n_over_band_max * 1.0 / nullif(fetched_retweeted, 0)) AS over_band_rate
FROM census_runs GROUP BY 1 ORDER BY min(collected_at);
```

Two traps worth stating once. The degree columns in `census_runs/` measure the
**fetch**, which `SNOWBALL_RETWEETERS_LIMIT` truncates at 300 — sound as a
guard, not readable as a distribution. And `nohub_amp` is not the metric of
record; `pairable` is (§5).

---

## 9. Why corroborated clusters collapsed, and what it was really telling us

Recorded 2026-08-02, immediately after the measurement layer landed. The first
thing the new series did was falsify a claim this document makes.

### The trajectory

All runs below use hub cap 100, so the cap is not the variable:

| run | co_retweet edges | co_reply | clusters | corroborated |
|---|---|---|---|---|
| 08-01 10:35 | 3,751 | 400 | 298 | **14** |
| 08-01 19:03 | 20,281 | 664 | 414 | 4 |
| 08-02 01:20 | 63,963 | 793 | 725 | 5 |
| 08-02 07:31 | 107,366 | 1,227 | 348 | 2 |
| 08-02 19:40 | 140,518 | 1,210 | 164 | **1** |

Banding shipped at 16:19 on 08-01. co_retweet edges grew **37x in 33 hours**;
co_reply grew 3x. Layer imbalance went 9:1 to 116:1.

This is the falsification condition Q1 names: corroborated clusters down while
the input measure rose.

### It is not the census, and not the null

`null_baseline` on degree-preserving shuffled traces:

| channel | input | tested | Bonferroni | FDR |
|---|---|---|---|---|
| co_retweet | real | 334,240 | 5,426 | 141,831 |
| co_retweet | shuffled | 45,323 | **0** | **1,817** |
| co_reply | real | 1,300 | 28 | 1,238 |
| co_reply | shuffled | 916 | **1** | **256** |

Bonferroni passes; **the null model is sound**. FDR admits 4% and 28% of pure
noise. That is not a bug - BH permits alpha of *discoveries* to be false, and
it always did. The "0/0 on shuffled" recorded 2026-07-08 (doc 03) was a
small-corpus artefact: back then the shuffled input was too small for BH to
reject anything, which set a false expectation that survived into this system's
defaults.

### The failure is in community detection

Planting a known cluster with `inject_synthetic` and scoring with
`evaluate_recovery`, at a 20-account/10-object plant both filters score
F1 = 1.0 - FDR's extra ~139,000 edges buy no recall. They diverge at the limit:

| plant (accounts x objects) | Bonferroni | FDR |
|---|---|---|
| 20 x 10 | 1.0 | 1.0 |
| 8 x 5, 4 x 5 | **1.0** | **0.0** |
| 8 x 3 and below | 0.0 | 0.0 |

At 8 x 5, **both filters contain all 28 planted pairs** - the statistics find
the group either way. But Leiden places **0 of 8** planted accounts in any
cluster under FDR, against **8 of 8** under Bonferroni. The marginal edges wire
the group to unrelated accounts until CPM at resolution 0.05 can no longer hold
it together.

So the chain is: banding scaled the census honestly -> co_retweet edges grew
37x -> FDR's marginal tail grew with them -> Leiden dissolved tight groups ->
clusters became co_retweet blobs -> co_reply's edges rarely landed inside them
-> corroboration collapsed.

**The census work is exonerated.** It scaled the input; a latent flaw in the
edge filter converted that into worse detection. Corroboration was doing its
job - the collapse was the signal that the new clusters were not real.

**Fixed** by defaulting community detection to Bonferroni
(`DEFAULT_EDGE_METHOD`). Both filters share a sensitivity floor of 3 shared
objects, so nothing is given up. `percentile` is disqualified outright: it
missed even the 20 x 10 plant on co_retweet.

### A third measurement error in this document

Q1's baseline - "65,570 co_retweet edges, 1,898 co_reply, 298 clusters, 14
corroborated" - **conflates two runs two days apart**. The edge counts are the
07-30 22:37 run; the cluster counts are 08-01 10:36, which actually had 3,751
and 400 edges. Same error as the Q2 figure in §3 and the account totals in §5.

There is therefore **no known-good baseline to return to**. It has to be
re-established from a calibrated pipeline, which is what §8's series is for.

### Still open

Two defects found here are not yet fixed, deliberately, so their effects stay
attributable:

- `aggregate_layers` normalises each layer's weight *magnitude* to <= 1 and
  then sums, so a layer's influence still scales with its edge **count**. At
  116:1 the smaller layer contributes nothing. Per-layer mass normalisation or
  a top-K budget would fix it independently of the filter choice.
- `co_retweet` unions *complete enumeration* (censused retweeters) with
  *sampling* (post-derived retweets) in one channel, giving it two sampling
  probabilities where the null assumes one. Splitting them would make each
  channel internally homogeneous and yield a genuinely independent third
  channel for corroboration.

---

## 10. Corroboration is unavailable, and not for a fixable reason

Recorded 2026-08-02/03, after §9's filter change. Three candidate fixes were
tried and all three are dead ends. The finding matters more than any of them.

### What the fixes did

**Windowing the engagement arm** (§7) worked as intended: co_retweet's
Bonferroni layer fell from 5,698 edges to **1,548** once traces outside
`COORD_LOOKBACK_DAYS` stopped accumulating. Layer imbalance improved from 203:1
to 43:1.

**Equal-mass layer normalisation did not, and would have broken the pipeline.**
The idea was to stop co_retweet drowning co_reply by giving each layer equal
total weight. Measured on live data:

| normalisation | clusters | members | corroborated |
|---|---|---|---|
| `max` (original) | 95 | 435 | 0 |
| `mass` | **0** | **0** | 0 |

CPM compares a community's internal weight against an **absolute**
`resolution_parameter` (0.05). Dividing 1,548 edges by their total puts every
weight near 0.0006, so no community clears the threshold and everything becomes
a singleton. Reverted; kept as a parameter with the result recorded, because
"balance the layers" is an obvious idea that needs a standing answer.

### The actual reason, which no weighting can fix

| | |
|---|---|
| accounts in the validated co_reply layer | 49 |
| of those, appearing anywhere in co_retweet | **0** |
| pairs validated in both channels | **0** |

**The two channels observe disjoint populations.** co_retweet is dominated by
accounts discovered through the retweeter census - they exist only as
engagement rows and have no posts at all. A co_reply trace requires a collected
post carrying `in_reply_to_id`. An account cannot be in both unless we have
both its retweets and its posts, and for census-discovered accounts we have
only the former.

Reweighting cannot manufacture connectivity that is not there. Neither can a
different edge filter, a different resolution, or splitting `co_retweet` into
its census and post arms - that last one would make corroboration *easier* to
achieve and *less* meaningful, since both arms would be observing the same
behaviour by two routes rather than two behaviours.

### What this means

- **A zero in `n_corroborated_clusters` currently means "cannot be evaluated",
  not "nothing is coordinated".** Do not read it as a detection result, and do
  not tune anything to raise it.
- The census is very good at discovering accounts it can only ever see through
  one channel. That is not a bug in the census; it is the cost of the discovery
  route, and it was invisible until corroboration was computed per run.
- The only fix is to make the populations overlap: collect timelines for
  accounts that appear in co_retweet clusters, so they acquire posts and become
  eligible for the reply channel. `adaptive.cluster_accounts` already promotes
  cluster members to targeted collection, which does exactly this - but it
  feeds the sampling loop `coordination_run`'s docstring warns about, and
  promoted accounts land in quarantined partitions excluded from every rate.
  Whether that is acceptable for corroboration purposes is an open decision,
  not a settled one.

### The fix that shipped

`run_census_timelines_once` (collector, one step per cycle) collects timelines
for a **random sample** of census-discovered accounts - 20 per cycle, 30 posts
each - writing to a new `posts/type=census_timeline` partition.

The candidate draw is **pooled, not per-cycle**. Measured against live R2 at
800MB/2 threads, the candidate query costs 464s with a posts anti-join and 215s
without, and the cost is dominated by scanning the parquet globs over the
network rather than by how many rows come back. Running that every cycle on a
Raspberry Pi to pick 20 handles is indefensible, so a pool of 500 is drawn every
12h and spent across cycles. The posts anti-join was dropped outright: it exists
to skip accounts that already have posts, and 179,106 of ~180,000 censused
accounts have none, so it doubled the cost to exclude about 1%. An `attempted`
map handles repeats, which is what actually matters - accounts with no tweets
never gain a post row and would otherwise be redrawn forever.

Random is the whole design. The obvious alternative, promoting accounts that
appear in coordination clusters, is what `adaptive.cluster_accounts` already
does, and it cannot be used here: it would give exactly the suspected accounts
more posts, more traces and more edges, manufacturing the corroboration it was
meant to measure. The `cib_timeline` quarantine does not save it either -
`_latest_posts_cte` builds coordination traces from `posts_source(platform)`
with `type='*'`, so targeted posts feed the projection like any other. The
quarantine protects prevalence rates, not coordination.

Random selection is exogenous to the outcome, so it closes the population gap
without conditioning on the answer. `census_timeline` is registered in
`TARGETED_TYPES` so it still stays out of every prevalence rate - these
accounts are retweeters of banded objects, not a sample of the population.

**Replies must be requested explicitly.** twscrape's `user_tweets` omits them,
and the first live pass proved it: 760 posts from 20 accounts carried **5** rows
with `in_reply_to_id`, so the pass generated almost no `co_reply` traces - the
one thing it exists to produce. `timeline(include_replies=True)` switches to
`user_tweets_and_replies`, and it is opt-in: turning it on for baseline
timelines would change what a timeline MEANS for every prevalence measurement
built on that partition.

Cost and pace: 20 accounts per cycle against ~168,000 census-discovered
accounts. This is deliberately slow. It is not trying to collect them all, only
to build a population where the two channels can be compared at all.

Note the deadlock it breaks. `cluster_accounts` requires `n_channels >= 2`
(effective value on pi0), corroboration is 0, so it promotes nobody - and the
one mechanism that could have given these accounts posts was gated behind the
metric its absence zeroes out. That path now logs when it promotes nobody, so
the state stops being silent.

### Status of the outcome measure

Until the populations overlap, the honest outcome measures are the ones that do
not require two channels: cluster count and size distribution on a calibrated
filter, plus planted-cluster recovery (§9), which is synthetic but reproducible
on demand. `pairable` (§5) remains the input measure.

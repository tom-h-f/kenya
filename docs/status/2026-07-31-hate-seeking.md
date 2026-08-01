# Hate-seeking collection: status report

**As of 2026-08-01.** Covers the hate-seeking build, the pi0 outage and recovery,
and the seed-selection rebuild. Everything below is measured on the live corpus;
where a number is an estimate it says so.

---

## 1. Where we are

The full loop is live for the first time:

```
collector   hate-seeking search + seed expansion  ->  posts/type=hate_*
analysis    kma.enrich (embed, classify, hate + measure cols)  ->  hatespeech/
analysis    coordination_run --persist (6-hourly)  ->  coordination/
collector   hate_signal + adaptive.promote  ->  next pass's targets
```

| | state |
|---|---|
| corpus | **193,862 posts**, all partitions collected within the hour |
| hate scoring | **193,862 scored, 0 unscored**, all carrying measure columns |
| coordination | 298 clusters / 1,406 accounts; **14 clusters / 60 accounts corroborated** across both channels; refreshed 6-hourly on tf1 |
| seed cohort | **217 accounts**, 68 with brigade evidence, 95 with repeat co-amplification |
| adaptive promotion | 30 dynamic targets, including coordination-cluster accounts |

Hosts:

| host | role | state |
|---|---|---|
| **pi0** (arm64, 3.7G) | collector daemon | healthy, current code, hate-seek enabled |
| **tf1** (x86_64, 3.7G) | enrich + coordination | coordination refreshing on schedule; **hate pass blocked on missing `HF_TOKEN`** |
| Modal (A100) | GPU drain | used for the 59k catch-up |

---

## 2. What was built

### Acquisition
16-term coded register (`config/hate_terms.yaml`) mirroring `kma.incitement.LEXICON`,
with false-positive-risk-tiered Kenya anchoring, ethnonym x menace queries in a
separate quarantined partition, rotation state, and a `collection_runs/` audit
trail recording every **rendered** query.

### Targeting
`hate_signal.py` ranks accounts as network seeds in pure SQL from the
`hatespeech/` prefix. Three independent qualifying routes (volume, repeat
co-amplification, brigade participation); ranking weights coordination shape
0.45 against toxicity 0.25.

### Mining
`mine_terms` contrasts the seed cohort's vocabulary against the corpus, ranked by
temporal novelty and gated on lift. Candidates are human-approved before they can
drive searches (`HATE_MINE_AUTOPROMOTE=0`).

### Frontier
`hate_frontier.py` records which accounts have been expanded so each pass takes
fresh seeds. Improves on the existing `follow_crawl` ledger in four ways: score
ordering rather than FIFO, an attempt cap, pruning, and atomic writes.

### Measurement integrity
`kma.db` first-seen scoping keeps targeted partitions out of every prevalence
denominator. Measure columns (`domain`, `coded_suspect`, `flagged`, …) are
persisted onto `hatespeech/` so the collector can scope to Kenya in pure SQL.

---

## 3. What measurement changed

The design was wrong in several places, and only running it against real data
showed it. In rough order of importance:

### The floors contradicted the weights
`n_posts >= 10 AND n_toxic >= 3` ran *before* scoring, cutting 37,211 authors to
29 and then 8. That discarded **936 of 939 brigade participants**, so
`n_brigade_convs` was exactly 0 for every seed and the `brigade` component —
weight **0.20** — contributed nothing at all. Brigading is many accounts each
posting once; a sustained-volume floor removes brigaders by construction.

Fixed by making qualification permissive across three routes and letting
*ranking* be the selective step.

| | before | after (partial corpus) | after (full corpus) |
|---|---|---|---|
| cohort | 8 | 50 | **217** |
| brigade evidence | **0** | 11 | **68** |
| repeat co-amplification | — | 19 | **95** |

### Co-amplification was measuring virality, not coordination
Of 385 toxic objects, **9 had more than 50 amplifiers (max 223) and those 9 alone
produced 43% of every co-amplification row**. `n_cotoxic_peers >= 2` was nearly
free: 1,559 accounts cleared it, but only 29 repeatedly co-amplified with the same
partner once hubs were excluded. Fixed with a hub cap and a repetition bar
mirroring `kma.coordination`.

### Persisted coordination edges could not be reused
They looked ideal — already hub-capped, repetition-filtered, FDR-validated — but
they are built over *all* posts, so they mean "these two coordinate", not "these
two co-amplify toxic content". Measured three ways:

| approach | cohort | co-amplification signal |
|---|---|---|
| substitute validated edges | 500+ | wrong question; only 2 of top 20 shared |
| intersect with them | 32 | **dead** — 0 for every account |
| local toxic projection | 50 | intact |

The local projection ships. `coordination/` keeps its correct role: cluster
promotion in `adaptive`.

### Observation effort could buy rank
`n_repeat_peers` and `n_brigade_convs` are raw counts carrying 0.45 of the weight,
so expanding an account inflated its own next score. Both are now ranked *within*
observation-volume strata (`ntile(4)` over `n_posts`) — collecting more of a
timeline moves an account to a higher-volume bucket rather than up the ranking.
At n=217 the buckets are 55/54/54/54.

### The hub cap had stopped filtering entirely
`hub_cap = max(50, 5% of accounts)`. The percentage term was meant to lift the cap
on small corpora, but it scales *with* the corpus: at 64,002 amplifying accounts
it gave 3,200, while the busiest object had 1,171 amplifiers. It excluded nothing.

Measured 2026-08-01 on the 14-day window: **1.9% of objects - the viral tail -
generated 99.3% of all 36.8M projected pairs**. That OOMed tf1 and did exactly
what the cap exists to prevent; `coordination.py`'s own docstring warns "one such
hub dilutes the aggregate null rate until real clusters vanish".

Bounded above (`HUB_CAP_MAX=100`). The effect is better detection, not just lower
memory:

| | vacuous cap (3,200) | bounded (100) |
|---|---|---|
| clusters | 1,078 | 298 |
| corroborated | 12 (47 accounts) | **14 (60 accounts)** |
| corroboration rate | 1.1% | **4.7%** |

~780 spurious single-channel clusters disappeared *and* more real clusters
survived the significance test. The same latent flaw was fixed in `hate_signal`,
where the rule had not yet degraded but would have.

Process note: this took four attempts. Three wrong fixes (a DuckDB memory cap, a
spilling hypothesis disproved by my own test, a time window that halved the wrong
thing) preceded the one measurement - degree distribution against pair count -
that found it. That query should have come first.

### Coordination is now time-bounded
`_latest_posts_cte` had no time bound, so every run processed the whole corpus.
Now scoped to `COORD_LOOKBACK_DAYS=14`, matching `MAX_AGE_DAYS` (collection) and
`HATE_LOOKBACK_DAYS` (seeding). Persisted clusters now mean "coordinated in the
last 14 days" rather than "ever" - a change in meaning worth knowing when
comparing against runs before 2026-08-01.

### An X query-precedence bug
X binds implicit AND tighter than OR, so `fukuza OR wafukuzwe (Kenya OR ...)`
parsed as `fukuza OR (wafukuzwe AND Kenya...)` — the leading alternative escaping
the anchors, worst on exactly the high-false-positive terms that most need
scoping. `build_query` now parenthesises multi-alternative keywords first.

### Term mining returned noise on its first run
Ranking by novelty alone filled the candidate list with terms the cohort barely
used (`since`, `trying`, `haha`) whose few occurrences merely happened to be
recent — nearly all with lift < 1, i.e. *under*-represented. Now lift gates and
novelty ranks. A second run surfaced three usernames among eight candidates, so
handles, hashtags and URLs are stripped before tokenising.

---

## 4. The outage, and what it actually was

pi0 was reported down ~6 days. It was not a downtime problem.

The collector was **up, with 54 active accounts, collecting zero** — every request
failing on `Couldn't get XClientTxId indices script`. That is
[twscrape #320](https://github.com/vladkens/twscrape/issues/320) (2026-07-20): X
stopped serving the transaction-ID script to logged-out requests. Fixed upstream
in **0.19.2**; pi0 ran 0.19.1.

**This failure mode is silent** — the pool reports accounts healthy and
`monitor stats` looks normal while volume goes to zero. Documented in
`docs/collection/README.md` with the log signature.

Our local monkeypatch for the older issue #248 was deleted: stock 0.19.2's
`Ctx.req` is byte-identical to it and uses the effective proxy where ours used the
account proxy. It had become a regression.

Recovery, once unblocked: **27,917** backfill posts, **1,972** timelines, **267**
hate-term, 200 metrics. Every partition returned to current.

### A second silent failure
tf1's enrich had been running for three weeks — and `hate_scored` appeared **zero
times** in its entire log. The container predated hate scoring being wired into
the loop, so it only ever ran embed and classify. Hate scoring had not run since
2026-07-28; the stale-score guard had tripped and hate targeting had silently
degraded to generic suspicion ranking.

Caught by noticing `scored` had not moved while the backlog grew 49k -> 59k.
Cleared with a Modal A100 drain (**59,083 posts**) and a tf1 rebuild.

---

## 5. Deliberate design positions

- **The classifier is a locator, not a label.** It cannot see the 2026 coded
  register — on 14 known-coded posts its mean `p_hate` *dropped* (0.070 -> 0.032).
  So term search only has to find a foothold account; account-scoped expansion
  then pulls that account at full recall with no operator and no model.
- **Targeted collection never enters a prevalence denominator.** Scoping is on
  *first-seen* partition, because a post re-collected by a hate pass would
  otherwise flip its `type` and drain the toxic tail out of the baseline.
- **Cluster promotion requires cross-channel corroboration.** 1,066 of 1,078
  clusters rest on a single channel; only 12 (47 accounts) are corroborated.
  Promoting at `n_channels >= 1` would hand the expansion passes nearly the whole
  author base.
- **Mined terms are human-gated.** They would drive searches about ethnic hate;
  auto-promotion risks both false accusation and corpus bias.
- **`hate_index` is reported separately from `inauthenticity_index`.**
  Coordination and hate are different claims; folding them together would make
  the existing index uninterpretable.

---

## 6. Open items

| item | note |
|---|---|
| **`accounts.txt` is committed to git** | 53 `username:password:email` lines, now cloned to pi0 and tf1, and in the history. Needs a decision. |
| **`HF_TOKEN` missing on tf1** | The hate pass fails on the private model repo, so no post collected after the Modal drain is being scored. The stale-score guard degrades hate targeting to suspicion ranking within 24h. **Blocking.** |
| **Coordination retry backoff** | `_coordination_pass` waits the full 6h after a failure rather than retrying. pi0's attempt died on an R2 HTTP timeout and then idled for hours. |
| **`HATE_SEED_WEIGHTS` unvalidated** | The 0.20 on `brigade` is live for the first time. Re-examining the weights properly needs labelled seeds. |
| **Anchoring may cost too much recall** | In the 14-day sweep, 92% of yield came from four *unanchored* low-risk terms; the anchored high-risk terms returned almost nothing and ethnonym x menace returned zero. |
| **Neither host is a git repo** | Both pi0 and tf1 now run from fresh clones alongside the old hand-copied trees; the originals remain as rollback. |
| **No cron on pi0** | `backfill`, `adapt`, `crawl-follows` remain manual. |
| **Frontier not yet exercised live** | Built and unit-tested; takes effect on the next expansion pass. |

---

## 7. Verification trail

- 112 collector tests, 250 analysis tests.
- The measure-column backfill reproduced every banked baseline **exactly**
  (n=134,579; kenya/ambiguous/offdomain 75,910/55,286/3,383; explicit_toxic 5,726
  = 4.364%; hate_flag 601/40; coded_suspect 31; lexicon 94) — proving zero drift
  between persisted and read-time computation.
- `monitor hate-seek --dry-run` renders every anchor tier offline with no network
  calls; tests assert `lang:`, `geocode:`, `near:`, `from:`, `to:` are never
  emitted.
- `monitor adapt --dry-run` promotes coordination-cluster accounts, confirming the
  analysis -> collection handoff that was dead before this work.

# How the monitor works

End-to-end walkthrough of the Kenya 2027 election monitor: what runs where, what
each stage produces, and the constraints that make the output trustworthy. Start
here; the per-subsystem docs go deeper.

---

## 1. The shape of the thing

```
                    pi0 (arm64, residential IP)
  ┌──────────────────────────────────────────────────────────┐
  │  monitor run  - one asyncio scheduler, cycles forever    │
  │                                                          │
  │   posts ─► snowball ─► [hate_seek ─► hate_expand] ─►     │
  │   metrics ─► follow_crawl ─► cooldown ─► repeat          │
  └──────────────────────────────────────────────────────────┘
                              │  immutable Parquet
                              ▼
                    Cloudflare R2  (the only shared state)
                              ▲
                              │
  ┌──────────────────────────────────────────────────────────┐
  │  kma-enrich --loop                          tf1 (x86_64) │
  │   embed ─► classify ─► hate-score ─► [coordination/6h]   │
  └──────────────────────────────────────────────────────────┘
```

Nothing talks to anything else directly. Both hosts read and write R2, and that
decoupling is what lets either restart, fall behind, or be rebuilt without
coordination. Modal (A100) is used on demand when scoring falls badly behind.

---

## 2. Collection (pi0)

`monitor run` loops these steps, each failure-isolated so one bad step never
kills the daemon:

| step | what it does | writes |
|---|---|---|
| **posts** | keyword search over daily windows + tracked-account timelines | `posts/type=search`, `type=timeline` |
| **snowball** | full retweeter census and reply threads for hot objects | `engagements/`, `posts/type=replies`, `type=hydrated` |
| **hate_seek** | coded-term register search (every Nth cycle) | `posts/type=hate_search`, `hate_target_search` |
| **hate_expand** | timelines + toxic-object snowball around hate seeds | `posts/type=hate_timeline`, `hate_replies` |
| **metrics** | re-fetch engagement on recent top posts | `metrics/` |
| **follow_crawl** | BFS over the follow graph from suspicious/hate seeds | `follows/` |

Two facts shape everything downstream:

- **X search reaches back 14 days** (`MAX_AGE_DAYS`). Anything missed inside that
  window is recoverable; anything older is gone permanently. This is why an
  outage is urgent even when nothing is broken.
- **Retweeter census is a census, not a sample.** For the objects it covers, the
  account x object incidence is near-total - which is what makes the statistical
  null used in coordination detection valid.

Throughput is governed by a pool of ~54 X accounts with per-account pacing, not
by wall-clock scheduling. Cycles run back to back.

### Adaptive targeting
Targets are not static. `adaptive.promote` adds bursting hashtags, flagged-story
terms and coordination-cluster members to the live target set, capped and expiring
after 7 days. This is the loop closing: analysis output becomes collection input.

---

## 3. Enrichment (tf1)

`kma-enrich --loop` runs three model passes, each in its own subprocess so their
weights never coexist in memory:

| pass | model | writes |
|---|---|---|
| embed | `paraphrase-multilingual-mpnet-base-v2` | `embeddings/` |
| classify | sentiment / emotion / stance | `labels/` |
| hate | fine-tuned afro-xlmr, 3-class | `hatespeech/` |

Every 6 hours it also rebuilds and persists coordination clusters.

All passes are incremental - they anti-join against what is already scored - so
falling behind is self-healing, and a large backlog can be drained on GPU via
`modal_backfill.py` instead.

### The hate classifier is a locator, not a label
Measured: on 14 known-coded 2026 posts, mean `p_hate` *dropped* against baseline
(0.070 → 0.032). It detects explicit toxicity well and the current coded register
badly. So it is never asked to recognise coded speech - it only has to find a
foothold account, after which account-scoped expansion pulls that account at full
recall with no model involved. The coded register is found separately, by mining
the vocabulary of accounts the classifier *did* surface.

---

## 4. Analysis

| module | question it answers |
|---|---|
| `authenticity` | does this account behave like a bot? |
| `semantic` | what is this post about? (embeddings, topics) |
| `incitement` | does this use documented coded-incitement language? |
| `hatespeech` | is this explicitly toxic, and is it Kenya-scoped? |
| `coordination` | which accounts act together more than chance allows? |
| `stories` | what claims are circulating, and which look seeded? |
| `measure` | is this even about Kenya? |

### Coordination detection, briefly
Behavioural traces (who acted on what, when) are projected into an
account × account graph, then filtered by a **degree-corrected null**: how
surprising is this overlap given how active both accounts are? Survivors are
FDR-corrected at q=0.01, clustered with Leiden across channels, and scored for
inauthenticity.

Only `co_retweet` and `co_reply` are used. `text_sim` and `fast_co_share`
validated **zero edges** on the full corpus and are excluded rather than left in
as reassuring noise.

Cross-channel corroboration is the strongest available evidence: of 298 clusters,
14 are corroborated across both channels. Single-channel clusters are candidates,
not findings.

---

## 5. The three things that keep this honest

### Sampling provenance
Targeted collection - hate-seeking search, seed expansion, coordination
promotion - deliberately oversamples the toxic tail. If those posts entered the
same partition as baseline collection, every prevalence and trend would be
inflated and uninterpretable.

So `posts/type=` encodes provenance, split into `BASELINE_TYPES` and
`TARGETED_TYPES`, and any rate uses `latest_posts(con, scope="baseline")`.
An unrecognised partition **raises** rather than defaulting into a bucket.

Crucially, scoping is on **first-seen** type. `latest_*` keeps the newest
`collected_at`, so a post first found by baseline search and later re-collected by
a hate pass would have its `type` flipped - and it is by construction a post the
hate query matched. Scoping on the latest row would drain exactly the toxic tail
out of the baseline denominator, deflating measured toxicity in a way that reads
as a real downward trend.

### Feedback isolation
Expanding an account collects more of its activity, which inflates its own future
coordination scores. Two guards: promoted accounts write to quarantined
partitions, and the count-based components of the seed score are ranked *within*
observation-volume strata, so collecting more of a timeline moves an account to a
different bucket rather than up the ranking.

### Triage, never verdict
Nothing here concludes that an account is a bot or a post is hate speech. The
coded-term register is documented NCIC/PeaceTech advisory language whose entries
mostly have innocent everyday senses (`mende` is a cockroach, `nyoka` a snake).
Machine-mined terms are human-gated before they can drive a search. Cluster
scorecards report an inauthenticity index *and* a separate hate index, because
coordination and hate are different claims and merging them makes both
uninterpretable.

---

## 6. Operating it

```bash
monitor stats                      # collection volume by partition
monitor hate-seek --dry-run        # render queries, no requests
monitor adapt --dry-run            # what would be promoted
monitor hate-mine --cohort 30      # coded-term candidates for review
python -m kma.enrich --once        # one enrichment pass
python -m kma.coordination_run --persist
modal run --detach modal_backfill.py   # GPU drain when scoring falls behind
```

Scheduled on pi0 via cron: a dense backfill every 3 days, coded-term mining
weekly. Everything else runs inside the always-on scheduler.

### Failure modes worth recognising

- **Collection volume goes to zero while everything looks healthy.** X changes how
  `x-client-transaction-id` is derived and a stale twscrape fails every request
  while still reporting accounts as active. Check the twscrape changelog first.
- **Hate targeting quietly degrades.** If `hatespeech/` goes stale past 24h, seed
  selection falls back to generic suspicion ranking. It logs, but the collector
  keeps running and volume looks normal.
- **A subsystem runs but does nothing.** Both happened here: an enrich container
  that had never been wired for the hate pass, and a hub cap whose 5%-of-accounts
  term grew with the corpus until it excluded nothing. Neither raised an error.
  Watch the *outputs*, not just the process status.

---

## Where to go next

| doc | for |
|---|---|
| `docs/collection/README.md` | collector internals, env knobs |
| `docs/collection/hate-seeking.md` | the coded-term register and targeting |
| `docs/collection/cib-collection.md` | why census-style collection matters |
| `docs/analysis/data-model.md` | every R2 prefix and column |
| `docs/analysis/phase-3-coordination.md` | the statistical method in full |
| `docs/status/` | what is currently working, and what is not |

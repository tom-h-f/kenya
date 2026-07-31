# Hate-seeking collection

How the monitor actively looks for Kenyan coded incitement, and the constraints
that keep the resulting corpus measurable and the method defensible.

## Why the classifier is not the search

The fine-tuned afro-xlmr classifier (`kma.hatespeech`, `p_hate >= 0.28`) is a
**locator, not a label**, and it has one measured blind spot that shapes this
whole design: on 14 known-flagged 2026 coded posts its mean `p_hate` *dropped*
against baseline (0.070 -> 0.032), and Opus gold on those posts was 0 `hate`.
It detects explicit toxicity well (5,726 explicit-toxic posts, 4.36% of Kenya
scope) and the current coded register badly (`coded_suspect` fires on 31 posts
corpus-wide).

So term search is never asked to *recognise* coded speech. It only has to find
a foothold account. After that the account-scoped passes - timelines,
retweeter census, reply threads, follow crawl - pull that account at full
recall with no search operator and no model involved.

## Acquisition

Three sources share one rotated budget (`kenya_monitor.hate_seek`):

| Source | Register | Partition |
|---|---|---|
| `lexicon` | `config/hate_terms.yaml`, mirroring `kma.incitement.LEXICON` | `hate_search` |
| `mined` | machine-derived candidates, human-gated | `hate_search` |
| `targets-of-hate` | ethnonym x menace conjunctions | `hate_target_search` |

**The register is not a list of verdicts.** Every term has an innocent everyday
sense (`mende` is a cockroach, `nyoka` a snake, `fukuza` the ordinary verb
"chase away"). A hit means a human should read the post. The terms come from
NCIC advisories and the PeaceTech Lab Kenya lexicon.

### Anchoring, not `lang:`

The platform `lang` field is useless for Swahili here - roughly 0% of Swahili
posts are tagged `sw` (measured, 2026-07-17 manipulation sweep) - so `lang:sw`
is never emitted. Scoping is done by conjoining a Kenya anchor group, sized by
each term's false-positive risk:

| `fp_risk` | anchors | why |
|---|---|---|
| `low` | none | `madoadoa`, `sangari`, `watajua hawajui` are near-unique; anchoring would cost the recall that matters most |
| `medium` | `anchors.core` (6) | modest scoping |
| `high` | `anchors.wide` (12) | `mende`, `nyoka`, `fukuza` are everyday Swahili; unanchored they return a region-wide noise firehose |

Accepted recall cost: anchoring means we only see coded posts that *also* name
a Kenyan target, and coded speech exists precisely to avoid naming targets.
That is fine, because the query only has to find the account.

`build_query` parenthesises multi-alternative keywords before appending
anything. X binds the implicit AND tighter than OR, so `fukuza OR wafukuzwe
(Kenya OR ...)` would otherwise mean `fukuza OR (wafukuzwe AND Kenya OR ...)` -
the leading alternative escaping every filter, on exactly the terms that most
need scoping.

Never emitted: `lang:`, `geocode:`, `near:`, `from:`, `to:`. Enforced by
`collector/tests/test_hate_seek.py`.

### Ethnonyms are never searched bare

A bare ethnonym returns ordinary community discussion. Only the menace
conjunction is emitted - `Kikuyu ("rudi kwao" OR waondoke OR ...)` - so the
query is about threat rather than about a community. These land in their own
`hate_target_search` partition because searching ethnonyms at all biases the
corpus toward ethnicised discourse and would corrupt any "how ethnicised is the
discourse" statistic computed on it.

## Budget

Hate-seeking is a **separate cycle step with its own semaphore**, never merged
into `PlatformTargets.keywords`. Merging would put it under the same
`COLLECT_CONCURRENCY=3` as the 35 baseline keywords (starving baseline
coverage) and write it into the `search` partition (destroying separability).

```
HATE_SEEK_ENABLED=0          # off until verified; then 1
HATE_SEEK_MAX_TERMS=8        # per pass, rotated least-recently-searched first
HATE_TARGET_MAX_TERMS=3      # reserved ethnonym slots
HATE_MINE_MAX_TERMS=5        # reserved mined-term slots
HATE_SEEK_WINDOW_LIMIT=15
HATE_SEEK_EVERY_N_CYCLES=3
HATE_SEEK_CONCURRENCY=1
```

Baseline is 35 keywords x 2 windows = 70 searches/cycle. Hate-seek at 8 terms x
2 windows every third cycle is ~8 searches/cycle amortised, roughly +8%.
Rotation state (`state/hate_seek.json`) sweeps the whole register across passes
without raising per-pass cost.

The step is ordered **after** baseline posts and snowball, so when the account
pool hits a rate-limit wall it hits discretionary work first.

## Sampling bias is the central constraint

Actively searching for hate speech oversamples the toxic tail by construction.
If those posts entered the same partition as baseline collection, every
prevalence, trend and rate in `notebooks/hatespeech.py` would be inflated and
uninterpretable.

Two rules keep this honest, both enforced in `kma.db`:

1. **Targeted partitions are excluded from prevalence.** `BASELINE_TYPES` vs
   `TARGETED_TYPES`; measurements use `latest_posts(con, scope="baseline")`.
   An unrecognised partition raises rather than defaulting into a bucket.
2. **Scope on first-seen type, never the latest row's.** `latest_*` keeps the
   newest `collected_at`, so a post first found by a baseline search and later
   re-collected by a hate pass would have its `type` flipped - and it is by
   construction a post the hate query matched. Scoping on the latest row would
   drain exactly the toxic tail out of the baseline denominator, deflating
   measured toxicity over time in a way that reads as a real downward trend.

See `docs/analysis/data-model.md` and `analysis/tests/test_scope.py`.

## Audit trail

Every targeted pass appends its **rendered** queries to
`collection_runs/platform=x/dt=/run=.parquet`:

```
run_id, target_type, term, source, fp_risk, query, since, until, n_posts, collected_at
```

`posts.source_query` keeps only the bare term (`hate:mende`), so this is the
only record of what was actually asked of the platform and when. Treat it as a
permanent, publishable artifact: it is the difference between "we monitored
hate speech" and "here is exactly what we searched for, on which day, and why
that term is in the register".

## Targeting: from per-post scores to network seeds

`kenya_monitor.hate_signal` ranks accounts as hate-network seeds in pure SQL,
reading the `hatespeech/` prefix the analysis side already wrote. The collector
runs no models.

**One toxic post is not a network.** But the floors that enforced that were
originally ANDed, which quietly broke the thing they were protecting. Measured
2026-07-31: `n_posts >= 10 AND n_toxic >= 3` cut 37,211 authors to 29, and the
peer floor to 8 - discarding **936 of 939 brigade participants**, so
`n_brigade_convs` was exactly 0 for every seed and the `brigade` component
(weight 0.20) contributed nothing at all. Brigading is many accounts each posting
once; a sustained-volume floor removes brigaders by construction.

Qualification is therefore **permissive across three independent routes**, and
*ranking* is what stays selective:

| route | qualifies on | catches |
|---|---|---|
| volume | `n_posts >= 10 AND n_toxic >= 3` | sustained solo toxic poster |
| co-amplification | `n_repeat_peers >= 2` | retweet ring member |
| brigade | `n_brigade_convs >= 2` | repeat pile-on participant |

A lone ranter still qualifies on volume - it is worth watching - but with no
co-amplification and no brigade it forfeits 0.45 of the weight and never reaches
the top of the list. Putting coordination in the floor instead is what caused the
bug above.

### Co-amplification needs a hub cap and a repetition bar

Measured on the same corpus: of 385 toxic objects, **9 had more than 50 distinct
amplifiers (max 223), and those 9 alone produced 43% of every co-amplification
row**. Without guards, "peers" mostly means "we both retweeted the same viral
post", which is organic. With a hub cap and a repetition requirement, 1,559
accounts clearing `peers >= 2` collapse to 29 that repeatedly co-amplify with the
same partner. `HUB_CAP_MIN`/`HUB_CAP_PCT`/`AMP_MIN_REPETITION` mirror
`kma.coordination`.

**Persisted `coordination/` edges are deliberately not reused here.** They looked
like the obvious answer - already hub-capped, repetition-filtered and
FDR-validated - but they are built over *all* posts, so they mean "these two
coordinate", not "these two co-amplify toxic content". Measured three ways:
substituting them gave a 500+ cohort sharing only 2 of the top 20 with the toxic
projection; intersecting gave `n_repeat_peers = 0` for every account. The local
toxic-scoped projection is what ships. `coordination/` keeps its proper role -
cluster-membership promotion in `adaptive`.

### Observation effort must not buy rank

`n_repeat_peers` and `n_brigade_convs` are raw counts, so expanding an account
collects more of its activity and inflates its own next score. Those two
components are therefore ranked **within observation-volume strata**
(`ntile(4)` over `n_posts`): pulling more of a timeline moves an account to a
higher-volume bucket rather than up the ranking. At the current cohort of 50 the
buckets come out 13/13/12/12. Bounded components (`toxic_rate_lb`,
`persistence`, `suspicion`) rank globally.

Below `MIN_COHORT_FOR_RANK` accounts, `percent_rank` is not a ranking (0 for one
row, 1/(n-1) steps): the score is suppressed to `NULL` and the evidence reported
instead.

Co-amplification unions collected retweets with the snowballed retweeter census
(`engagements/`), so it does not depend on our having happened to collect each
retweet as a post. Components are percentile-ranked (counts log1p'd first) so
one heavy tail cannot dominate.

If the enrich worker stops writing, `scores_are_stale` trips at 24h and every
caller falls back to the generic suspicion ranking rather than silently
freezing the seed set on stale scores.

### Expansion, and why seeds stay quarantined

Seeds drive timelines (`hate_timeline`), a toxic-object snowball
(`hate_replies` / `hate_hydrated`), and `--top-hate` seeding of
`monitor follows` / `monitor crawl-follows`. Timelines are the highest-value
pass: they need no search operator and no model, so they recover the coded
register the classifier structurally cannot see.

Seeds are **never** added to `targets.accounts`. Promoting them into baseline
collection would inflate their post counts and co-action degree, so they would
look more coordinated on the next run purely because we watched them harder.
The degree-corrected null in `kma.coordination` corrects for object popularity,
not for author over-sampling - it cannot save us from that.

The same reasoning caps cluster promotion. Measured on the live corpus
(2026-07-28): 906 clusters / 4,732 accounts, of which **894 clusters (4,674
accounts) rest on a single channel and only 12 clusters (58 accounts) are
corroborated across `co_retweet` and `co_reply`**. Promoting at
`n_channels >= 1` would hand nearly the whole active author base to the
expansion passes, which is not a signal. `CLUSTER_MIN_CHANNELS=2` is the floor;
`coordination/` still archives everything, and consumption filters.

## Mining the coded register

`hate_signal.mine_terms` contrasts the seed cohort's vocabulary against the
whole corpus. This is how the 2026 register gets found at all: the cohort is
located by explicit toxicity and coordination, then its *language* is mined.

Ranked by **novelty** - the rise in cohort share over the last 3 days versus
the preceding window - rather than raw log-odds lift, because new coded terms
appear suddenly, which is far more specific than a term merely being
cohort-flavoured. Lift is reported alongside for review.

Floors: `c_in >= 8` occurrences and **`a_in >= 3` distinct authors**. The
author floor is the one that matters - one account repeating a word is a verbal
tic, not a register.

## The closed loop

```
collector  hate-seeking search + seed expansion  ->  posts/type=hate_*
analysis   kma.enrich (hate + measure columns)   ->  hatespeech/
analysis   kma.coordination_run --persist        ->  coordination/
collector  hate_signal + adaptive.promote        ->  next pass's targets
```

`hatespeech/` carries `domain`, `in_kenya_scope`, `lexicon_hits`,
`coded_suspect`, `explicit_toxic` and `flagged` as persisted columns, so
`hate_signal` scopes to Kenya in pure SQL without importing `kma` or
re-implementing the domain regex. Rows written before the rollout are brought
up to contract with `python -m kma.hatespeech --refresh-measure` (CPU only -
the model never re-runs).

Cluster scorecards carry `toxic_share`, `coded_share`, `n_toxic_authors`,
`toxic_concentration` and a separate `hate_index`. `hate_index` is deliberately
**not** folded into `inauthenticity_index`: coordination and hate are different
claims, and a cluster can be plainly coordinated and entirely non-toxic. Triage
is 2-D. `toxic_concentration` is the Herfindahl index of flagged posts across
members - the cluster-level restatement of "one post is not a network", since a
cluster where one member carries all the toxicity scores near 1 and ranks down.

## Governance of mined terms

Mined terms are machine-derived candidates that would drive searches about
ethnic hate. Auto-promoting one risks both false accusation and corpus bias, so
`HATE_MINE_AUTOPROMOTE=0` by default: candidates are written to
`state/mined_terms.json` and logged, `monitor hate-mine --dry-run` shows each
with counts and example posts, and a human moves terms into
`hate_terms.yaml`. When enabled, mined terms always carry `fp_risk: high` and
therefore the widest anchoring available.

## Keeping the two registers in sync

`config/hate_terms.yaml` mirrors `kma.incitement.LEXICON` as **data**. The
collector deliberately carries no `kma` dependency and no ML dependencies - the
same precedent as `suspicion.py` mirroring `kma.authenticity` in pure SQL. Add
a term in one place and you must add it in the other;
`collector/tests/test_hate_seek.py::test_register_mirrors_the_analysis_lexicon`
is the contract that fails otherwise.

## Commands

```bash
monitor hate-seek --dry-run                    # print rendered queries, no requests
monitor hate-seek --dry-run --terms mende      # inspect one term
monitor hate-seek --terms madoadoa --limit 10  # small-scale live run
monitor stats                                  # per-partition post counts
```

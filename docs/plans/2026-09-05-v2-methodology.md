# v2 methodology: adopt Luceri et al. (WWW 2024) wholesale

Decided 2026-09-05. This document supersedes the coordination-detection method
described in `docs/analysis/phase-3/` for all new work. Objectives it serves are
in [../OBJECTIVES.md](../OBJECTIVES.md).

---

## 1. The decision

Adopt, without modification and without blending, the coordination-detection
methodology of:

> Luca Luceri, Valeria Pantè, Keith Burghardt, Emilio Ferrara. **"Unmasking the
> Web of Deceit: Uncovering Coordinated Activity to Expose Information
> Operations on Twitter."** Proceedings of the ACM Web Conference (WWW) 2024.
> arXiv:2310.09884. https://arxiv.org/abs/2310.09884

including its evaluation protocol. Where the paper and the existing pipeline
disagree, the paper wins. Parameters are copied, not re-derived, until the
benchmark is reproduced; only then are they tuned.

### Why this paper and not the alternatives

The chosen north-star metric is **recall of accounts against X's published
information-operations archive**. That decides it: this is the only candidate
methodology whose published evaluation *is* that metric.

| Candidate | What it is | Why not chosen |
|---|---|---|
| **CooRnet / CooRTweet** (Giglietto; Righetti & Balluff, *Computational Communication Research* 2025) | Generalized coordinated-behaviour detection: `time_window` + `min_repetition` over any action type. Heavily replicated on Italian and European elections. | Validated by face validity and case study, never against labelled operations. Cannot be scored on the north star. CrowdTangle's shutdown also archived the original CooRnet. |
| **Pacheco et al., "Uncovering Coordinated Networks on Social Media"** | Single-trace similarity networks + thresholding. | This is what v1 already derives from. No labelled evaluation; it is the thing being replaced. |
| **Elmas et al., "Ephemeral Astroturfing Attacks"** (IEEE EuroS&P 2021) | Detects paid trend-seeding: accounts post keyword-bearing tweets then delete them to push a trend. | Closest to Kenya's *documented* attack pattern (see §7), but it detects one specific attack and needs trend snapshots plus deletion checks that we do not collect. Recorded as a separate future collection experiment - deliberately **not** merged into v2. |
| **IOHunter** (Luceri group, AAAI 2025) | GNN foundation model over the same fused similarity network. | Same data prerequisites, plus a training burden. Adopting the 2024 paper first makes IOHunter a drop-in successor later. |

## 2. What v2 keeps and what it discards

**Keeps** (data layer and everything that is not coordination detection): the R2
corpus and `kma.db` read patterns, the baseline/targeted provenance split,
`kma.measure` relevance, `kma.authenticity`, `kma.hatespeech`, `kma.incitement`,
`kma.stories`, and the dossier/adjudication layer.

**Discards:**

- The degree-corrected configuration-model null, and Bonferroni/FDR correction,
  *as the detector*. v2 does not test edges for significance at all.
- Leiden clustering as the detection step. Community detection becomes optional
  descriptive grouping, applied after accounts are classified. This dissolves the
  frozen-resolution problem entirely: there is no resolution parameter in the
  detection path.
- The hub cap as a detector-side filter. TF-IDF weighting is the paper's
  mechanism for handling popular entities: they are down-weighted, not deleted.
- Cross-channel corroboration (`n_channels >= 2`) as the evidence tier. Fusion
  across five traces plus centrality replaces it.
- `min_repetition`.

**A prediction that follows from this, worth writing down before it is tested:**
`text_sim` and `fast_co_share` were dropped from v1 because they validated *zero
edges* on the full corpus. They were killed by the significance null, not by an
absence of signal. Under this method they are built unthresholded and contribute
through node centrality, so they should come back. If they still produce nothing,
something is wrong with the trace construction, not with the traces.

## 3. The v2 specification

### 3.1 Behavioural traces

Five traces. For each, build a **bipartite user x entity graph weighted by
TF-IDF** over entity popularity, then project to a user x user similarity network
whose edge weight is the **cosine similarity of the users' TF-IDF vectors**.

| Trace | Entity | Our source | Status |
|---|---|---|---|
| Co-Retweet | the retweeted tweet | `engagements/` incidence + `posts` `is_repost` / `repost_of_id` | have |
| Co-URL | URL in the tweet | `posts.urls` (list<string>) | have, never built |
| Hashtag Sequence | the ordered hashtag sequence, min 3 tags | `posts.hashtags` (list<string>) | have; **verify order is the in-tweet order**, not a set |
| Fast Retweet | the retweeted *author*, when the retweet lands within the time threshold | RT rows from search carry `created_at`; `engagements/` is incidence-only with no timestamp | partial - see §5 |
| Text Similarity | n/a - direct similarity network | `embeddings/` | have, wrong model - see §5 |

Text Similarity is the exception to the recipe: no bipartite stage. Retweets are
excluded; text is stripped of punctuation, stopwords, emoji and URLs; only tweets
with **at least four words** count; embeddings come from a sentence transformer
(the paper uses `stsb-xlm-r-multilingual`), cosine similarity via FAISS within a
**one-year sliding window**; two users are linked if they post at least one pair
of similar tweets, and the edge weight is the **average** text similarity.

### 3.2 Parameters (copy these)

The paper's optimized values, which beat the values used by prior work:

| Trace | Prior-work value | Optimized value | Prior AUC | Optimized AUC |
|---|---|---|---|---|
| Fast Retweet | 10s | **60s** (50th pct) | 0.53 +/- 0.03 | 0.62 +/- 0.13 |
| Co-Retweet | 99.5th pct | **80th pct** | 0.55 +/- 0.03 | 0.69 +/- 0.09 |
| Co-URL | 99.5th pct | **80th pct** | 0.61 +/- 0.04 | 0.72 +/- 0.09 |
| Hashtag Sequence | 5 hashtags | **3 hashtags** (65th pct) | 0.59 +/- 0.07 | 0.68 +/- 0.16 |
| Text Similarity | 0.7 | **0.95** (96th pct) | 0.47 +/- 0.04 | 0.52 +/- 0.05 |

### 3.3 Detection: unsupervised

1. **Fuse.** Two users are linked in the fused network if they are linked in
   **any** individual similarity network. (The paper tested weight aggregation and
   max-centrality fusion; plain union won.)
2. **Score.** Unweighted **eigenvector centrality** on the fused network. Nodes
   absent from a network get centrality 0. Weighted centrality did **not**
   improve performance.
3. **Prune.** Drop nodes below a centrality threshold. Conservative operating
   point from the paper: **10^-2**, giving precision > 99% at average AUC > 0.70.

Expected performance to reproduce: **fused unsupervised AUC 0.83, F1 0.76**, and
node pruning beating edge filtering on precision by an average of **+0.42** at
comparable recall.

### 3.4 Detection: supervised

`node2vec` over the fused network: **128 dimensions, 16 walks per node, 16 steps
per walk**, then a Random Forest with 10-fold cross-validation.

Expected: **AUC 0.94, F1 0.82, precision 0.96** per campaign; on the global
multi-campaign task **precision 0.95, recall 0.70, F1 0.78, AUC 0.92**.

Ablation from the paper: Co-Retweet contributes most, Fast Retweet least.

### 3.5 The unit of prediction changes

v1 predicts **clusters**. v2 predicts **accounts** (IO driver vs organic), and
grouping is a post-hoc description. Every downstream consumer that reads clusters
- collector targeting, dashboard, scorecards - has to be re-pointed at accounts
or at post-hoc groups. This is the largest integration cost in the change.

## 4. Phase 1 - rebuild the benchmark first

**Nothing is applied to the Kenya corpus until the paper's numbers are
reproduced.** The benchmark is both the acceptance test for the copy and the
bench that every later variant is scored on.

1. **Acquire the IO archive data.** Six country-level campaigns: China, Cuba,
   Egypt & UAE, Iran, Russia, Venezuela - 49M tweets in the paper. Source: the X
   transparency moderation-research archive, with the Internet Archive mirror
   (`archive.org/details/X_Twitter_Information_Operations`) as fallback. The
   mirror is one `ioa_tweets.csv` of **105.9 GB** covering all 21 countries plus
   a 15.7 MB `ioa_users.csv`; our six are a subset. **Accounts under 5,000
   followers are anonymised in the mirror** - verify before building anything
   that pseudonymous ids are stable (networks survive) and that tweet text is
   retained (the text-similarity trace does not survive otherwise).
2. **Get a control group. Both routes, in parallel** (§4.1).
3. **Implement** the five trace builders, fusion, centrality and the node2vec+RF
   model in a new module (`kma.coord2`), with the paper's parameters hard-coded
   as defaults.
4. **Gate:** reproduce the fused unsupervised AUC/F1 and the supervised
   precision within a stated tolerance across the six campaigns. Failing the
   gate means we copied it wrong; it does not mean the method is wrong.
5. **Run v1 on the same benchmark** for a like-for-like comparison (decided:
   full comparison, not an overlap anchor). Stated in advance so the result is
   not read as a rigged test: v1's degree-corrected null assumes near-total
   census incidence, and the IO archive is not a census, so v1 is expected to
   underperform for reasons that are partly structural rather than purely
   methodological. Report it that way whichever direction it lands.
6. Only then run v2 on the Kenya snapshot.

**Everything in this phase runs on Modal.** tf1 has 3.8 GB of RAM and cannot
hold any stage of it; the laptop cannot hold the 105.9 GB source. Embeddings go
on A100, graph construction and node2vec on large-RAM CPU containers. Use
`--spawn`, never a blocking `.remote()`.

### 4.1 The control group is the real gate

New constraints found 2026-09-05, and they are worse than the plan first assumed:

- The BLOC repo (`anwala/general-language-behavior`) **ships no data**. It tells
  you to fetch drivers from X transparency and build your own control set.
- The control recipe needs Twitter's **academic search API**, which no longer
  exists. The original construction is not reproducible by anyone today.
- Our own collector cannot substitute: X search reaches back 14 days and the
  controls are 2010-2020 accounts.

Both routes run in parallel, because the first one's cost is latency:

1. **Ask the authors** (Nwala; Luceri) for the control set directly. Exact
   replication if it lands, weeks of latency if it does not, and it may not be
   redistributable.
2. **Rebuild from the Internet Archive Twitter Stream Grab** (monthly JSON,
   2011-2022): filter to accounts tweeting the IO drivers' hashtags in the same
   windows. Obtainable today and era-matched, but it is a ~1% spritzer sample,
   so coverage of any specific hashtag is thin, and it is a different sampling
   process from the paper's - results are comparable in spirit, not strictly.

Rejected: using the Kenya corpus as negatives (era and language mismatch make
the classes trivially separable and every score inflates).

**The old numbers are not a baseline.** The 2026-08-15 measurements (recall
18.6%, FPR <= 0.5%, the resolution sweep) came from
`analysis/investigations/2026-08-15-io-ground-truth/`, which exists in **no git
branch and not on tf1**. It is lost. Treat those figures as history, not as a
comparison point, and re-measure v1 on the rebuilt benchmark if a v1-vs-v2
comparison is wanted.

## 5. Data gaps this exposes

| Gap | Detail | Options |
|---|---|---|
| **Fast Retweet timestamps** | `engagements/` records retweeter incidence with **no retweet timestamp**, by design. Only RT-inclusive search rows carry `created_at`. | Build the trace from search RT rows only and record the coverage loss; or add timestamps to the census where the API exposes them. |
| **Embedding model mismatch** | We hold 668k embeddings from `paraphrase-multilingual-mpnet-base-v2`; the paper uses `stsb-xlm-r-multilingual`. A cosine threshold is not portable across embedding spaces. | **Decided: keep our model everywhere, benchmark included, and set the threshold at the 96th percentile of the observed similarity distribution.** The paper's 0.95 *is* that percentile in its own space, so percentile is the parameter and 0.95 is only its value there. Accepted cost: if the text-similarity trace misses the paper's AUC, the encoder and our code are confounded as explanations - so judge the gate on the fused result and the other four traces, and treat TS in isolation as indicative only. |
| **Hashtag order** | The trace needs an ordered sequence. `posts.hashtags` is a list; whether it preserves in-tweet order is unverified. | Verify against raw twscrape entities before building. |
| **Co-URL coverage** | Never built. Unknown how many of our posts carry URLs, and t.co shorteners may need resolving to match the paper's entity definition. | Measure first. |

## 6. Phase 2 - freeze the corpus

Every experiment names an immutable snapshot. Without this, a method comparison
is confounded by collection that happened while it ran.

- Write `bench/snapshot=<date>/manifest.parquet`: every object path in every
  prefix with its size, etag and row count, plus the collector `code_version`
  range it covers.
- A snapshot is a manifest, not a copy, so long as R2 objects are immutable per
  run - which they are.
- Current inventory, measured 2026-09-05 (parquet footer counts, raw rows before
  latest-state dedup):

| prefix | files | rows |
|---|---|---|
| posts/search | 901 | 497,390 |
| posts/timeline | 600 | 457,472 |
| posts/replies | 1,332 | 793,185 |
| posts/hydrated | 593 | 53,961 |
| posts/hate_search | 58 | 1,000 |
| posts/hate_target_search | 4 | 6 |
| posts/hate_timeline | 71 | 42,218 |
| posts/hate_replies | 161 | 107,377 |
| posts/cib_timeline | 262 | 335,859 |
| posts/census_timeline | 212 | 189,643 |
| authors | 4,424 | 6,343,118 |
| engagements | 3,688 | 5,176,353 |
| follows | 2,266 | 1,847,371 |
| metrics | 585 | 304,055 |
| embeddings | 1,438 | 668,294 |
| labels | 1,426 | 668,813 |
| incitement | 323 | 916,191 |
| hatespeech | 818 | 1,807,693 |
| topics | 1 | 663,294 |
| census_runs | 292 | 292 |
| coordination | 1,095 | 17,904,692 |
| stories | - | prefix absent |

Total posts ~2.48M raw rows across all partitions. `stories/` was never
persisted, and `follows/` is thin because the crawl was dead for 27 days.

## 7. Phase 3 - the collection replay harness

A **collection policy** is a function from the state observable at time *t* to a
set of fetch requests. Replay scores a candidate policy against the frozen
snapshot: given only what the policy could have known at *t*, what would it have
fetched, and how much of the v2 signal would that have bought?

- **Score:** coordination-useful yield per request - fused-network edges, and
  (once the supervised model transfers) predicted IO-driver accounts, per API
  request spent.
- **What replay can test:** object and account selection, ranking, banding, TTL,
  and budget allocation across arms - anything that re-orders or re-weights
  fetches over data we already touched.
- **What replay cannot test:** any policy that needs data never collected - a
  different keyword set, a different platform surface, a different search
  product. Those need live parallel arms, which are **out of scope by decision**.
- **The bias to state on every replay result:** the frozen corpus is itself the
  output of the incumbent policy, so replay systematically favours policies that
  resemble the incumbent. The retweeter census is the partial exception, because
  its account x object incidence is near-total for the objects it covers.

## 8. Phase 4 - dashboard

The published series restarts on v2 by decision; nothing carries over. Only
composition-standardised or within-partition rates get published (see
`OBJECTIVES.md` C1/A7).

## 8a. Deviations from the paper, forced by data

Three, all discovered by running the method rather than by reading it. Each is a
departure from a faithful copy and each is recorded here so the copy stays
honest about where it diverges.

| # | Deviation | Why it was forced |
|---|---|---|
| 1 | **Centrality threshold is a swept percentile selected on validation**, not the paper's fixed `1e-2`. | The reference implementation does this too - the paper quotes 1e-2 only as a conservative operating point. An absolute cut cannot port between graphs, because eigenvector centrality is L2-normalised across nodes: typical values run ~0.06 on a 240-node campaign and ~0.001 on a million-author graph. |
| 2 | **Isolated nodes are rewired** to five random non-isolated nodes before centrality. | Also in the reference implementation, and absent from the paper's prose. The check must be neighbour-based, not degree-based: a self-loop contributes 2 to `degree`, so a node whose only edge is to itself reads as connected by degree and isolated by neighbours. |
| 3 | **Minimum-activity floor: users acting on fewer than 2 distinct entities are dropped from a trace** (`coord2.MIN_ENTITIES_PER_USER`). | Not in the paper or the reference at all. See below. |

### Why the activity floor was unavoidable

The paper's stated defence against popular entities is TF-IDF: they are
down-weighted, not deleted. That holds for users with varied activity and fails
completely for users with a single action. Two users whose only action is the
same object have identical one-hot vectors, so their cosine is 1.0 whatever the
IDF weight - the normalisation cancels it.

Measured on snapshot `2026-09-05-promotion-off`, 2026-09-07:

- **54.8%** of `co_retweet` users acted on exactly one entity.
- One viral tweet (`2016031742109348277`) drew **308** of them into a clique,
  212 of which had no other retweet in the corpus.
- Eigenvector centrality collapsed onto it: the **top 100 accounts shared a
  single centrality value**, 1/sqrt(308) = 0.056946.

| floor | fused nodes | edges | components | largest | distinct values in top 100 |
|---|---|---|---|---|---|
| 1 | 28,661 | 583,348 | 1,828 | 18,299 | **1** |
| 2 | 12,186 | 373,788 | 245 | 11,379 | **90** |
| 3 | 8,630 | 291,407 | 180 | 8,122 | 100 |

A floor of 2 makes the ranking usable and consolidates the graph; the top
accounts are stable between floors 2 and 3, so it is finding structure rather
than reshuffling noise. It costs 58% of accounts, all of which are incapable of
expressing coordination under any method.

**This is a partial vindication of v1.** Its hub cap defended against exactly
this, with its own measured justification (420 of 422 censused objects were hubs
carrying no coordination signal), and v2 discarded it on the paper's authority.
CooRTweet spells the same idea `min_repetition`.

**Consequence for the benchmark:** the IOHunter gate ships PREBUILT trace
networks, so it never exercised this path and the gate's 5/6 pass says nothing
about it. Trace construction remains the untested half of the copy.

## 9. Risks, stated plainly

1. **Actor-type mismatch.** The IO archive is state-backed operations. Kenya's
   documented problem is domestic disinformation-for-hire: influencers paid
   US$10-15 per campaign, coordinated in WhatsApp groups, pushing hashtags into
   the trending list - 11 campaigns, 23,000 tweets, 3,742 accounts in Madung's
   2021-22 Mozilla work. Optimising recall on the archive optimises for a
   different adversary. This is the single largest methodological risk in the
   plan, it follows directly from the chosen north star, and the mitigation is
   *not* to blend methods: it is to keep Elmas-style trend forensics as a
   separate, separately-evaluated track.
2. **Temporal transfer.** The archive covers 2010-2020 X. We are detecting on
   2026-27 X, with different platform mechanics and a different retweet surface.
3. **Cost.** Fused similarity networks over ~2.5M posts and ~1.2M authors on
   pi0/tf1-class hardware. Text similarity is quadratic, mitigated by FAISS and
   the sliding window. **Measure it; do not estimate it**, and measure on the
   host that will run it.
4. **The paper's own limitation, inherited:** control users were selected by a
   different process from IO drivers, so some of the separation the model learns
   may be an artifact of how the control group was built rather than of
   coordination.

## 10. Decisions taken 2026-09-05

| # | Question | Decision |
|---|---|---|
| 1 | Compare v1 against v2, or just switch? | **Full comparison on the benchmark.** Caveat recorded in §4 step 5. |
| 2 | Text-similarity encoder | **Keep our model**, threshold as the 96th percentile per space. |
| 3 | Control group | **Both routes:** ask the authors and rebuild from the Stream Grab, in parallel. |
| 4 | Where the benchmark runs | **Modal for everything.** |
| 5 | Collector targeting in transition | **Turn cluster promotion off.** See §11. |
| 6 | tf1-only code | **Recovered** to branch `recover/tf1-deployed-state`, pushed 2026-09-05. |

## 11. Transition: cluster promotion goes off

Decided: stop promoting coordination-cluster members to timeline targets while
v2 is built, rather than freezing on stale clusters or keeping v1 alive purely
as a targeting service.

**What this costs, stated so it is not discovered later.** Promotion is what
puts suspected coordinated accounts into `posts/type=cib_timeline` - 335,859
rows to date. Turning it off means:

- no new dense sampling of suspected coordinated accounts, so the pool of
  objects the retweeter census can band and enumerate thins over time;
- the loss is **unrecoverable** past the 14-day search horizon, unlike an
  analysis change, which can always be re-run on stored data;
- v2's own training and evaluation on the Kenya snapshot will be done against a
  corpus whose densification stopped at the switch-off date, so the snapshot
  before and after that date are not the same population. Split on it.

What is *not* affected: hashtag-burst promotion, hate seeding and expansion, the
retweeter and conversation censuses, and baseline search - all select
independently of clusters.

Re-enable by pointing promotion at v2's top-centrality accounts, which needs no
clusters: `adaptive.py` consumes a set of handles, and both methods emit one.

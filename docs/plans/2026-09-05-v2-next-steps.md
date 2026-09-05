# v2 execution plan

Every remaining step from here to a v2 pipeline in production, with what each
one delivers and how we know it worked. Companion to
[2026-09-05-v2-methodology.md](2026-09-05-v2-methodology.md) (what the method
is and why) and [../OBJECTIVES.md](../OBJECTIVES.md) (what the project is for).

Written 2026-09-05. Supersedes the seven-step sketch of the same date.

**Ordering principle:** the benchmark is the product of the next few weeks, not
the Kenya detector. Nothing touches the Kenya corpus until the reproduction gate
passes, because a detector we cannot score is what we already have.

Every step below states **Deliverable**, **Acceptance**, **Runs on**, and
**Blocks**. A step is not done when the code exists; it is done when the
acceptance line is true.

---

## Step 0 - complete, 2026-09-05

- Deployed code recovered into git. tf1's adjudication layer, pi0's two
  unpushed commits (`ef09ec7` twscrape 0.20.1, `6837f58` SearchTimeline POST),
  and the live-but-unmerged `fix/collector-seed-oom` are all on master
  (`787670d`). 656 tests green.
- Cluster promotion switched off, deployed, and verified firing on pi0 at
  08:54:25 UTC (`cluster promotion disabled; no accounts promoted`). Collection
  resumed writing at 09:04.
- Control-set request sent to Nwala and Luceri.
- Corpus inventory measured (methodology doc §6).

---

# Track A - detection methodology

## A1. The frozen snapshot

**Why first:** every comparison in this plan is meaningless while the corpus
moves underneath it, and there is currently no known-good state to return to.

**Deliverable.** `kma.bench`:

- `snapshot(name) -> str` writes `bench/snapshot=<name>/manifest.parquet`: one
  row per R2 object across every prefix, with path, size, etag, row count
  (parquet footer, not a scan), prefix, and hive partition values. Plus a
  `meta` row group recording creation time, the collector `code_version` range
  the objects span, and the `kma` git sha.
- `pin(con, snapshot) -> None` makes every `kma.db` reader resolve through the
  manifest rather than a glob, so a pinned read cannot see later objects.
- `diff(a, b)` reports objects added between two snapshots, by prefix.

A snapshot is a manifest over immutable per-run objects, never a copy.

**Name the first one for the promotion switch-off.** 2026-09-05 splits the
corpus into two populations: before it, coordination-cluster members were being
promoted into `cib_timeline`; after it, they are not. Any series spanning that
date must split on it.

**Acceptance.** Two `latest_posts` reads pinned to the same snapshot id, taken a
week apart, return byte-identical row counts per partition.

**Runs on.** Laptop. Footer reads only; the full inventory took minutes.

**Blocks.** A7, A8, and all of Track B.

## A2. Acquire and verify the IO archive

**Deliverable.** The six paper countries as per-country Parquet on a Modal
volume, derived from `ioa_tweets.csv` (105.9 GB, all 21 countries) and
`ioa_users.csv` (15.7 MB).

**Verify before building anything on it.** Three checks, each of which changes
the plan if it fails:

1. **Are sub-5,000-follower user ids stable pseudonyms, or per-row hashes?**
   The mirror anonymises those accounts. If ids are not stable across a user's
   rows, no similarity network can be built and the whole approach collapses.
   Test: group by id, confirm a plausible tweets-per-account distribution
   rather than one row per id.
2. **Does tweet text survive anonymisation?** If not, the text-similarity trace
   is unavailable and the fused network is four traces, not five. Recoverable,
   but it changes the expected numbers.
3. **Do URLs and hashtags survive?** Two more traces depend on them.

**Acceptance.** Per-country driver counts and tweet volumes reported against the
paper's Table 1 (China 5,191 / 13.8M, Cuba 503 / 4.8M, Egypt & UAE 240 / 1.5M,
Iran 209 / 9.9M, Russia 3,487 / 9.8M, Venezuela 33 / 9.5M), with any gap
explained. A large unexplained mismatch means the mirror is not what the paper
used, and the gate would be measuring the wrong thing.

**Runs on.** Modal. 105.9 GB is beyond both tf1 (3.8 GB RAM) and comfortable
laptop storage.

**Blocks.** A3 route B, A4, A5.

## A3. The control group - the critical path

Two routes in parallel, because route A's cost is latency and route B's cost is
work.

**Route A (sent).** Await Nwala / Luceri. If the set arrives, use it and record
that the benchmark is a true replication.

**Route B (start now regardless).** Rebuild an era-matched control set from the
Internet Archive Twitter Stream Grab:

- Fetch the monthly archives covering the drivers' active windows.
- Extract the drivers' hashtags from A2, filter Stream Grab accounts to those
  using the same hashtags in the same periods.
- Cap at 100 tweets per account, matching the paper's construction.
- Exclude any account appearing in `ioa_users.csv`.

**Record provenance in the harness, not just in prose.** Every benchmark number
carries which control set produced it. The two are not interchangeable: the
Stream Grab is a ~1% sample drawn by a different process from the paper's
academic-API search.

**Acceptance.** A control set of the same order as the paper's per country
(their totals: China 76,286, Cuba 30,099, Egypt & UAE 370, Iran 16,885, Russia
31,317, Venezuela 3,865), with per-account tweet counts and window overlap
reported against the drivers.

**Stop rule.** If neither route yields a usable control set, the supervised half
of the method is unavailable. Fall back to unsupervised centrality scored by
rank rather than AUC, and record that as a decision taken openly rather than a
drift. Do not substitute the Kenya corpus as negatives - era and language
mismatch make the classes trivially separable and every score inflates.

**Blocks.** A5 (partially), A6 (entirely).

## A4. Implement `kma.coord2`

Build order is cheapest-first so failure is cheap.

1. **Shared bipartite builder.** `user x entity` incidence, TF-IDF weighting
   over entity popularity, cosine-similarity projection to `user x user`. One
   implementation, three entity extractors:
   - `co_retweet` - entity is the retweeted tweet id
   - `co_url` - entity is the URL (decide and record whether t.co links are
     resolved; the paper's entity is the URL as shared)
   - `hashtag_sequence` - entity is the ordered hashtag sequence, minimum 3
     tags. **Verify our `posts.hashtags` list preserves in-tweet order** before
     trusting this trace; it is a `list<string>` and the ordering guarantee is
     unconfirmed.
2. **`fast_retweet`** - entity is the retweeted *author*, restricted to
   retweets landing within 60s. Note the coverage limit: `engagements/` is
   incidence-only with no timestamp, so on the Kenya side this trace is built
   from RT-inclusive search rows only. Report the resulting coverage.
3. **`text_similarity`** - the exception. No bipartite stage. Non-retweets only,
   punctuation/stopwords/emoji/URLs stripped, minimum four words, FAISS cosine
   within a one-year sliding window, edge weight is the mean similarity over
   qualifying pairs. Threshold is the **96th percentile of the observed
   similarity distribution** in whichever embedding space is in use, per the
   Q2 decision.
4. **Fusion.** Union of edges: two users are linked in the fused network if they
   are linked in any individual network.
5. **Detection.** Unweighted eigenvector centrality; absent nodes score 0; prune
   below `10^-2`.
6. **Supervised.** node2vec (128 dimensions, 16 walks per node, 16 steps per
   walk) then Random Forest with 10-fold cross-validation.

**Deliverable.** `kma.coord2` with the paper's parameters as hard-coded
defaults, plus unit tests on synthetic graphs with known structure.

**Acceptance.** Each trace builder reproduces a hand-computed TF-IDF cosine on a
fixture of ten users. Fusion is verified as a union, not an intersection.

**Runs on.** Laptop for tests, Modal for real data.

## A5. The reproduction gate

**Validate small first.** Egypt & UAE only: 240 drivers, 370 controls, 1.9M
tweets. The smallest campaign and the cheapest way to be wrong.

**Deliverable.** A benchmark report per campaign: AUC, F1, precision, recall for
each siloed network and for the fused network, unsupervised and supervised.

**Acceptance, stated before the run, not after:**

- Fused unsupervised: AUC ~0.83, F1 ~0.76.
- Fused supervised: AUC ~0.94, F1 ~0.82, precision ~0.96.
- Node pruning beats edge filtering on precision by roughly +0.42 at comparable
  recall.
- Per-trace optimized AUCs land near the paper's Table 2: fast retweet 0.62,
  co-retweet 0.69, co-URL 0.72, hashtag sequence 0.68, text similarity 0.52.
- Tolerance is agreed and written down before results are seen.

**On failure.** A missed gate means we copied it wrong, not that the method is
wrong. Debug in this order: control-set provenance, entity extraction, TF-IDF
normalisation, then fusion. Text similarity is the one trace where the encoder
and our code are confounded by the Q2 decision, so judge the gate on the fused
result and the other four; treat TS in isolation as indicative only.

**Then scale** to Cuba, Iran, Russia, Venezuela, China, changing only the scale
parameter.

**Runs on.** Modal, `--spawn`, never a blocking `.remote()`.

## A6. Supervised model and cross-campaign transfer

**Deliverable.** The node2vec + RF classifier, evaluated on the paper's three
tasks: per-campaign detection, global multi-campaign classification, and
forecasting engagement from historical years.

**Acceptance.** Global task near precision 0.95, recall 0.70, F1 0.78, AUC 0.92.
Ablation reproduces the paper's ordering: co-retweet contributes most, fast
retweet least.

**Depends on.** A3 delivering a control set. Without one, this step does not run.

## A7. v1 on the same benchmark

Per the Q1 decision, a full comparison rather than an overlap anchor.

**Deliverable.** `kma.coordination` run against the same six campaigns and the
same control set, reporting the same metrics.

**State the caveat with the result, in both directions.** v1's degree-corrected
null assumes near-total census incidence and the IO archive is not a census, so
part of any gap is structural rather than methodological. This is written here,
before the result, so it cannot be deployed selectively afterwards.

**Acceptance.** A single table, v1 against v2, same campaigns, same controls,
same metrics, with the structural caveat attached.

## A8. Transfer to the Kenya snapshot

**Deliverable.**

- v2 run over the pinned snapshot from A1.
- Report: accounts above the centrality threshold; their overlap with v1's last
  published clusters; their Kenya share; their authenticity distribution against
  size-matched random account groups.
- **The prediction from the methodology doc gets checked here:** `text_sim` and
  `fast_co_share` validated zero edges under v1's significance null. Under
  centrality they should return. If they still produce nothing, the trace
  construction is wrong, not the traces.

**Acceptance.** The run completes on the pinned snapshot, is reproducible from
the snapshot id alone, and its cost is measured on the host that ran it rather
than estimated.

**Depends on.** A1, A5.

## A9. Validate the relevance gate

The cheapest unmeasured thing in the project, and load-bearing for what v2
output means (`OBJECTIVES` A1).

**Deliverable.** A hand-labelled sample of a few hundred posts, stratified
across `domain_bucket` values, with measured precision and recall for
`kma.measure`, plus a characterisation of its blind spots (the lexicon proxy is
known to be weak, but the error rate has never been quantified).

**Acceptance.** Precision and recall reported with confidence intervals and the
sample published alongside, so the number can be re-derived.

**Why here.** v2's Kenya output is filtered through this gate, so an unmeasured
gate means an uninterpretable result.

---

# Track B - collection methodology

Independent of Track A until A8; can proceed in parallel once A1 lands.

## B1. The replay harness

**Definition.** A collection policy is a function from the state observable at
time *t* to a set of fetch requests. Replay scores a candidate policy against
the frozen snapshot: given only what the policy could have known at *t*, what
would it have fetched, and how much signal would that have bought?

**Deliverable.** `kma.replay`:

- A `Policy` protocol: `candidates(state_at_t) -> list[FetchRequest]`.
- A time-ordered replay driver that never lets a policy see an object collected
  after *t*.
- The incumbent policy implemented first, as the baseline every candidate is
  scored against.

**Acceptance.** Replaying the incumbent policy over the snapshot reproduces the
objects the collector actually fetched, within a stated tolerance. If the
harness cannot reproduce what actually happened, it cannot score what did not.

## B2. The scoring metric

**Deliverable.** Coordination-useful yield per request: fused-network edges
obtained per API request spent, and once A8 lands, predicted IO-driver accounts
per request.

**Every replay result carries the incumbent-bias statement.** The frozen corpus
is itself the output of the incumbent policy, so replay systematically favours
policies resembling the incumbent. The retweeter census is the partial
exception, because its account x object incidence is near-total for the objects
it covers.

## B3. Candidate policies to test

In rough order of expected value:

1. **Band bounds.** `SNOWBALL_BAND_MIN..MAX` is currently 3..100, tied to
   `HUB_CAP_MAX`. v2 has no hub cap - TF-IDF down-weights popular entities
   instead - so the upper bound may no longer need to exist. Test it.
2. **Budget split** between the retweet census, the conversation census, and
   hate expansion.
3. **TTL** (`SNOWBALL_REFRESH_HOURS`, currently 12) against re-census value.
4. **Object ranking** within the band.

**Acceptance per policy.** Yield per request against the incumbent, on at least
three runs in fresh processes and in both orders, because the run-to-run noise
band on these passes is ~20% and single-run comparisons prove nothing.

---

# Track C - production and operations

## C1. Account pool health

**Current state, measured 2026-09-05:** 51 active / 54 total, 0 locked.
`@Adiletfu`, `@Braoj9g` and `@Saniye6i6a` all last succeeded 2026-08-28 within
four minutes of each other, at ~20,500-20,650 lifetime requests against
~23,900-24,000 for healthy accounts. They fail relogin with Cloudflare 403 on
every `ACCOUNT_SYNC_HOURS` pass, which is the HTML noise in the logs.

**Open question, deliberately not yet tested.** Whether the block is
account-level (three accounts suspended together, nothing to fix but replace) or
login-path level (Cloudflare refusing the login endpoint from pi0's IP, in which
case the pool decays one account at a time as sessions expire and volume looks
normal until it does not). Distinguishing evidence is whether a *healthy*
account can complete a fresh login from pi0 - which risks that account, so it is
a decision, not a default.

**Deliverable.**

- A pool-health metric persisted per cycle (active count, locked count, failed
  relogins) so decay is visible as a series rather than as log noise.
- An alert threshold: investigate if active drops below 48.
- A replacement procedure for dead accounts.

**Priority.** Higher than it looks. `MAX_AGE_DAYS=14` means collection lost to a
decaying pool is unrecoverable, and this is the exact shape of the documented
"volume goes to zero while everything looks healthy" failure.

## C2. Turn adjudication on

**Blocked on:** `ANTHROPIC_API_KEY` in tf1's `~/kenya-monitor-2027/.env`, plus
`ADJUDICATE_REFRESH_HOURS` set above 0. Without a key the pass fails loudly by
design.

**Deliverable.** Layer three live, adjudicating v2 output rather than v1
clusters once A8 lands.

**Acceptance.** Agreement between the adjudicator and a human on a held-out set
of clusters, and the rate at which adjudication changes the ranking.

**Note the open question of unit:** the IO ground-truth work suggested the unit
of judgement should be the campaign, not the cluster. v2 predicts accounts, so
this needs settling when adjudication is re-pointed.

## C3. Re-point collector targeting

**Deliverable.** Promotion reads v2 centrality ranks instead of coordination
clusters. `adaptive.py` consumes a set of handles, so this is a source swap, not
a rewrite: replace `cluster_accounts` with a centrality-rank reader, keep the
Kenya-share and channel gates that already exist, and flip
`CLUSTER_PROMOTION_ENABLED` back on.

**Acceptance.** A promotion pass promotes accounts, logs its source as v2, and
the promoted set is measurably more Kenya-relevant than the v1 cluster path was
(the v1 path's corroborated tier was 8.3% Kenya-referencing).

**Cost of the gap until then**, recorded so it is not rediscovered: no new dense
sampling of suspected coordinated accounts, so the pool of objects the census
can band thins over time, and the loss is permanent past the 14-day horizon.

## C4. Republish the dashboard on v2

Per the Q4 decision the series restarts; nothing carries over.

**Deliverable.**

- Producer reads v2 output, still from the persisted run rather than rebuilding.
- Only composition-standardised or within-partition rates published
  (`OBJECTIVES` C1/A7).
- The two broken coordination readers fixed or retired - both union every
  historical pass instead of the latest (42.7x and 1.39x inflation).

**Still blocked on the user:** an R2 API token scoped to both buckets, and the
Zero Trust Access application.

## C5. Retire the v1 coordination pass

Only after C3 and C4. Until then tf1 keeps computing v1 clusters, which is the
input the dashboard and any residual targeting still read.

---

# Sequencing

```
A1 snapshot ──┬─> A8 Kenya transfer ──> C3 re-point ──> C4 dashboard ──> C5 retire v1
              └─> B1 replay ──> B2 metric ──> B3 policies
A2 archive ───┬─> A4 coord2 ──> A5 gate ──> A6 supervised ──> A7 v1 comparison
A3 controls ──┘                                    │
                                                   └─> A8
A9 relevance gate ── alongside A8
C1 pool health ── independent, start now
C2 adjudication ── needs a key, then follows A8
```

**Can start today, in parallel:** A1, A2, A3 route B, C1.

**Critical path:** A3. Everything supervised waits on a control set, and route A
is out of our hands.

---

# Cross-cutting measurement rules

These are paid for in past mistakes and apply to every step:

1. **Never time two variants back to back in one process.** The second hits an
   httpfs cache the first filled. This has produced two false speedup claims on
   this project already. Fresh process per variant, both orders, at least three
   runs against a ~20% noise band.
2. **Measure on the host that will run it.** A query verified only on tf1 is not
   verified for pi0.
3. **Never aggregate a series across a parameter change** without splitting on
   `code_version`.
4. **Never report a failed query as an empty result.** A dead query once read as
   "no accounts cleared the floors" for a day.
5. **Watch outputs, not process status.** Every silent failure on this project
   was a healthy-looking container producing nothing.
6. **Every benchmark number carries its snapshot id and its control-set
   provenance.**

---

# Risk register

| Risk | Detail | Mitigation |
|---|---|---|
| **Actor-type mismatch** | The north star is recall against state-backed IOs. Kenya's documented problem is domestic disinformation-for-hire: influencers paid US$10-15, coordinated in WhatsApp groups, seeding trends. Every step optimises for the former. | Deliberate, taken for the sake of having ground truth at all. Restate it in anything published. Keep Elmas-style trend forensics as a separate track, never merged. |
| **No control set** | Route A may not answer; route B is a different sampling process. | Stop rule in A3: fall back to unsupervised rank-scored detection, declared openly. |
| **Archive anonymisation** | Sub-5k-follower ids may be unusable, or text stripped. | Verified in A2 before anything is built on it. |
| **Temporal transfer** | Archive is 2010-2020 X; we detect on 2026-27 X. | Report it as a limitation; the forecasting task in A6 is the closest available evidence on it. |
| **Pool decay** | Three accounts dead, cause undiagnosed. | C1, prioritised. |
| **Cost** | Fused similarity networks over ~2.5M posts and ~1.2M authors; text similarity is quadratic. | Measure, do not estimate. FAISS plus the sliding window. Modal for everything. |
| **Collection gap** | Promotion is off, so densification thins from 2026-09-05. | Bounded by C3. The snapshot names the date so the two populations stay separable. |

---

# Deliberately out of scope

- Live parallel collection arms and a standing random control arm. Replay is the
  only collection test for now, with its incumbent bias stated on every result.
- Elmas-style ephemeral-astroturfing detection, despite matching Kenya's
  documented attack pattern more closely than IO detection does. Separate track;
  merging it into v2 would defeat the point of copying one methodology.
- IOHunter. It becomes a drop-in once the fused network exists.
- Facebook, TikTok, WhatsApp. Unchanged from `OBJECTIVES` §5.

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

**Verifier.** `kma.ioa_verify` runs the three checks above plus a fourth, and
takes a sample or the full file: `uv run kma-ioa-verify PATH [--limit N]`.

**Measured 2026-09-05** against the Internet Archive mirror (the only source
left: X's own transparency page now 404s). `ioa_users.csv` read in full,
`ioa_tweets.csv` sampled by range request over its first 3 MB of 113.72 GB:

- 87,287 user rows, 86,971 distinct userids, all 21 countries rather than the
  paper's six. 85,064 of 87,287 (97.5%) userids are hashed - the sub-5,000
  follower anonymisation - and 2,223 are numeric.
- **Check 1 answered: hashed ids are stable pseudonyms.** In a 4,470-row
  sample, 10 hashed users carry 4,468 rows, mean 446.8 tweets each, max 1,946.
  Per-row hashing would have given one row per id. Similarity networks are
  buildable.
- **Check 2 answered: text survives**, 100% of sampled rows.
- **Check 3 answered: entities survive.** URLs on 89.6% of rows, hashtags on
  4.5%, stored as an ordered list literal. All five traces are supported:
  co_retweet via `retweet_tweetid`, fast_retweet via `retweet_userid` plus
  `tweet_time`, co_url via `urls`, hashtag_sequence via `hashtags`,
  text_similarity via `tweet_text`.

**Check 4, and the one that is still open: campaign attribution.** Neither file
carries a country or campaign column. The mirror is a plain concatenation of
X's per-campaign exports and campaign identity was lost in the making of it;
there is one embedded header row mid-file in `ioa_users.csv`, which is what a
concatenation looks like. `check_campaign_attribution` reports the evidence for
segmenting it - embedded header rows, contiguous runs of `account_language`,
contiguous `userid` blocks, `account_creation_date` spread - and deliberately
reports evidence rather than guessing a labelling, because a fabricated country
column would sit silently under every per-campaign number in the gate.

**Consequence, stated so it is not discovered at A5:** if campaigns cannot be
attributed, per-country evaluation is impossible. The fallback is the paper's
Task 2, global classification over all drivers combined. That keeps a valid
benchmark and loses the per-campaign comparison, the per-trace Table 2
reproduction per country, and the Table 1 driver-count acceptance check above.
It is not implemented.

**Also blocked by attribution:** `fast_retweet` on the archive needs the
ORIGINAL tweet's timestamp to measure the 60s delay, and the archive carries
only the retweet's own `tweet_time`. The original is joinable only when it is
also an archive row, so an operation retweeting an organic account leaves
nothing to time. Report `coord2.fast_retweet_coverage` with any fast-retweet
number from this data.

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

**Delivered 2026-09-05** as `kma.coord2`, 57 offline tests, no network and no
credentials in any of them. Four things the next reader needs:

1. **Hashtag order: in-tweet order in practice, unguaranteed in principle.**
   The chain X `entities.hashtags` -> twscrape `[x["text"] for x in ...]` ->
   `list(tw.hashtags)` -> `list<string>` never sorts or dedupes, so the list is
   as X returned it, which is ascending `indices`. But twscrape DISCARDS
   `indices`, so the order cannot be verified or repaired from what we persist;
   for long-form posts the text comes from `note_tweet` while the hashtags come
   from the truncated `legacy` entity set, so the sequence can be short of the
   text; and none of this has been checked against live rows, because a
   worktree has no R2 credentials. `hashtag_sequence_traces(order="text")`
   re-derives the sequence from the post text and is the safer default wherever
   `text` is populated.
2. **Thresholds are percentiles throughout**, per the Q2 decision, taken over
   the realised (non-zero) edge weights rather than over all user pairs.
3. **Ambiguities resolved by choice, not by evidence**, all recorded in the
   module docstring: fast_retweet gets no similarity percentile because Table 2
   sweeps its time interval instead; the paper does not say whether node pruning
   runs on filtered or unfiltered networks, so `percentile=None` runs the second
   reading; hashtag case is folded.
4. **The text-similarity percentile is the weak point of the copy.** The paper's
   0.95 = 96th percentile is not consistent with a percentile over all in-window
   pairs, which is what an exhaustive scan produces. See A5's debugging order.

**node2vec** is implemented directly - biased walks plus skip-gram with negative
sampling in torch - rather than by adding gensim, which would be a new top-level
dependency with a history of scipy pins that break this environment.

## A5. The reproduction gate - RUN 2026-09-05, 5/6 PASS

**Decided: gate against the IOHunter benchmark**, not the WWW 2024 datasets.
The IOHunter release (Minici, Luceri, Fabbri, Ferrara, AAAI 2025, CC-BY-4.0,
zenodo.org/records/13357621) ships per-country pickles containing the five trace
networks (`coRT`, `coURL`, `hashSeq`, `fastRT`, `tweetSim`) at a text-similarity
threshold of 0.7, the fused `graph`, driver/control `labels`, and five
train/val/test `splits`.

**These are NOT the WWW 2024 datasets** and the WWW acceptance numbers do not
apply: this release has 533 positives for Venezuela against Table 1's 33 drivers,
and 275 for Russia against 3,487. Same countries, same construction, different
inclusion rules. Targets are IOHunter Table 2, Macro-F1 over five seeds.

**What this gate tests, and what it does not.** The traces arrive pre-built, so
it exercises our fusion and detection against a known answer and says nothing
about our trace construction. Two separate claims; do not merge them. Trace
construction is still untested and needs the raw archive.

**Result** (`uv run kma-iohunter-gate --data-dir <unpacked>/data/processed`):

| country | nodes | edges | positives | ours | target | delta | within 2 SD |
|---|---|---|---|---|---|---|---|
| UAE | 9,391 | 2,118,833 | 3,349 | 84.64 | 84.66 | -0.02 | yes |
| cuba | 20,247 | 4,737,799 | 461 | 57.92 | 57.92 | -0.00 | yes |
| russia | 716 | 10,431 | 275 | 87.83 | 87.65 | +0.18 | yes |
| venezuela | 5,021 | 56,741 | 533 | 95.05 | 95.05 | 0.00 | yes |
| iran | 14,486 | 394,447 | 4,659 | 71.43 | 60.83 | **+10.60** | **no** |
| china | 23,028 | 411,311 | 768 | 63.67 | 63.66 | +0.01 | yes |

Five of six reproduce, four of them to within 0.2 Macro-F1 points. The mean is
76.76 against a target of 74.96.

**The Iran anomaly, unresolved.** We score 10.6 points ABOVE the published
number. Being better than the reference is a divergence, not a success, and it
is not explained by: the rewiring seed (identical across five seeds), the
eigenvector convergence settings (identical at max_iter 50/100/1000 and tol
1e-6/1e-8), or the isolated-node definition (fixing that moved Iran further out,
from +7.05 to +10.60, which does show Iran carries self-loop-only nodes). Treat
any Iran number from this pipeline as suspect until the cause is found.

**Two spec corrections this settled, from the reference implementation rather
than from argument:**

- The centrality threshold is a **swept percentile selected on train+val**, not
  the fixed 1e-2 the paper quotes as a conservative operating point. An absolute
  cut cannot port between graphs, because eigenvector centrality is
  L2-normalised across nodes: typical values run ~0.06 on a 240-node campaign
  and ~0.001 on a 1.2M-author graph.
- **Isolated nodes are rewired** to five random non-isolated nodes before
  centrality is computed. We had no such step. The check is neighbour-based, not
  degree-based: a self-loop contributes 2 to `degree`, so a node whose only edge
  is to itself reads as connected by degree and isolated by neighbours.

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

**Scoped down 2026-09-05.** Collection runs for weeks, to feed the v2 build,
then stops or is rebuilt. Most of what was here earns its keep only over months,
so it is dropped rather than carried as pretend work.

## C1. Account pool - watch, do not invest

**Measured state:** 50 active / 54 total. Three accounts died in a single wave
on 2026-08-28 within four minutes of each other; a fourth was spent deliberately
on 2026-09-05 to diagnose the cause. **Eight days passed between the wave and
that test with no further deaths**, so there is no measured background decay
rate - one event, not a trend.

**Diagnosed cause: X's login flow, not rate and not our IP.** Of the four dead,
two route through proxies and two do not, and all four fail identically at
`onboarding/task.json` with a Cloudflare 403 - the same from pi0, tf1 and a
laptop. Read paths were patched on 2026-08-30 (`twscrape_compat.py`, SearchTimeline
as POST); the auth path never was, and twscrape 0.20.1 is the latest release, so
there is no upgrade to take.

**Consequence:** a lapsed session cannot be re-created by password login. There
is a route that avoids login entirely - `accounts.sync` marks an account active
when `accounts.yaml` supplies fresh cookies containing `ct0` - but exporting
those is manual browser work per account.

**Decision: do nothing unless the number moves.** 50 accounts is ample for a
horizon of weeks. Do not spend more accounts on diagnosis. If active count drops
below ~40, revisit the cookie-refresh route then.

**Do not "protect" the pool by slowing down.** The 25% pacing cut made on
2026-09-05 was reverted the same day: it was a defence against rate pressure,
and rate pressure is not what killed these accounts. It cost throughput and
bought nothing.

## C2. Adjudication - dropped from this track

No `ANTHROPIC_API_KEY` goes on tf1. Model work runs on Modal. If v2 needs an
adjudication layer, it is designed as a Modal pass against v2 output, not as a
revival of the dormant tf1 service.

## C3. Collector targeting - dropped

Promotion stays off. Re-pointing it at v2 centrality ranks only pays back if
collection continues long enough for the densification to matter, and it does
not. The gap is now a known property of the corpus rather than a debt: the
snapshot `2026-09-05-promotion-off` names the date the two populations split.

## C4. Dashboard - deferred until v2 has something to publish

The R2 dual-bucket token and the Zero Trust application stay unrequested until
then. The two broken coordination readers are moot if v1 output is never
republished.

## C5. Retire v1 - moot

Nothing needs retiring if collection stops. tf1 can keep computing v1 output
until it is switched off.

# Sequencing

```
A1 snapshot (done) ──┬─> A8 Kenya transfer
                     └─> B1 replay ──> B2 metric ──> B3 policies
A2 archive ──┬─> A4 coord2 ──> A5 gate ──> A6 supervised ──> A7 v1 comparison
A3 controls ─┘                                    │
                                                  └─> A8
A9 relevance gate ── alongside A8
Track C ── scoped to watching the pool; nothing to build
```

**Can start today, in parallel:** A2, A3 route B, A4. A1 is done.

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
| **Pool decay** | Four accounts dead. Cause diagnosed 2026-09-05: X's login flow, not rate and not our IP. A lapsed session cannot be re-created. | Watch only. 50 accounts is ample for a horizon of weeks; revisit the cookie route below ~40. |
| **Cost** | Fused similarity networks over ~2.5M posts and ~1.2M authors; text similarity is quadratic. | Measure, do not estimate. FAISS plus the sliding window. Modal for everything. |
| **Collection gap** | Promotion is off, so densification thins from 2026-09-05. | Accepted. Collection stops in weeks anyway, and the snapshot names the date so the two populations stay separable. |

---

# Deliberately out of scope

- Live parallel collection arms and a standing random control arm. Replay is the
  only collection test for now, with its incumbent bias stated on every result.
- Elmas-style ephemeral-astroturfing detection, despite matching Kenya's
  documented attack pattern more closely than IO detection does. Separate track;
  merging it into v2 would defeat the point of copying one methodology.
- IOHunter. It becomes a drop-in once the fused network exists.
- Facebook, TikTok, WhatsApp. Unchanged from `OBJECTIVES` §5.

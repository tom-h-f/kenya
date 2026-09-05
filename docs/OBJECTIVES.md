# Objectives

What this project is trying to achieve, stated so that each objective has a
measurable outcome, a named mechanism, and a known failure mode. Written
2026-09-05, at a point where the pipeline works end to end but its methodology
has never been tested against alternatives.

Companion documents: [how-it-works.md](how-it-works.md) describes what runs,
[collection/README.md](collection/README.md) the collector, and
[analysis/README.md](analysis/README.md) the analysis phases. This document
describes what all of that is *for*, and how we would know it was working.

---

## 0. Purpose

Observe manipulation of the public conversation around the 2027 Kenyan general
election on X, in near-real time, and produce three things:

1. **An archive.** X search reaches back 14 days. Anything not collected inside
   that window is gone permanently. The archive is the only irreplaceable
   output; every other objective can be retried against stored data, and this
   one cannot.
2. **Triage leads.** Ranked accounts, clusters, and claims that a human should
   look at, each with the evidence that produced the rank. Never a verdict.
3. **Defensible public statistics.** A small number of measurements about
   toxicity, coordination, and narrative that survive the criticism that they
   are artifacts of how the data was collected.

The election is scheduled for August 2027, roughly 23 months out. That horizon
sets the priorities: the archive and the measurement infrastructure matter more
now than detection accuracy, because detection can be re-run on stored data and
collection cannot.

## 1. Standing constraints

These bound every objective below and are not design choices we can revisit.

- **The 14-day horizon.** Collection outages destroy data. Uptime is a research
  asset, not an ops nicety.
- **One collector host.** pi0, residential IP, ~54-account twscrape pool, DuckDB
  under a 600 MB budget inside a 1 GB container. Throughput is governed by
  per-account pacing, not by wall-clock scheduling.
- **No ground truth for Kenyan CIB.** There is no labelled set of coordinated
  Kenyan election accounts. Everything we call validation is a proxy: X's
  published information-operations archive, synthetic injection, degree-
  preserving shuffles, and human adjudication.
- **The corpus never stops moving.** It is append-only, and the collector's own
  parameters change what lands in it. Any measurement not pinned to a frozen
  snapshot is confounded by collection changes made while it was running.
- **Triage, never verdict.** The coded-term register is documented advisory
  language whose entries have innocent everyday senses. The community/ethnicity
  proxy is aggregate-only. No output labels a person.

---

## 2. Collection objectives

### C1. A baseline sample that can carry rates and trends

Keyword search and account timelines sample the discourse so that prevalence
("what share of election talk is toxic") and trend ("is that rising") can be
computed at all.

- **Mechanism:** `posts/type=search`, `type=timeline`, `type=replies`,
  `type=hydrated`, scoped by `latest_posts(con, scope="baseline")`.
- **Metric of record:** composition-standardised prevalence, plus the baseline
  mix itself (share of baseline rows by type, weekly).
- **Known failure:** the baseline/targeted split does not protect against
  composition drift *within* baseline. Between July and August 2026 the baseline
  mix moved from 70.7% search / 12.1% replies to 14.3% / 72.6% because the
  collector widened its conversation arm, and replies carry 3.2x the hate rate
  of search. The raw toxicity series appeared to rise 78%; standardised to the
  earlier mix it was flat then falling. **A raw prevalence series from this
  corpus is not publishable.**

### C2. Object-level density sufficient for a valid coordination null

Coordination statistics test whether two accounts co-act on shared objects more
than chance allows. That test is only valid if the account x object incidence is
near-complete for the objects it covers, which is why the retweeter census is a
census and not a sample.

- **Mechanism:** snowball census of objects in the `SNOWBALL_BAND_MIN..MAX`
  (3-100 amplifiers) band, TTL'd per object.
- **Metric of record:** amplifiers surviving the hub cap, pairable accounts per
  channel, and tested pairs per pairable account. *Accounts discovered is an
  actively misleading metric* - an account whose only traces are on hub objects
  cannot become a cluster member by any path.
- **Established:** censusing the most-reposted objects is close to useless (420
  of 422 censused objects were hubs; effective coverage of the useful band was
  0%). Banding is simultaneously cheaper per object and strictly more useful.

### C3. A second, independent channel that can corroborate the first

A single-channel cluster is a candidate; a cluster corroborated across two
behavioural channels is a finding.

- **Mechanism:** conversation census (`type=replies`), banded and TTL-aware,
  feeding `co_reply` alongside `co_retweet`.
- **Metric of record:** bridge accounts, shared validated pairs, corroborated
  cluster count - always read together, because one bridge account flickering in
  and out moves the cluster count.
- **Status:** fixed and verified 2026-08-12 (bridge accounts 1 -> 397, shared
  pairs 0 -> 408, corroborated clusters 0 -> 23 of 128).
- **The open problem:** corroboration now works and *what it surfaces is
  engagement farming, not election CIB*. Corroborated clusters are 8.3%
  Kenya-referencing against 51.5% for single-channel ones. The strongest
  evidence tier is currently the least on-topic.

### C4. Reach into the coded and toxic tail, without contaminating the baseline

Targeted collection deliberately oversamples the toxic tail. Its value depends
entirely on it staying out of every denominator.

- **Mechanism:** coded-register search, seed expansion, coordination promotion,
  all written to `TARGETED_TYPES` partitions; scoping is on **first-seen** type
  so a re-collected baseline post cannot be drained out of the baseline
  denominator by a hate pass.
- **Metric of record:** coded-register coverage, seeds promoted per pass, and
  the share of targeted rows that would have been missed by baseline collection.
- **Known failure:** the classifier that selects seeds is a *locator*, not a
  label. On 14 known-coded 2026 posts its mean `p_hate` dropped against baseline
  (0.070 -> 0.032). It finds explicit toxicity and misses the coded register, so
  it is only ever asked to find a foothold account.

### C5. Social-graph structure

Follower edges are the corroborating structure that behaviour alone cannot
supply, and the one signal an account farm cannot cheaply fake.

- **Mechanism:** `monitor follows` + `crawl-follows` BFS from suspicious and
  hate seeds.
- **Metric of record:** edges per day and crawl-frontier drain rate.
- **Known failure:** this step was dead for 27 days behind a `TypeError` while
  the container reported healthy, and burned 8.7 hours per cycle building a
  queue before crashing. Watch the output, not the process.

### C6. Temporal fidelity

Engagement growth over time and second-resolution behaviour during bursts.

- **Mechanism:** metrics refresh on top posts; burst detection (hourly volume
  z >= 3) skips the cooldown so the next cycle starts immediately.
- **Metric of record:** metric snapshots per post per day; median lag from post
  creation to first observation.

### C7. Continuity

- **Metric of record:** rows/day, cycle minutes, and gap-days (days with no
  collection). A gap-day is a permanent hole.
- **Known failure:** throughput fell 56,743 -> 9,110 rows/day across two days in
  August 2026 with every container reporting healthy.

### C8. Cost and safety of collection

Stay inside the account pool's sustained throughput, do not get the pool banned,
and keep every collector-side query inside the memory budget.

- **Metric of record:** requests per cycle per account, pool health, peak
  container memory.
- **Rule:** every collector query must be windowed, projected, spillable, and
  staged - and measured on pi0, because tf1 has the headroom to absorb a query
  that kills pi0 at the same cap.

---

## 3. Analysis objectives

### A1. Relevance - is this even about Kenya?

Everything downstream is meaningless if the corpus has drifted off-domain, and
it demonstrably has: corroborated clusters include a Ugandan pod, a Nigerian
reply pod, and a US follow-train.

- **Mechanism:** `kma.measure`, a lexicon proxy producing `domain_bucket` and
  `kenya_share`.
- **Metric of record:** precision and recall of the gate against a hand-labelled
  sample. **Currently unmeasured.** This is the cheapest high-value gap in the
  whole pipeline: the gate is load-bearing for cluster promotion
  (`CLUSTER_MIN_KENYA_SHARE`) and its error rate is unknown.

### A2. Account authenticity

Rank accounts by how much their profile and behaviour resemble a manufactured
account.

- **Mechanism:** `kma.authenticity` - transparent weighted features plus an
  anomaly rank. No labels, so this is ranking, not classification.
- **Metric of record:** separation between detected clusters and size-matched
  random account groups (permutation test, effect size), plus sanity behaviour
  on known-real principals.
- **Known failure:** `anomaly_rank` flags mega-influencers as outliers and must
  never be read alone. The weights and scale constants are hand-set and have
  never been calibrated against data.

### A3. Coordination - which accounts act together beyond chance

- **Mechanism:** behavioural traces -> account x account projection -> degree-
  corrected configuration-model null -> multiple-testing correction -> Leiden
  clustering across channels -> scorecards.
- **Metrics of record:**
  - **Recall vs X's IO archive: 18.6%** at the deployed Leiden resolution.
  - **False positives <= 0.5%** of clustered accounts, measured by degree-
    preserving shuffle rather than by labels.
  - **Null yield at burn-in >= 100.** At burn-in 5 the null reports 60 clusters,
    i.e. the detector's own signal; at 100 it reports 2.
  - Chance share, corroboration rate, Kenya share of clustered accounts.
- **The unresolved trade:** dropping Leiden resolution 0.05 -> 0.001 takes
  recall 18.6% -> 50.6% *and* chance share 28.6% -> 14.0% - better on both axes -
  but takes live clusters 154 -> 673, breaking the published series and
  multiplying collector targeting 6.6x. The resolution is currently frozen for
  reasons that have nothing to do with detection quality.
- **Refuted, do not rebuild:** a "high self-amplification means mere engagement
  pod" filter. Confirmed operations self-amplify *more* than our pods (d =
  +0.78), because an operation runs engagement bait alongside its political
  assets. Pod-versus-operation is not a statistical question.

### A4. Narrative and claims - what is being pushed

- **Mechanism:** embeddings -> topics; `kma.stories` claim clustering,
  trusted-media corroboration gap, story scorecards; `kma.framing`.
- **Metric of record:** `kma.eval` - known real disinformation cases walked
  stage by stage, reporting the first stage at which each case drops out
  (collection / embedding / clustering / scoring). A change that raises recovery
  is an improvement; one that loses a case is a regression.
- **Caveat that must stay attached to output:** a corroboration gap is a triage
  flag, not proof of falsity. Trusted outlets lag breaking news.

### A5. Harm - hate speech and ethnic incitement

- **Mechanism:** fine-tuned afro-xlmr 3-class classifier (threshold 0.28 on
  `p_hate`, not argmax) plus a documented coded-term lexicon with zero-shot NLI.
- **Metric of record:** unanimous-test macro-F1 (0.688 on the promoted model,
  +9.1 points over the data-recipe reference) and, separately, a **coded-register
  challenge set**, because the two measure different things and only the second
  matters for this corpus.
- **Established negative results:** class weighting is harmful (-6.7 points);
  focal loss, LLRD and label smoothing are neutral or worse; training on
  LLM-labelled 2026 posts *regressed* coded detection by training the model into
  the labeller's blind spot. Dual-labelling shows a 4.56x hate-conservatism
  asymmetry between labellers, replicated.

### A6. Adjudication - is this a campaign worth a human's attention

Statistics answer *do these accounts act together*. They cannot answer *why*,
and that is by construction, not a tuning gap. So the architecture is three
layers: **detect** (statistics), **filter** (relevance), **adjudicate** (a
reader with dossiers).

- **Metric of record:** agreement between the adjudicator and a human on a held-
  out set of clusters, and the rate at which adjudication changes the ranking.
- **Status:** built, dormant. `ADJUDICATE_REFRESH_HOURS=0` because there is no
  `ANTHROPIC_API_KEY` on the host. A human reader with dossiers got 4/6
  operations and 0/6 false alarms on the IO ground-truth set; the two misses were
  the operation's own pure engagement-bait accounts.
- **Open question of unit:** the evidence suggests the unit of judgement should
  be the *campaign*, not the cluster.

### A7. Publishable statistics

A small set of numbers that can be defended against "that is an artifact of your
collection".

- **Metric of record:** for each published series, whether it is standardised or
  within-partition, and whether its window starts at collection start rather
  than collection start minus the search horizon.
- **Currently unpublishable:** the raw toxicity series (composition drift, C1),
  and the structural-zero corroboration claim (now false).
- **Currently broken in the producer path:** two coordination readers union
  every historical pass instead of the latest (42.7x and 1.39x inflation).

---

## 4. What is not currently answerable

Stated plainly, because these are the gaps a new methodology has to close.

1. **There is no unbiased denominator.** Nothing in the collector samples the
   Kenyan election conversation at random. Every prevalence is conditional on a
   target list that the pipeline itself keeps editing.
2. **There is no control arm.** Nothing measures what targeted collection
   missed, so "the collector found X" and "X is what is there" cannot be
   separated.
3. **There is no frozen baseline corpus.** Parameter changes and the corpus move
   together, so a before/after comparison confounds the two. There is no
   known-good state to return to; it can only be rebuilt from the persisted
   run series, split on `code_version`.
4. **No collection variant has ever been compared to an alternative.** Every
   change so far was reasoned about, shipped globally, and evaluated against a
   corpus that changed underneath it.
5. **The relevance gate's error rate is unknown** (A1), while being load-bearing
   for what the collector chases next.
6. **Recall is capped by a parameter frozen for non-detection reasons** (A3).
7. **tf1 runs code that is not in origin.** The deployed adjudication layer and
   the IO ground-truth work exist only as an rsync on that host.

## 5. Non-goals

Facebook, TikTok, WhatsApp actors; archival beyond the 14-day horizon;
per-person labelling of any kind; automated publication of any verdict.

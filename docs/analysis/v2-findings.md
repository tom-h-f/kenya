# What v2 says about the corpus

Findings from building and running the v2 coordination methodology, 2026-09-05
to 2026-09-08. Method and rationale: [../plans/2026-09-05-v2-methodology.md](../plans/2026-09-05-v2-methodology.md).
Execution plan: [../plans/2026-09-05-v2-next-steps.md](../plans/2026-09-05-v2-next-steps.md).

Everything below is measured on snapshot `2026-09-05-promotion-off`: 20,788 R2
objects, 39,183,192 rows, 1,045,318 deduplicated posts, 331,138 authors.

---

## 1. The headline

v2 reproduces on published benchmarks and then behaves very differently on our
corpus, for reasons that are properties of the data rather than faults in the
copy. Three things follow, and the third is the one that matters:

1. **The copy is faithful.** 5 of 6 benchmark datasets reproduce the published
   unsupervised numbers, four of them within 0.2 Macro-F1.
2. **Our corpus cannot supply the method's main signal.** The benchmark datasets
   are dominated by co-URL and text similarity; ours has almost no URL sharing.
3. **v1 and v2 surface almost entirely different accounts** - 22 of 500 overlap -
   and nothing we hold can say which is right.

## 2. The gate: does the copy work

Scored against the IOHunter release (Minici, Luceri, Fabbri, Ferrara, AAAI 2025,
CC-BY-4.0), which ships prebuilt trace networks, driver/control labels and
splits for the same six countries as the WWW 2024 paper. Targets are its Table 2,
Macro-F1 over five seeds; tolerance was two published standard deviations, fixed
before any result was seen.

| dataset | NodePruning | target | | node2vec+RF | target | |
|---|---|---|---|---|---|---|
| UAE | 84.64 | 84.66 | pass | 94.58 | 96.97 | fail |
| cuba | 57.92 | 57.92 | pass | 96.16 | 91.53 | fail |
| russia | 87.83 | 87.65 | pass | 88.43 | 83.43 | pass |
| venezuela | 95.05 | 95.05 | pass | 90.14 | 90.32 | pass |
| iran | 71.43 | 60.83 | **fail** | 79.33 | 80.50 | pass |
| china | 63.67 | 63.66 | pass | 80.32 | 83.89 | fail |

Unsupervised 5/6. Supervised 3/6 at a fixed epoch budget, with the **mean at
88.16 against a target of 87.77** - the aggregate matches while individual
countries scatter.

**Iran - explained 2026-09-11: the benchmark disagrees with itself, not with
us.** IOHunter's own NodePruning code, run verbatim with its own seed on the
released pickles, reproduces its published mean and standard deviation to two
decimals for russia, venezuela and china - and gives Iran **71.31 +/- 0.76**,
not the published 60.83. Ours is 71.43. The published Iran figure cannot be
produced by the published code on the published data, so the gap sits between
the paper and its data release. The eigenvector computation was also ruled out
(every iteration scheme gives 71.43-71.46). Evidence and scripts:
`analysis/investigations/2026-09-11-iran-divergence/`.

*Superseded:* **Iran is unexplained.** We score 10.6 points ABOVE the published number, which
is a divergence and not a success. Ruled out: the rewiring seed (identical across
five seeds), eigenvector convergence settings (identical at max_iter 50/100/1000
and tol 1e-6/1e-8), and the isolated-node definition (correcting it moved Iran
further out). Treat any Iran figure from this pipeline as suspect.

**What the gate does NOT test.** The release ships trace networks already built.
So this validates our fusion and detection against a known answer and says
nothing about our trace construction - which is where every problem in §4 lives.

## 3. Our corpus is a different kind of data

Share of trace edges by trace:

| dataset | coRT | coURL | hashSeq | fastRT | tweetSim |
|---|---|---|---|---|---|
| UAE | 11.5 | 14.6 | 1.0 | 0.1 | **72.8** |
| cuba | 1.7 | **92.4** | 0.2 | 0.0 | 5.8 |
| russia | 13.9 | **79.4** | 2.4 | 2.1 | 2.1 |
| venezuela | 26.8 | **70.6** | 0.9 | 0.9 | 0.8 |
| iran | 48.6 | 48.4 | 1.2 | 1.0 | 0.8 |
| china | 27.1 | **70.9** | 0.2 | 1.6 | 0.1 |
| **KENYA** | **47.4** | 0.1 | 0.2 | 0.0 | 52.2 |

The coURL:coRT ratio runs 1.0 to 54.6 across the benchmarks and is **0.003**
here. The cause is not collection: only **6.19%** of original Kenyan posts carry
a URL at all (1.52% of retweets), and inheriting URLs from retweet parents
recovers 4,779 rows out of 364,287. Collection cannot capture links users did not
post.

**Why the sources differ.** X's disclosure archive is *account-complete* - every
tweet an attributed account posted across its lifetime - and state operations
exist to push content, so their signal lives in shared links. Our corpus is
*object-complete but account-sparse*: a 14-day keyword window plus a retweeter
census over banded objects, with retweets deliberately oversampled because v1
needed co-retweet.

This matters more than it looks. The paper reports that no single trace captures
an operation (co-retweet alone gets 58-62% of drivers in china, russia and iran,
against 84-97% fused) and that trace populations have near-zero mutual
information. Running one trace is not a weakened version of the method; it is its
weakest component.

Building the text-similarity trace fixed the imbalance and confirmed the point:
the two traces agree on almost nothing. Of the top 500, all 500 appear in text
similarity and 136 in co-retweet, and the new top 500 shares **one** account with
the co-retweet-only ranking.

## 4. Four deviations, all forced by data

Recorded in full in §8a of the methodology doc. Each was discovered by running
the method, not by reading it.

| # | Deviation | Why |
|---|---|---|
| 1 | Centrality threshold is a swept percentile chosen on validation, not a fixed `1e-2` | The reference implementation does this; eigenvector centrality is L2-normalised, so an absolute cut cannot port between graph sizes |
| 2 | Isolated nodes rewired before centrality | In the reference, absent from the paper. The check must be neighbour-based: a self-loop contributes 2 to `degree` |
| 3 | Minimum-activity floor of 2 entities per user | Not in the paper or reference. See below |
| 4 | Text similarity uses an absolute 0.85, not a 96th percentile | See below |

**The activity floor.** The paper's defence against popular entities is TF-IDF
down-weighting. That fails completely for single-action users: two users whose
only action is the same object have identical one-hot vectors, so their cosine is
1.0 whatever the IDF weight, because normalisation cancels it. Measured: 54.8% of
co_retweet users acted on exactly one entity, one viral tweet drew 308 of them
into a clique, and **the top 100 accounts shared a single centrality value**
(1/sqrt(308) = 0.056946). At a floor of 2 the top 100 carried 90 distinct values.
This is a partial vindication of v1, whose hub cap defended against exactly this.

The floor initially covered only the four bipartite traces, so single-post
accounts re-entered through text similarity: **81 of the top 500 had exactly one
post**, and that group averaged 0.223 Kenya share against 0.416 for the rest.
Extending the floor to text similarity raised the top 500's mean Kenya share from
0.373 to 0.404 and the median from 0.374 to 0.456.

**The text threshold.** The paper reports 0.95 as the 96th percentile of its
similarity distribution. Over all in-window pairs that makes 4% of every pair an
edge: measured here, the 96th percentile resolved to **0.5198** and produced
**3,116,957 edges from 16,478 posts**. The figure is only coherent as a
percentile over top-k neighbours from an index. We use an absolute 0.85, chosen
because that is where near-duplicates sit in our encoder's space and it yields an
edge count commensurate with co-retweet. **This value is ours, not the paper's,**
and the sensitivity is steep - 3.1M edges at 0.52, 107,625 at 0.70, 510 at 0.85
on comparable samples.

*Checked 2026-09-11: 0.85 is not where near-duplicates sit.* Of the 279,060
cross-author post pairs at or above 0.85 behind the top 500, 3% are near-copies
and the median pair shares no words; a blind reader labelled 100 of them 26 same
message, 15 same topic, 59 unrelated. Short Swahili and Sheng replies cluster in
this encoder's space. A char-4gram word-overlap floor of 0.10 on top of the cut
removes the unrelated pairs while keeping 92% of the reader's matches, but then
the text trace no longer reaches the top 500 at all - see
`analysis/investigations/2026-09-11-textsim-sensitivity/findings.md`.

## 5. What v2 surfaces on the Kenya corpus

Latest run: `coord2/platform=x/kind=scores/dt=2026-09-08/run=20260908T081903Z.parquet`.

| trace | trace rows | users | edges |
|---|---|---|---|
| co_retweet | 364,287 | 11,608 | 371,259 |
| text_similarity | - | 11,764 | 244,308 |
| hashtag_sequence | 13,516 | 283 | 1,366 |
| co_url | 55,134 | 495 | 1,103 |
| fast_retweet | 1,463 | 60 | 222 |

Top 500 by centrality: **mean Kenya share 0.404, median 0.456, 353 of 500 at or
above the 15% gate** the collector already uses for promotion.

For contrast, v1's *strongest* evidence tier - cross-channel corroborated
clusters - was 8.3% Kenya-referencing, which is what discredited it. On the same
instrument v2's top 500 is roughly five times more on-topic.

**v1 and v2 disagree almost completely.** Against v1's latest run (1,314
clustered accounts), the v2 top 500 overlaps by **22 accounts, 4.4%**; before
text similarity it was 0. Both read the same corpus. Neither has ground truth.
Note that the sticky-union reader `latest_coordination_clusters` inflates v1 to
13,767 accounts and would report a flattering 30.6% - use
`coordination_run_latest`.

## 6. The relevance gate, finally measured

`kma.measure.domain_bucket` gates cluster promotion in the collector and produced
every Kenya-share figure above, and its error rate had never been measured.
Stratified sample of 300 posts, blind-labelled, scored with stratum weights:

| | |
|---|---|
| precision | **0.980** (95% CI 0.930-0.994) |
| recall | **0.508** |
| F1 | 0.669 |
| usable labels | 269, plus 31 `unclear` |

When it says Kenya it is right 98 times in 100. It **misses about half of all
Kenya-relevant posts**, because 28.6% of the `ambiguous` bucket - 74.7% of the
corpus - is genuinely Kenyan. The misses are what a keyword regex misses: Swahili
and Sheng naming no Kenyan entity, replies to Kenyan accounts, coded shorthand.

**Consequence: every Kenya-share figure in this document is a floor, not an
estimate.** The v2 top 500's 0.404 is consistent with a true share near 0.8.
v1's 8.3% corrects to roughly 16%. Relative comparisons survive, because both
were measured with the same instrument; absolute values should not be quoted.

**Provenance:** these are model labels (`claude-opus-5`), recorded in the sample's
`label_source` column. The measurement is "how far the regex agrees with a
careful reader", not human ground truth. That is a real and independent check -
the labeller reads context, the gate matches keywords - but it is a weaker claim,
and this project has been burned before by treating LLM labels as truth.

**Human check, 2026-09-11 (finishing-the-revamp B2).** Tom labelled 100 of the
300 posts blind: stratified across the gate's buckets, with neither the bucket
nor the model's label shown, and a literal machine translation beside each
non-English post. He and the model agree on **90 of 100 (Cohen's kappa 0.83)**,
and on **every Kenya call** - all ten disagreements are `offdomain` against
`unclear`. On those posts:

| | against Tom's labels | against the model's, same posts |
|---|---|---|
| precision | **0.971** (95% CI 0.851-0.995) | 0.971 |
| recall | **0.655** | 0.606 |
| usable labels | 91, plus 9 `unclear` | 87, plus 13 `unclear` |

The recall gap on the same posts comes only from which ones each labeller set
aside as unclear. So the model labels hold up as a measure of Kenya relevance,
the precision figure is now human-confirmed, and the model-based recall is, if
anything, slightly conservative. The 100-post subset carries no recall interval
and is not the same sample as the 300, so the full-sample 0.508 is not replaced
by it; the "floor, not an estimate" reading above stands. Tooling:
`kma-measure-eval human-sheet` and `agree`.

## 7. What remains unresolved

- **Which method is right.** v1 and v2 agree on 4.4% of accounts. No statistic
  we hold can adjudicate that; it needs a reader with dossiers, which is what the
  detect/filter/adjudicate architecture said from the start.
- **Iran**, +10.6 against the published benchmark, cause unknown.
- **Trace construction is untested.** The gate ships prebuilt networks, so the
  5/6 pass covers fusion and detection only.
- **The 0.85 text threshold is ours**, and it determines most of the ranking.
- **Account depth.** The benchmark's real structural advantage is complete
  account histories. Deep timelines reach ~3,200 tweets per account, well past
  the 14-day search horizon, so this is the one collection change that could move
  our data toward the shape the method was built for.


---

## 8. A2: the v1/v2 disagreement, adjudicated

Run 2026-09-09. 14 dossiers, 7 per method, shuffled and read **blind** - the
reader did not know which method produced which case until after every verdict
was recorded.

| method | cases | political | strongly coordinated-political |
|---|---|---|---|
| v1 | 7 | 1 (14%) | 1 (14%) |
| **v2** | 7 | **6 (86%)** | **4 (57%)** |

**v1's output reads as engagement farming.** Its largest clusters co-retweet
generic good-morning bait across many different authors (53 and 99 accounts),
global football (168), Hindi-language and camping content (89), and motivational
quotes (30). One of seven was political.

**v2's output reads as Kenyan political coordination.** Pro-Sifuna campaign
amplification carrying `#SifunaWaKenya` and "SISI NDIO SIFUNA" (27 and 18
accounts), anti-Ruto activity with `#KasongoHatesKenyans`, `#wantam` and
`#rutomustgo`, amplifying `skmusyoka`, `rigathi` and `citizentvkenya`. Two of the
seven were thin on evidence (4 accounts each) but still on-topic.

This confirms by reading content what the lexicon gate suggested by matching
keywords - v2's top 500 at 40.4% Kenya share against v1's corroborated tier at
8.3% - and it is the first evidence for either method that does not depend on
`kma.measure`.

### Three caveats, and the first is serious

**The sample is not like-for-like.** v1's cases were its seven LARGEST clusters;
v2's were groups of 4-40 members, which deliberately excluded its two largest
blobs of 255 and 145. Engagement pods are precisely the large ones, so this
choice flatters v2. A matched-size comparison is the obvious next experiment and
has not been run.

**n = 7 per method.** Enough to see a 14% against 86% gap, not enough to put an
interval on it.

**These are model verdicts, not human ones** - the same provenance caveat as the
A9 labels. Recorded rather than presented as ground truth.

### What it settles anyway

The disagreement itself is no longer a mystery. v1 and v2 overlap on 22 of 500
accounts because **they are detecting different phenomena**: v1's
significance-tested co-retweet/co-reply channels find reciprocal engagement pods,
which are the most abundant coordination on the platform, and v2's fused
similarity network with centrality finds accounts pushing the same political
material. Neither is malfunctioning.

For this project's purpose - manipulation of the Kenyan election conversation -
that makes v2 the right instrument, with the sampling caveat above kept attached
to the claim.

### One structural finding about v2's output

The top 500 is **one connected component with density 0.491** - half of all
possible pairs among them are edges. It does not decompose at the deployed Leiden
resolution (0.05 gives one group of 500) and shatters entirely above 2.0. Only at
0.5 does it split usefully, into 11 groups sized 255, 145, 28, 19, 16, 8, 7, 6.

So v2's Kenya output is not 500 findings; it is one dense mass plus a handful of
separable groups. Anything reported per-account from it is reporting on a
population that is largely one object.

### Size-matched rerun, 2026-09-11

The first caveat above has now been tested. All 11 v2 groups from a seeded split
of the same top 500 (255, 143, 27, 19, ...) were paired with the nearest-size v1
clusters and read blind by an isolated headless Claude Code reader using the
adjudicator's own rubric. Verdicts persisted as
`coordination/platform=x/kind=verdicts/dt=2026-09-11/run=20260911T161407Z.parquet`.

| method | cases | political | Kenya-relevant |
|---|---|---|---|
| v1 | 11 | 0 | 0 |
| v2 | 11 | 3 (3 of 7 with evidence) | 9 |

**The 6-of-7 political figure above does not survive size-matching.** v2's two
largest groups are Kenyan engagement pods, and its political groups are open
pro-Sifuna campaigning, not concealed operations. What survives is the direction
and the relevance gap: v2 surfaces Kenyan activity (9 of 11) where v1 surfaces
none (0 of 11). v2's four "unclear" cases are exactly the groups whose dossier had
no jointly-amplified content - the dossier is built from co-retweets, so
text-similarity groups arrive without their evidence. Detail:
`analysis/investigations/2026-09-11-textsim-sensitivity/findings.md`.

*Revised the same day.* Those four groups had no evidence to show: the post
pairs linking them are unrelated Sheng replies, so "unclear" was the right call.
Every v2 group in the sample is held together almost entirely by text edges, so
"v2 surfaces Kenyan activity" may be language rather than coordination, and is
not to be quoted until the text trace and the ranking are fixed and A2 is
redone.

*Fixed 2026-09-12.* The text trace now strips @mentions and requires a
word-overlap floor on every pair (`coord2.TEXT_MIN_OVERLAP`, 0.10 char-4gram
Jaccard): 50,085 edges instead of 409,084, rebuilt on Modal and verified against
an independent replay. The ranking was the harder half. The fused Kenya graph is
one giant component, and its leading eigenvector localises on the densest core,
so any top 500 is one block. Replacing the score fails the reproduction gate
(per-component eigenvector 3/6, PageRank 3/6, global 6/6), so the score stays the
paper's and the REPORT changes: Leiden communities over the fused graph, each
ranked by its own leading eigenvector, the 500-account budget spread across 21
communities. Detail: `analysis/investigations/2026-09-12-component-ranking/findings.md`.

*In production the same day.* The weights are in R2
(`models/relevance/<slug>/`), a deployed Modal app scores new posts daily
(`kma-relevance`), and the collector's promotion gate reads the scores too -
measured on pi0 at 5 s and 245 MB for the added join, inside its 600 MB budget.
The serving model is `kenya-relevance-afroxlmr-2026-09-13`, retrained on
mention-stripped text to match how it is served: precision 0.997 / recall 0.980
against Tom's labels, against 0.971 / 0.655 for the keyword gate. It calls 31.8%
of the corpus Kenyan where the gate calls 21.5%. Rebuilt on those scores, the
community listing holds the same 17 communities with listed Kenya share mean
0.909 / median 1.000, against 0.738 / 0.886 under the gate. Plan and the drift
caveats: `docs/plans/2026-09-13-relevance-in-production.md`.

*The learned gate is adopted, 2026-09-13.* `kma.relevance` now decides
`kenya_share` in the analysis layer - the classifier's call where a post has a
persisted score, `measure.domain_bucket` where it does not (posts collected
since the last scoring pass, and any corpus with no scores at all). It reaches
`coord2_run.attach_relevance` and the dossier's Kenya block; both take
`use_model=False` to reproduce the old figures. **Every Kenya-share figure
recorded before this date was computed with the keyword gate**, whose measured
recall is 0.655 against 1.000 for the model, so they understate relevance and
are not comparable with figures computed after it. The collector's own
promotion gate (`CLUSTER_MIN_KENYA_SHARE`, SQL in `adaptive.py`) is unchanged:
it runs on pi0 inside a 600 MB budget and joining a 1.28M-row prefix per pass
has not been measured there.

*A2 redone on the community report, 2026-09-12.* 21 listed v2 communities
against 21 size-matched v1 clusters, read blind: v1 2 political and 3
Kenya-relevant, v2 6 political and 12 Kenya-relevant. No influence operation on
either side and nothing political at high confidence. v1's cases are
engagement pods; v2's are a mix of Kenyan political amplification, off-domain
fan and hashtag campaigns, and pods. "v2 surfaces Kenyan activity, v1 does not"
holds on the fixed pipeline; a third of v2's listing is off-domain, which the
community ordering has to account for.

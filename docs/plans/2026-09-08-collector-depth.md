# Collector depth: parent hydration, then deep timelines

Plan for the two collector changes that serve the v2 methodology, decided
2026-09-08. Evidence: [../analysis/v2-findings.md](../analysis/v2-findings.md).

---

## Why depth, and why these two

Measured on snapshot `2026-09-05-promotion-off`:

| | |
|---|---|
| authors | 331,138 |
| **with exactly 1 post** | **251,171 (75.9%)** |
| with 2 or fewer | 287,244 (86.7%) |
| mean posts per author | 3.2 |
| clearing the 2-entity activity floor | 79,967 |
| with 20+ posts | 8,047 |

Three quarters of the corpus is accounts seen exactly once. That single fact
explains why 54.8% of co-retweet users have one entity, why the activity floor
discards 58% of accounts, why 238,197 users fell below the text floor, and why
the fused graph fragments. v2 needs per-account behavioural history; we have a
wide, one-post-deep sample.

Distinct retweeted objects: **228,272**, of which we hold **61,052** and
**167,220 are missing**. `fast_retweet` needs the original's author and timestamp,
which is why that trace yields 222 edges and is effectively dead.

**Not on this list: URL capture.** Measured as a property of the discourse rather
than the collector - 6.19% of original posts carry a link, 1.52% of retweets, and
inheriting from parents recovers 4,779 rows of 364,287. Collection cannot capture
links users did not post.

**Track B comes after both.** Replay scores policies against the frozen corpus and
by its own definition cannot evaluate a policy that needs data never collected,
which is precisely what both of these are.

---

## The constraint that must not be missed

**`hydrated` is a BASELINE type** (`kma.db.BASELINE_TYPES`). So is `search`,
`timeline` and `replies`. A 167k-row backfill written to `type=hydrated` would
change the baseline composition by more than the 2026-08-06 conversation widening
did - and that widening is what made the raw toxicity series unpublishable, by
moving the mix from 70.7% search / 12.1% replies to 14.3% / 72.6% when replies
carry 3.2x the hate rate.

**Both tasks therefore write NEW partition types, added to `TARGETED_TYPES`:**

- `posts/type=parent_backfill`
- `posts/type=deep_timeline`

Targeted types are excluded from every prevalence denominator by
`latest_posts(con, scope="baseline")`, and an unrecognised type raises rather
than defaulting into a bucket. Coordination reads all types, so v2 still sees the
new data - which is the whole point.

---

## Budget

Pool: 51 active accounts, per-account pacing `uniform(3, 12)` seconds, mean 7.5s.
That is ~8 requests/account/minute, so ~408/minute pool-wide before rate limits.
Rate limiting will reduce this materially; twscrape rotates when an account is
limited.

| task | requests | theoretical floor | assume |
|---|---|---|---|
| parent hydration (all 167,220) | 1 per id, no batch API | ~7 hours | days |
| deep timelines (12,000 accounts x 200 posts) | ~10 pages each = 120,000 | ~5 hours | days |

Both are plausible inside the remaining runway, but only if prioritised and
resumable. Neither may be a single unbounded pass.

`hydrate()` already exists (`collectors/x.py`, `api.tweet_details(int(pid))`,
one id per request - there is no batch lookup through the GraphQL path).
`timeline(account, limit, include_replies=True)` already exists and correctly
uses `user_tweets_and_replies`, because plain `user_tweets` omits replies.

---

## Task 1: parent hydration

**Goal.** Hydrate the missing retweet parents that matter for coordination, so
`fast_retweet` becomes usable and co-retweet entities gain their originals.

**Prioritise, do not drain.** Hydrating all 167,220 in id order wastes the budget
on objects with one amplifier, which carry no coordination signal. Rank by
distinct retweeters already in the corpus and hydrate the **3-100 amplifier band
first** - the same band the census already selects on, for the same measured
reason (420 of 422 objects censused by raw popularity were hubs, and
`validated_edges` discards hubs).

**Todo list**

1. `SELECT` the missing parents with their amplifier counts, windowed and
   projected, staged into a temp table. Must run inside pi0's 600 MB budget.
2. Rank: amplifier count in `[SNOWBALL_BAND_MIN, SNOWBALL_BAND_MAX]` first, then
   by amplifier count descending within the band.
3. `monitor hydrate-parents --limit N` writing `posts/type=parent_backfill`.
4. Resumable ledger (`state/parent_backfill.json`) so restarts do not refetch;
   follow the `follow_crawl.json` pattern.
5. Add `parent_backfill` to `TARGETED_TYPES`; assert an unknown type still raises.
6. Tests: band prioritisation, ledger resume, provenance (rows land in the
   targeted partition and are absent from `scope="baseline"`).
7. Measure one bounded pass on pi0 before any bulk run.

**Acceptance.** A bounded pass writes to the targeted partition, the ledger
prevents refetching, `latest_posts(scope="baseline")` row count is unchanged, and
`fast_retweet` edge count rises from its current 222 on a rebuild.

### Task 1 handoff, 2026-09-08

Built and tested offline; **no collection pass has been run and nothing is
deployed.** `monitor hydrate-parents --limit N [--dry-run] [--status]`,
`kenya_monitor.parent_backfill`, ledger at `state/parent_backfill.json`,
`parent_backfill` added to `kma.db.TARGETED_TYPES`. Collector suite 184 -> 207;
`analysis/tests/test_scope.py` 19 -> 21.

`run_parent_backfill_once` is deliberately NOT in `run_scheduler`'s cycle. At
one request per id the backlog would become the cycle's dominant consumer of
pool budget and starve baseline collection.

**Candidate query, measured.** Four staged statements over two projected
columns, no window functions - the DISTINCT `(repost_of_id, author_id)` pair set
makes latest-snapshot dedup unnecessary, because neither column can change for
a given post id. The largest materialised relation is bounded by parent
cardinality (228,272), not corpus rows (39,183,192), because the held-ids stage
is a semi-join against the already-materialised parent ids. Measured on a local
synthetic corpus of 38,737,818 rows reproducing the real 228,272 / 61,052 /
167,220 split, at `threads=2`: completes with **peak RSS 157 MB and no spill at
`memory_limit=600MB`**, and still completes at 40MB. That covers memory only -
pi0's real cost is dominated by R2 reads, which cannot be measured from a
worktree with no `.env`, so step 7 (a bounded pass on pi0) is still outstanding.

**Two things about this task's premise that the plan got wrong.** Both were
found by reading `kma.coord2` rather than by running anything.

1. **Hydrating parents adds zero `co_retweet` edges.** `co_retweet_traces`
   reads `retweet_post_id` off the RETWEET row; the parent post's own row has a
   NULL `retweet_post_id` and contributes no trace. So "co-retweet entities
   gain their originals" is a readability and dossier gain, not an edge gain,
   and the band-first ranking - which is the co-retweet-shaped prioritisation -
   optimises for a benefit that does not exist.
2. **`fast_retweet`'s entity is the retweeted AUTHOR, not the retweeted post**
   (`fast_retweet_traces`, entity `coalesce(rt.retweet_user_id, orig.user_id)`,
   and `retweet_user_id` is always NULL because the collector never stored it).
   Two users who fast-retweeted *different* posts by the same author form an
   edge, so a parent with ONE amplifier still contributes. The trace needs the
   parent only for `orig.created_at`. That means the trace the plan names first
   is served by VOLUME of parents and is actively deprioritised by the band
   ranking. The two stated goals pull in opposite directions.

**In-band supply is small, and that is arithmetic, not an estimate.** 228,272
objects share 364,287 retweet rows. If x objects hold 3 or more amplifiers then
`3x + (228,272 - x) <= 364,287`, so **x <= 68,007** - at most 30% of the
population before any skew. Under a power law consistent with both totals the
synthetic corpus puts it at 6,777 in-band of 228,272 (3.0%), with 203,452
objects (89%) holding exactly one amplifier. Since the snowball hydrate arm
already ranks by engagement, the 61,052 held parents are density-biased, so the
in-band share of the *missing* 167,220 is lower still. Expect the band-first
head of the queue to be low thousands of ids - hours of pool budget, not days -
after which `hydrate-parents` logs a warning and continues into 1-amplifier
objects. **Run `--dry-run` against live R2 before committing budget: it prints
`missing in band`, and that number decides whether this task is worth days.**

**Not done, deliberately.** No per-pass provenance row. `census_runs/` does not
fit the schema and `collection_runs/` is the rendered-search-query audit trail;
adding a half-fitting row to either is worse than logs. If the bulk pass runs,
give it its own prefix.

## Task 2: deep timelines

**Goal.** Turn one-post accounts into accounts with trace vectors, for the
accounts v2 actually surfaced.

**Target set.** The ~12,000 accounts in the fused graph, ranked by v2 centrality
from `coord2/kind=scores`, not all 331,138 authors. Depth bounded per account
(start at 200 posts, roughly 10 pages) rather than the ~3,200 maximum, because
the marginal value of page 40 is far below that of page 1 on another account.

**Todo list**

1. Read the target set from the persisted v2 scores; fall back to suspicion rank
   if none exist.
2. `monitor deep-timelines --limit N --depth D` using
   `timeline(include_replies=True)`, writing `posts/type=deep_timeline`.
3. Resumable ledger with a refresh TTL, so a second pass extends rather than
   repeats.
4. Add `deep_timeline` to `TARGETED_TYPES`.
5. Tests: target ordering, depth bound, ledger TTL, provenance.
6. Measure one bounded pass on pi0.
7. Re-run the v2 pass afterwards and report the change in: accounts clearing the
   activity floor, fused component structure, and top-500 Kenya share.

**Acceptance.** Accounts clearing the 2-entity floor rises from 79,967, and the
v2 re-run reports its deltas against this plan's baseline figures.

**The feedback-isolation trap.** Deep-timelining accounts *because v2 surfaced
them* inflates their own future centrality - more posts means more chances to
match. v1 already had this problem and solved it two ways: quarantined partitions,
and ranking count-based components within observation-volume strata. `type=deep_timeline`
handles provenance, but **any v2 re-run after this must report whether the newly
deep accounts rose in rank**, because that would be the artefact and not a finding.

---

## Order

1. Task 1, parent hydration - smaller, and unblocks a dead trace.
2. Task 2, deep timelines - the larger structural win.
3. Track B replay harness.

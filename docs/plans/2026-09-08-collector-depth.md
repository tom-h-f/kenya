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
which is why that trace yields 222 edges and is effectively dead. In coverage
terms: **161,146 of 364,287 retweet rows have a held parent, 44%**, and that is
the quantity Task 1 moves.

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

**Goal.** Hydrate missing retweet parents to raise `fast_retweet` coverage from
44%, so that trace becomes usable. Dossiers also gain the ability to show what
was amplified, but note that this adds **no** `co_retweet` edges - see the
handoff below.

**Prioritise, do not drain.** Hydrating all 167,220 in id order wastes the
budget. Rank by **distinct amplifier count DESCENDING, unbanded**.

~~Rank by distinct retweeters already in the corpus and hydrate the 3-100
amplifier band first.~~ **Superseded 2026-09-08.** The band was wrong here.
Hydration serves exactly one trace, `fast_retweet`, whose entity is the
retweeted AUTHOR, so every retweet row whose parent lands becomes a candidate
trace row and the quantity to maximise per request is retweet rows unlocked -
which is the amplifier count. v2 has no hub cap: TF-IDF down-weighting handles
popular entities, and that is the one place the paper's mechanism works as
advertised. The census's reason to band (`validated_edges` discards hubs) does
not carry over, so banding would have forfeited the 10 densest objects in the
whole backlog. `--band-only` survives as an explicit opt-in.

**Todo list**

1. `SELECT` the missing parents with their amplifier counts, windowed and
   projected, staged into a temp table. Must run inside pi0's 600 MB budget.
2. Rank by amplifier count descending, unbanded. (Was: band first. Superseded -
   see above.)
3. `monitor hydrate-parents --limit N` writing `posts/type=parent_backfill`.
4. Resumable ledger (`state/parent_backfill.json`) so restarts do not refetch;
   follow the `follow_crawl.json` pattern.
5. Add `parent_backfill` to `TARGETED_TYPES`; assert an unknown type still raises.
6. Tests: densest-first ranking, `--band-only` as opt-in, ledger resume,
   provenance (rows land in the targeted partition and are absent from
   `scope="baseline"`).
7. Measure one bounded pass on pi0 before any bulk run.

**Acceptance.** A bounded pass writes to the targeted partition, the ledger
prevents refetching, `latest_posts(scope="baseline")` row count is unchanged, and
`fast_retweet` coverage rises from 161,146 of 364,287 retweet rows (44%).

### Task 1 handoff, 2026-09-08

Built and tested offline; **no collection pass has been run and nothing is
deployed.** `monitor hydrate-parents --limit N [--dry-run] [--status]
[--band-only]`, `kenya_monitor.parent_backfill`, ledger at
`state/parent_backfill.json`, `parent_backfill` added to
`kma.db.TARGETED_TYPES`. Collector suite 184 -> 208;
`analysis/tests/test_scope.py` 19 -> 21.

`run_parent_backfill_once` is deliberately NOT in `run_scheduler`'s cycle. At
one request per id the backlog would become the cycle's dominant consumer of
pool budget and starve baseline collection.

**The metric of record is `fast_retweet` coverage, not ids fetched.** The trace
times a retweet against `orig.created_at`, so a retweet row becomes a candidate
trace row only once its parent is in the corpus: **161,146 of 364,287 rows,
44%**, measured on live R2 for snapshot `2026-09-05-promotion-off`. The pass
reports `retweet_rows_unlocked`, `--status` reports `rows_per_request`, and
`--dry-run` prints the coverage its selection would move it to. Watch
`rows_per_request`: the ranking is densest-first over a backlog that is 90.4%
single-amplifier objects, so it decays towards 1.0 and there is little point
continuing once it gets there.

**Amplifier distribution of the 167,220 missing parents**, live R2 2026-09-08:

| amplifiers | objects | share |
|---|---|---|
| 1 | 151,215 | 90.4% |
| 3-100 (the census band) | 5,545 | 3.32% |
| >100 (hubs) | 10 | 0.006% |

The a-priori bound is `3x + (228,272 - x) <= 364,287`, so at most 68,007 objects
could ever hold 3 or more amplifiers; the real distribution is far more skewed
than the bound allows for. Banding would have restricted the whole pass to
5,545 objects AND skipped the 10 densest ones.

**Candidate query, measured.** Four staged statements over two projected
columns, no window functions - the DISTINCT `(repost_of_id, author_id)` pair set
makes latest-snapshot dedup unnecessary, because neither column can change for
a given post id. The largest materialised relation is bounded by parent
cardinality (228,272), not corpus rows (39,183,192), because the held-ids stage
is a semi-join against the already-materialised parent ids. Measured on a local
synthetic corpus of 38,737,818 rows reproducing the real 228,272 / 61,052 /
167,220 split and the missing-amplifier distribution above, at `threads=2`:
completes with **peak RSS 157 MB and no spill at `memory_limit=600MB`**, and
still completes at 40MB. That covers memory only - pi0's real cost is dominated
by R2 reads, which cannot be measured from a worktree with no `.env`, so step 7
(a bounded pass on pi0) is still outstanding.

**Dry-run head under the shipped ranking** (synthetic corpus, `--limit 500`):
the 10 hubs first at 102-115 amplifiers, then down through the band. 500 ids
unlock 4,946 retweet rows, 9.9 rows per request, moving coverage 46.4% -> 47.7%
on the fixture. Run `--dry-run` against live R2 before committing budget.

**What was wrong with the plan's premise**, found by reading `kma.coord2` and
since verified independently by the coordinator:

1. **Hydrating parents adds zero `co_retweet` edges.** `co_retweet_traces`
   reads `retweet_post_id` off the RETWEET row; the parent post's own row has a
   NULL `retweet_post_id` and contributes no trace. "Co-retweet entities gain
   their originals" is a readability and dossier gain, not an edge gain.
2. **`fast_retweet`'s entity is the retweeted AUTHOR, not the retweeted post**
   (`coalesce(rt.retweet_user_id, orig.user_id)`, and `retweet_user_id` is
   always NULL because the collector never stored it). Two users who
   fast-retweeted *different* posts by the same author form an edge, so a
   single-amplifier parent does contribute. That is what makes the band wrong
   here and volume right.

**Feedback isolation is only half solved by the partition.** A hydrated parent
is a post BY its author, so backfilled-parent authors gain activity and can
newly clear v2's 2-entity activity floor. Selection ranks by in-corpus
amplifier count, so the authors who gain most are the ones the corpus already
amplifies most. `type=parent_backfill` keeps the rows out of every prevalence
denominator but does nothing about that ranking artefact: **any v2 re-run after
a bulk pass must report whether backfilled-parent authors rose in rank.** Same
trap as Task 2, reached from the other direction. Recorded in the module
docstring as well as here.

**Not done, deliberately.** No per-pass provenance row. `census_runs/` does not
fit the schema and `collection_runs/` is the rendered-search-query audit trail;
adding a half-fitting row to either is worse than logs plus the ledger. If the
bulk pass runs, give it its own prefix.

## Task 2: deep timelines

**Goal.** Turn one-post accounts into accounts with trace vectors, for the
accounts v2 actually surfaced.

**Target set.** ~~The ~12,000 accounts in the fused graph, ranked by v2
centrality from `coord2/kind=scores`~~ **Corrected 2026-09-08: that is 500
accounts, not 12,000.** `coord2_run.run` takes `.head(top)` before `persist`, so
a persisted run is the top 500 by centrality and the fused population is not
written anywhere. Reaching 12,000 needs a v2 re-run at `--top 12000`. Depth
bounded per account (start at 200 posts, roughly 10 pages) rather than the
~3,200 maximum, because the marginal value of page 40 is far below that of page
1 on another account.

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

**Acceptance.** ~~Accounts clearing the 2-entity floor rises from 79,967~~
**Superseded 2026-09-08 - unreachable from this target set, see finding 1 in the
handoff below.** Restated: the median held-post count of the accounts v2 ranked
rises from 3 towards `depth`, and the v2 re-run reports its deltas against this
plan's baseline figures INCLUDING whether deep-timelined accounts rose in rank.

**The feedback-isolation trap.** Deep-timelining accounts *because v2 surfaced
them* inflates their own future centrality - more posts means more chances to
match. v1 already had this problem and solved it two ways: quarantined partitions,
and ranking count-based components within observation-volume strata. `type=deep_timeline`
handles provenance, but **any v2 re-run after this must report whether the newly
deep accounts rose in rank**, because that would be the artefact and not a finding.

### Task 2 handoff, 2026-09-08

Built and tested offline; **no collection pass has been run and nothing is
deployed.** `monitor deep-timelines --limit N --depth D [--dry-run] [--status]
[--min-kenya-share X]`, `kenya_monitor.deep_timelines`, ledger at
`state/deep_timeline.json`, `deep_timeline` added to `kma.db.TARGETED_TYPES`.
Collector suite 208 -> 246; `analysis/tests/test_scope.py` 21 -> 23.

`run_deep_timelines_once` is deliberately NOT in `run_scheduler`'s cycle. Each
account is ~10 paginated requests, so it would become the cycle's dominant
consumer of pool budget - and this pass conditions collection on v2's own
output, so it must be an explicit, dated, bounded act rather than something the
corpus accumulates while nobody watches which accounts it favours.

**Three things in the plan were wrong. The first invalidates this task's stated
acceptance criterion.**

1. **A pass targeting persisted v2 scores cannot move the 79,967 figure at
   all.** v2's activity floor runs BEFORE centrality (`similarity_network`
   calls `min_activity`, then `detect` scores the fused graph), so an account
   with one post has fewer than 2 entities in every trace and can never appear
   in a `kind=scores` run. Every target already clears the floor. The 251,171
   one-post authors are precisely the accounts v2 cannot rank, so they are
   never in the target set. What this pass buys is vector DENSITY for accounts
   currently ranked on 2-to-19 observations - the ability to adjudicate the
   ranking - and the acceptance criterion has to be restated in those terms.
   Moving 79,967 needs the accounts v2 DISCARDED, which have no centrality to
   rank them by; `runner.census_discovered_handles` is the precedent and its
   `ORDER BY random()` is the argument. That is a separate pass, not a flag.
2. **`collectors.x.timeline` drops everything older than 14 days**
   (`MAX_AGE_DAYS`, applied per post), so "reuse `timeline`" would have quietly
   voided the whole task: a depth-200 pass would have spent ~10 requests per
   account and kept only what a keyword search could already reach. Added
   `deep_timeline(user_id, limit, include_replies=True)` beside it, sharing one
   private implementation, with no cutoff. It is also addressed by NUMERIC ID
   rather than handle, which saves the `user_by_login` request (~9% of the
   per-account budget) and removes a false terminal outcome for every renamed
   account. `timeline`'s behaviour is unchanged.
3. **The target set is 500 accounts, not ~12,000** - see the correction above.
   That makes the whole persisted target set affordable in one pass: 500 x ~10
   = ~5,000 requests, against the plan's 120,000 estimate for 12,000 accounts.

**Targeting rule.** Strata on held post count, centrality DESCENDING within
each stratum, strata ascending: `<2` (below v2's floor), `2..19` (thin),
`20..depth-1`, `>=depth` (saturated). Cost is near-constant per account and
benefit is not, so rank by what depth buys. Both objectives agree on the order:
a high-centrality account with two posts is ranked high BECAUSE of the entities
the floor distrusts, so it is at once the least trustworthy row and the cheapest
to settle. Strata rather than a blended `centrality / log(1 + n_posts)`, which
weights two incommensurable quantities by an indefensible constant, and every
boundary is measured - 2 is `coord2.MIN_ENTITIES_PER_USER`, 20 is where this
corpus's depth distribution breaks (8,047 of 331,138 authors, the 97.6th
percentile), `depth` is the pass's own saturation point.

The 20-post boundary is not decoration. Without it, and given finding 1, the
whole target set falls in one stratum and the ranking collapses to plain
centrality - the synthetic dry-run put a 39-post account above a 2-post one.

**Known imprecision:** strata count POSTS, v2's floor counts distinct ENTITIES
per trace. Five retweets of one object are five posts and one co_retweet entity.
Posts are used because they are what `depth` directly increases, they are
trace-agnostic, and they are what the plan's 79,967 counts (331,138 - 251,171).
Entity counts are measured and reported beside them everywhere.

**Dry-run head, synthetic corpus** (39,387,988 rows over 1,036,526 distinct
posts, reproducing the real 331,138 / 251,171 / 79,967 author-depth
distribution; the target-set depth mix is a fixture assumption, not a
measurement). Corpus-wide the selector reproduces **79,967 of 331,138 clearing
the 2-post floor (24.1%)**, and 42,316 clearing a 2-entity co-retweet floor. At
`--limit 50 --depth 200` it selects 50 accounts holding **2..18 posts, median
3**, all in stratum 1, ~500 requests; 0 of them cross the post floor, so the
corpus-wide 79,967 does not move - which is finding 1, visible in the output
rather than buried. On the fixture 455 of 500 targets are thin and 258 of 500
sit below a 2-entity co-retweet floor. **Run `--dry-run` against live R2 before
committing budget.**

**Candidate query, measured.** Four staged statements, two projected columns per
scan, no window function, no `count(DISTINCT ...)`. Nothing post-level is
materialised: a DuckDB in-memory temp table counts against `memory_limit` and
CANNOT spill, so a 1,036,526-row DISTINCT staged as a temp table peaked at 299
MB and OOMed under a 150MB cap. Collapsing each DISTINCT into its aggregate
(`suspicion._beh_sql`'s shape) costs a second glob scan - the same trade
`parent_backfill._pb_held` takes - and peaks at **274 MB at
`memory_limit=600MB`, threads=2**, completing down to 100MB by spilling. No
materialised relation exceeds distinct authors (331,138). Memory only: pi0's
real cost is R2 reads, unmeasurable from a worktree with no `.env`, so step 6 (a
bounded pass on pi0) is still outstanding.

**Feedback trap.** Recorded in the module docstring, `kma.db.TARGETED_TYPES` and
the CLI help, not only here. Beyond the quarantined partition, every deepened
account is written to a new **`deep_timelines/`** prefix, one row per account,
carrying `user_id`, `source`, `source_run` (which scores object it came from),
`rank_metric`, `rank_value`, `stratum`, `held_posts_before`,
`held_entities_before`, `depth`, `posts_written`, `status`. Keyed on `user_id`,
so the check is a join onto `coord2/platform=x/kind=scores` rather than
archaeology, and the pre-treatment counts are the covariate to condition on.
Its own prefix, as Task 1's handoff recommended after rejecting `census_runs/`
and `collection_runs/`. **Any v2 re-run after a bulk pass must report whether
deep-timelined accounts rose in rank.**

**Resumability is two mechanisms**, because neither suffices. `deepened_expr`
excludes accounts with `type=deep_timeline` posts inside the TTL, in the
candidate SQL before the LIMIT, so it self-heals if the ledger is lost
(`census.censused_expr`'s reasoning). The ledger covers what R2 cannot learn:
an account that returned nothing writes no post row. Note there is NO
self-draining anti-join here, unlike the parent backfill - an account always
has posts, so depth is a TTL question and the ledger is load bearing.
`no_posts` is terminal (a by-id fetch cannot tell suspended from protected from
empty, and none of them change on this timescale); `failed` retries after the
TTL, bounded, following `follow_crawl.is_due`.

**Also fixed while building:** `_score_targets` originally caught
`duckdb.Error`, which silently substituted the suspicion fallback for ANY fault
- including `InvalidInputException` from converting a tz-aware timestamp without
`pytz`, which is not a dependency of this project. Narrowed to `IOException`
(absent prefix, the one recoverable case) and the timestamp is cast to VARCHAR
in SQL rather than fetched.

**Shared-code changes, all additive except one refactor.** `storage.py`,
`collectors/base.py` and `suspicion.py` gained code only; the sole deletion in
the whole diff is three lines of `x.timeline` moved verbatim into the private
`_timeline` helper. `suspicion._score_sql` gained a projected
`platform_user_id AS user_id` column so the fallback keys on ids like the
primary path; its only other consumer joins on `handle` and projects named
columns, so nothing else sees it.

**Not done, deliberately.** No bulk pass, no pi0 deployment, no v2 re-run. Step
7's deltas (floor-clearing count, fused component structure, top-500 Kenya
share) cannot be reported before a pass runs, and per finding 1 the
floor-clearing delta will be 0 from this target set.

---

## Order

1. Task 1, parent hydration - smaller, and unblocks a dead trace.
2. Task 2, deep timelines - the larger structural win.
3. Track B replay harness.


---

# Step 7: measured on pi0, 2026-09-08

Both bounded passes ran on pi0 in one-off containers (`docker run --rm --memory 1g`),
never `docker exec` into the running collector, which shares its 1 GB cgroup.
Live collection was not restarted: neither command is in the scheduler cycle.

## Parent hydration

| | |
|---|---|
| selected / hydrated / not_found / failed | 200 / 199 / 1 / 0 |
| authors written | 141 |
| retweet rows unlocked | **9,124** |
| fast_retweet coverage | 43.2% -> 45.5% |
| runtime | ~27 min (candidate query ~21, fetches ~6) |

**The candidate query costs ~21 minutes on pi0 against live R2.** That is the
number no worktree or laptop run could produce, and it dominates a bounded pass.
Amortise it: a pass of 200 spends 78% of its wall clock on selection.

**Amplifier range reached 20..242**, so real hubs do exist among the missing
parents. The original band-first ranking would have excluded the single most
valuable ids; the unbanded descending ranking reaches them first. At ~45.6 rows
unlocked per request, roughly **3,700 fetches take coverage past 60%** - that,
not 167,220, is the number worth planning against.

`not_found` fired once on real data, exercising the terminal-status path that
stops a deleted parent from re-occupying the queue head forever.

## Deep timelines

| | |
|---|---|
| selected / deepened / no_posts / failed | 20 / 20 / 0 / 0 |
| posts written | **7,974** |
| authors written | 1,531 |
| floor_cleared | **0** |

**The `no_posts` risk did not materialise** (0 of 20). Reading a timeline by
numeric id cannot distinguish suspended, protected and genuinely empty accounts,
and all three are recorded terminal, so a high rate would mean accounts written
off permanently on soft failures. Sample is small; re-check on any larger pass.

**`floor_cleared` is 0, exactly as predicted, and the acceptance criterion in
this plan was wrong.** The activity floor runs BEFORE centrality, so a one-post
account can never appear in `coord2/kind=scores` and can never be a target: every
target already clears the floor. Dry-run over 50 targets: they hold 2..19 posts,
median 6, all in stratum 1, and 0 would cross. What the pass buys is vector
density for accounts ranked on 2-to-19 observations. Moving the corpus-wide floor
count needs the accounts v2 DISCARDED, which have no centrality to rank them by -
a separate random-sampling pass.

**Unexpected, worth chasing:** most dry-run targets showed `entities=0` while
holding 2-19 posts - accounts with 16 held posts contributing no trace entity at
all. They are in the ranking purely through text similarity, which is not
entity-based. Depth on these buys text-similarity density, not bipartite traces.

**Discrepancy: `--depth 200` produced ~399 posts per account** (7,974 over 20).
Depth is not acting as a hard per-account cap, so a bulk pass costs roughly
double the budgeted requests. Resolve before scaling.

## Baseline drift note

The floor-clearing figure moved from 79,967 of 331,138 to 88,134 of 364,151
between planning and this pass - partly hydration's new authors, partly
collection continuing. Any before/after comparison must pin a `bench` snapshot;
an unpinned read cannot support one.

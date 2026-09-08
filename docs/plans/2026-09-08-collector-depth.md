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

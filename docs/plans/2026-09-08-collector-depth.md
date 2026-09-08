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

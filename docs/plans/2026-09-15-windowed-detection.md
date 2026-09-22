# Route 3: detect campaigns as events, not as a three-month average

## The argument

Campaigns are events. The detector runs over a whole snapshot, so a campaign
that ran for six hours is averaged into three months of ordinary behaviour and
disappears.

This is also the route that most directly attacks the failure the 2026-09-14
depth test exposed. That test found 0 of 390 deepened accounts kept a top-500
place against 22.7% for an untouched control: the global ranking was scoring
accounts for having little history. **Burstiness inverts that.** An account with
twenty co-actions inside one hour is a dense local signal, and density is exactly
what deepening destroyed in the global ranking - a habitual retweeter's activity
spreads out under deepening, a campaign's does not.

## The measurement that decides the design

Account-windows clearing the co-retweet entity floor, over the last 60 days of
snapshot `2026-09-14-deep500`:

| window | >= 2 entities | >= 5 | >= 10 |
|---|---|---|---|
| per day | **701** | 310 | 179 |
| per week | 2,855 | 1,397 | 941 |

**Daily windowing is viable.** The fear was that a day would leave almost nobody
above the floor; the answer is 701 accounts at floor 2 and 310 at floor 5. That
is enough to build a graph from.

It also sets the floor/window trade concretely. Floor 10 daily leaves 179
accounts, which is thin; floor 10 weekly leaves 941. So the choice is between a
tighter time resolution with a lower floor and a coarser one with a floor high
enough to escape the one-hot pathology.

## Design

`coord2_run` already takes `--days`, so no new detector is needed.

1. **Window the run.** Daily and weekly variants over the same snapshot.
2. **Score burstiness, not lifetime centrality.** The quantity is the
   concentration of an account's co-action in time - the share of its co-actions
   falling in its densest window, against the number of windows it is active in.
   An account active in one window is bursty by definition and must be handled
   explicitly rather than scoring maximum, which is the same one-hot trap in a
   new coordinate system.
3. **Persist per-window scores** so a campaign has a start and an end. That is
   what makes it reportable as an event rather than as a list of accounts, and
   it is what a story needs.
4. **Keep the global run.** The two answer different questions and the
   comparison between them is informative: an account central globally but never
   bursty is a hub; bursty but never central is a candidate.

## The test this must pass

The same one the global ranking failed, and the apparatus already exists.

Re-run `investigations/2026-09-14-depth-rerank/01_rank_check.py` against windowed
scores using the **existing 97-account holdout**. If windowed rankings retain
treated and untreated accounts at similar rates, the ranking is measuring
behaviour rather than sparsity and the method is usable. If treated accounts
again fall out, windowing has not fixed the underlying problem and the honest
conclusion is that this corpus cannot support account-level ranking at all.

This is cheap because the holdout is already collected. **Do not deepen anything
else to run it** - a second bulk pass would consume the candidates the next
measurement needs, and the 30-day refresh TTL means the arm cannot be rebuilt for
a month.

## Sequencing

Do the `MIN_ENTITIES` sweep first: re-run at floors 5, 10 and 20, reporting
holdout retention at each. It is one number per run, it is the prerequisite for
trusting any ranking windowed or not, and its answer changes which window width
is worth building. The table above says what each floor costs in population.

## Unresolved

1. Daily or weekly as the primary unit? The measurement supports daily at floor
   2-5; weekly is the only option at floor 10.
2. How to score an account active in exactly one window - excluded, floored, or
   flagged? It is the one-hot problem again and it needs deciding before the
   scoring code, not after.
3. Does burstiness need its own benchmark check? The IOHunter datasets are not
   time-resolved in the same way, so the 6-of-6 reproduction does not transfer
   to a burstiness score. This may be a method we cannot externally validate,
   which is worth knowing before it is relied on.

---

## Measured and decided, 2026-09-22

The `MIN_ENTITIES` sweep this plan sequenced first has run, over the last 60
days of `2026-09-14-deep500` (59 complete days, 8 complete weeks, four
bipartite traces - the text trace carries no timestamps and is absent from
every number below). Scripts and result CSVs:
`analysis/investigations/2026-09-22-windowed-floor/`.

### Population by floor

| width | floor | accounts | account-windows | median per window | communities 4+ | largest |
|---|---|---|---|---|---|---|
| day | 2 | 6,532 | 18,795 | 279 | 698 | 295 |
| day | 3 | 4,456 | 13,233 | 206 | 561 | 189 |
| day | 5 | 3,110 | 9,067 | 151 | 432 | 102 |
| day | 10 | 2,066 | 5,261 | 90 | 286 | 91 |
| day | 20 | 1,164 | 2,669 | 45 | 173 | 79 |
| week | 2 | 7,977 | 16,092 | 2,103 | 167 | 461 |
| week | 3 | 5,238 | 10,487 | 1,285 | 139 | 283 |
| week | 5 | 3,989 | 7,384 | 915 | 125 | 338 |
| week | 10 | 3,423 | 5,386 | 695 | 116 | 471 |
| week | 20 | 2,596 | 3,585 | 445 | 108 | 216 |

### Planted campaigns

A synthetic hashtag-for-hire campaign in one day - campaign tweets, hashtag
sequence, authors and URLs inside a six-hour burst - with each planted account
also carrying a real account's organic actions borrowed from another day.
Light is 12 accounts on 4 of 6 campaign tweets each, heavy 20 on 10 of 15.
Three seeds per cell. `recall` is over the community holding most of the plant.

| width | floor | light recall | heavy recall | precision (found) |
|---|---|---|---|---|
| day | 2 | 1.00 | 1.00 | 1.00 |
| day | 3 | 1.00 | 1.00 | 1.00 |
| day | 5 | 0.83 | 0.63 | 1.00 |
| day | 10 | **0.06** | 0.55 | 1.00 / 0.33 |
| day | 20 | 0.00 | 0.03 | 0.33 |
| week | 3 | 1.00 | 1.00 | 0.97 |
| week | 5 | 0.86 | 1.00 | 1.00 |
| week | 10 | **0.06** | 1.00 | 0.93 / 0.33 |

Floor 10 is where a light campaign stops existing in the graph, at both
widths. Precision is 1.00 wherever a campaign is found, so the floor decides
presence, not purity.

### The null, and what it disqualifies

Every window's incidence curveball-shuffled per trace (burn-in 100, both degree
sequences preserved). The shuffled corpus fills the report with about TWICE as
many account-windows as the real one at every floor - v2 keeps a percentile of
realised edge weights and has no significance test, so a null cannot report
zero. Report size is therefore not evidence. What differs is shape: real daily
communities are larger than the null's (295 against 163 at floor 2; 91 against
58 at floor 10), while weekly floors 2 and 3 invert it, the null building a
single community of 1,255 and 1,024 against 461 and 283 real.

### The depth test - PASSED

Re-run on windowed scores over the 59 days both `2026-09-14-deepened` and
`2026-09-14-deep500` cover. An account-window is shown when the account is in a
community of 4+.

| width | floor | arm | shown before | retained | gained |
|---|---|---|---|---|---|
| day | 2 | treated (390) | 77 | **84.4%** | 1,293 |
| day | 2 | holdout (97) | 24 | **100%** | 0 |
| day | 3 | treated | 6 | 83.3% | 2,458 |
| day | 3 | holdout | 2 | 100% | 0 |
| week | 2 | treated | 144 | 70.1% | 159 |
| week | 2 | holdout | 52 | 100% | 0 |
| week | 3 | treated | 40 | 82.5% | 373 |
| week | 3 | holdout | 9 | 100% | 0 |

Against the global ranking's 0.0% treated retention. Deepening ADDS treated
accounts to windowed reports - 1,293 to 2,557 account-windows across 354 to 366
of the 390 - while the holdout gains none. **Burstiness inverts the depth
failure, as this plan predicted.** At floors 5+ neither arm was shown before
deepening, so retention is undefined there and only `gained` is readable.

### Decision

**Detection: daily, floor 3. Series: weekly, floor 5.** The plan's prior of
daily 5 / weekly 10 is overturned on both counts. The criterion is what each
stage can still recover from: a floor that loses a light campaign is
unrecoverable downstream (rules out 10 and 20); of the rest, prefer the report
least explained by degree structure (rules out weekly 2 and 3); among daily
floors keeping the light plant, 2 is the value the depth test condemned for the
global ranking and 5 already costs a sixth of the light plant and two fifths of
the heavy one, leaving 3.

`kma.windowed.DETECTION` and `kma.windowed.SERIES` carry these.

### Unresolved questions, answered

1. **Daily or weekly as primary?** Daily, floor 3. Weekly stays as the series,
   at floor 5.
2. **One-window accounts?** Unchanged: `burst.concentration` refuses to score
   an account below `MIN_WINDOWS`, and the sweep scores communities per window
   rather than ranking accounts across windows, so the question does not reach
   the detection path.
3. **Does burstiness need its own benchmark?** It cannot have one. The IOHunter
   datasets are pre-built graphs with no usable time axis, so the 6-of-6
   reproduction does not transfer and recall here rests on synthetic plants.

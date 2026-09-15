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

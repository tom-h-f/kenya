# Finishing the revamp: outstanding items

Written 2026-09-08. What is left between here and a v2 revamp that can be called
done, in dependency order rather than wish order.

Context: [../analysis/v2-findings.md](../analysis/v2-findings.md),
[2026-09-05-v2-next-steps.md](2026-09-05-v2-next-steps.md),
[2026-09-08-collector-depth.md](2026-09-08-collector-depth.md).

---

## Is adjudication the only thing left?

No, but it is the only thing that unblocks a *finding*. Everything else on this
list improves an instrument; adjudication is what turns instrument output into a
claim someone can act on.

The reason is structural rather than a gap to be closed by more work.
Coordination statistics answer *do these accounts act together*. They cannot
answer *why*, and that was settled here on 2026-08-17 as a property of the
method, not a tuning problem. Right now v1 and v2 agree on **22 of 500 accounts**
and no statistic we hold can adjudicate between them.

So: **P0 is adjudication. Everything else is P1 or lower**, and two P1 items are
correctness gaps serious enough that a finding published without them would be
fragile.

---

## P0: adjudication

**A1. Port adjudication to Modal.** `kma.adjudicate` and `kma.dossier` exist,
were live-verified on tf1, and are dormant. The decision of 2026-09-08 is that no
`ANTHROPIC_API_KEY` goes on tf1 and model work runs on Modal, so this is a new
Modal app reading `coord2/kind=scores` and writing verdicts - not a revival of
`ADJUDICATE_REFRESH_HOURS`.

Acceptance: a bounded pass adjudicates N clusters or account groups and writes
verdicts to R2 with the dossier that produced each one.

**A2. Adjudicate the v1/v2 disagreement.** Sample from each side - say 20 from
v2's top 500 and 20 from v1's latest clusters - and have them read blind. This is
the experiment that decides which method to trust, and it cannot be run any other
way. The IO ground-truth work established the shape: a reader with dossiers got
4/6 operations and 0/6 false alarms, and the two misses were the operation's own
engagement-bait accounts.

Acceptance: a verdict per sampled account, blind to which method surfaced it,
with agreement reported per method.

*Rerun size-matched 2026-09-11.* 11 v2 groups against the nearest-size v1
clusters, read blind by headless Claude Code: v1 0 of 11 political and 0 of 11
Kenya-relevant; v2 3 of 11 political and 9 of 11 Kenya-relevant. The 2026-09-09
6-of-7 political does not survive; the relevance gap does. Follow-up:
`kma.dossier` must show text-similarity co-action, because all four of v2's
unclear verdicts were groups with no co-retweet evidence. See `v2-findings` §8.

*Superseded the same day.* Building that exhibit showed the four groups have no
co-action to show: at 0.85 most text-trace pairs are unrelated Sheng replies.
A word-overlap floor fixes the pairs, but then eigenvector centrality hands the
whole top 500 to one co-retweet block and the real paraphrase campaigns drop
out. **A2 is blocked on two fixes, not one: the text trace's pair precision and
the ranking.** Neither result from the size-matched run stands until both are
done and A2 is redone. See the textsim investigation's `findings.md`.

*2026-09-12: both fixes made.* Text trace: mention stripping plus a word-overlap
floor, rebuilt on Modal. Ranking: the paper's score is kept (alternatives fail
the gate) and v2 reports by community instead of one global top 500. A2 redone
on the community report: see the component-ranking investigation's
`findings.md`.

**A3. Unit of judgement - DECIDED 2026-09-08.**

**Judge the campaign. Score the account.**

The two are different questions and the confusion between them is what made this
item look hard. The unit a reader can actually judge is the campaign; the unit a
detector emits, and therefore the unit any v1-vs-v2 comparison must be scored on,
is the account.

The evidence is already in `coordination.campaigns`, and it is a measured
failure rather than a preference. Adjudicating confirmed information operations
on 2026-08-15, a reader called 4 of 6 correctly, and **both misses were the
operation's pure engagement-bait clusters** - content like "#NBSKatchup Am
following the first 100 people to Retweet", which is genuinely
indistinguishable from the reciprocal pods in our own corpus. They are not
judgeable on their own content, and no statistic recovers them either, because
an operation runs bait assets deliberately alongside its political ones: cluster
IO-0 carries Jumia giveaways AND #MuhooziOurNextPresident in one group.

So a per-account verdict asked in isolation is unanswerable for a predictable
and important subset of the accounts that matter most. What connects bait to
political assets is who they amplify, which is exactly what `campaigns` groups
on.

**How this resolves the v1/v2 comparison.** Verdicts are rendered on campaigns,
then inherited by member accounts for scoring, which makes v1's clusters and
v2's accounts commensurable: both reduce to a set of accounts carrying an
inherited verdict. `coordination.inherit_verdicts` already implements
inheritance and does so **one-directionally**, which is the right constraint - a
campaign judged coordinated lends that to its members, but a member judged
organic does not clear its campaign.

**What this means for A1 and A2.** The adjudication pass takes the unit as a
parameter and defaults to campaign. A2 samples accounts from each method, groups
them into campaigns, judges the campaigns blind, and reports agreement per
method at the account level. Anything that asks a reader for a bare per-account
verdict is asking a question the 2026-08-15 result says cannot be answered.

---

## P1: correctness gaps that would make a finding fragile

**B1. Trace construction is untested.** The benchmark gate passed 5/6 - but the
IOHunter release ships **prebuilt** trace networks, so the gate exercised our
fusion and detection and never touched our trace builders. Every problem found on
the Kenya side lived in trace construction: the single-action clique, the text
floor, the hashtag-order question. Build traces from the raw archive for one
campaign and compare against the shipped networks.

*2026-09-11: blocked as specified.* The release cannot support the comparison:
`data.zip` is 56 entries, all under `data/processed/`, and the shipped nodes are
bare integers with no attributes and no id mapping anywhere, so they cannot be
matched to archive accounts. With campaign attribution in the archive still open
(next-steps A2 check 4), the routes left are asking the authors for the node-id
mapping, or comparing aggregate structure per trace once attribution is solved.
See `analysis/investigations/2026-09-11-iran-divergence/findings.md`.

**B2. The relevance-gate measurement rests on model labels.** Precision 0.980 and
recall 0.508 come from 300 posts I labelled, recorded as
`claude-opus-5 (model labels, not human ground truth)` in the sample's
`label_source`. Every Kenya-share figure in the project is now corrected by that
recall, so it is load-bearing. A human pass over even 100 of the 300 would
convert it from indicative to defensible, and this project has been burned by
treating LLM labels as truth before - though on coded incitement, which is much
harder than topical relevance.

*Done 2026-09-11.* Tom's blind pass over 100 of the 300: agreement with the
model 90 of 100, kappa 0.83, and no disagreement on any Kenya call. Gate
precision 0.971, now human-confirmed; recall on the subset 0.655 against his
labels and 0.606 against the model's on the same posts. See `v2-findings` §6.

**B3. Iran, +10.6 Macro-F1 above the published benchmark.** Being better than the
reference is a divergence. Seed, convergence settings and isolated-node handling
are all ruled out. Any Iran figure from this pipeline is suspect until explained.

---

## P2: instruments that need time rather than work

**C1. Track B per-id reproduction.** Currently recall 0.683 after the clean-pass
correction, against a bar of 0.95. The blocker is evidence, not code: 69.4% of
objects are re-censused on a 12-hour TTL, and replay infers TTL from engagement
writes, which fails for censused-but-empty objects.

`census_ttl/` now captures per-object selection every pass (deployed
2026-09-08). Needs: **one code change** - `seed_ledger` to prefer `census_ttl/`
over inference - plus **two weeks of accumulation** (~3 passes/day, ~22% clean,
so ~25 clean passes), then a fresh snapshot and a re-run.

**The current snapshot can never pass.** Its TTL working set was a 24-hour cache,
never captured, gone permanently. Passing is only available prospectively.

**C2. Bulk depth passes.** Both are bounded-tested only: 200 parents of 167,220,
and 20 accounts of 500. Hydration is worth ~3,700 fetches to take `fast_retweet`
coverage past 60%; deep timelines is ~5,000 requests for the whole persisted
target set. Both must re-check their terminal-status rates at scale, and any v2
re-run afterwards **must report whether deepened accounts rose in rank** - the
`deep_timelines/` prefix records pre-treatment state so that check is a join.

---

## P3: deferred by decision, not oversight

*2026-09-11: part of the reason for deferring these was a weeks-long collection
horizon, and that is gone - collection continues until August 2027 (see
next-steps Track C). The dashboard and the one-post authors come back after
P0-P2. Elmas-style forensics and IOHunter stay separate by methodology choice,
which never depended on the horizon.*

- **Dashboard.** Blocked on an R2 token scoped to both buckets and the Zero Trust
  application, both unrequested by decision until v2 has something to publish.
- **The accounts v2 discards.** 251,171 one-post authors can never be ranked by
  centrality, so moving the activity-floor count needs a random-sampling pass,
  not a targeted one. `census_discovered_handles`' `ORDER BY random()` is the
  precedent.
- **Elmas-style trend forensics.** Closest published method to Kenya's documented
  attack pattern; deliberately a separate track, never merged into v2.
- **IOHunter.** A drop-in successor once the fused network exists.

---

## Order

1. A3 (decide the unit) - cheap, and A2 is uninterpretable without it.
2. A1 (Modal adjudication pass).
3. A2 (adjudicate the disagreement). **This is the finish line for the revamp.**
4. B1 and B2 in parallel with the above; both are independent.
5. C2 when pool budget allows; C1 when the two weeks have elapsed.

B3 is a research question with no deadline. P3 stays deferred.

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

**A3. Decide the unit of judgement.** v1 judged clusters, v2 predicts accounts,
and the IO work concluded the right unit is probably the *campaign*. This has to
be settled before A2's results mean anything, because a per-account verdict and a
per-cluster verdict are not comparable.

---

## P1: correctness gaps that would make a finding fragile

**B1. Trace construction is untested.** The benchmark gate passed 5/6 - but the
IOHunter release ships **prebuilt** trace networks, so the gate exercised our
fusion and detection and never touched our trace builders. Every problem found on
the Kenya side lived in trace construction: the single-action clique, the text
floor, the hashtag-order question. Build traces from the raw archive for one
campaign and compare against the shipped networks.

**B2. The relevance-gate measurement rests on model labels.** Precision 0.980 and
recall 0.508 come from 300 posts I labelled, recorded as
`claude-opus-5 (model labels, not human ground truth)` in the sample's
`label_source`. Every Kenya-share figure in the project is now corrected by that
recall, so it is load-bearing. A human pass over even 100 of the 300 would
convert it from indicative to defensible, and this project has been burned by
treating LLM labels as truth before - though on coded incitement, which is much
harder than topical relevance.

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

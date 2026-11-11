# Leads and canary without a model

Decided 2026-11-11. The leads pass and the canary were built in September
(`2026-09-22-detection-readiness.md`, workstreams B and C) and never deployed;
retiring v1 on 2026-10-10 removed the only alert sender. Nothing told anyone
when the daily pass surfaced a group.

## Decisions

1. **No Anthropic API.** The model reader is replaced by a rule
   (`kma.leads_rule`): kept, and template-labelled or at least half
   non-retweet trace labels. `modal_leads.py` stays in the repo, undeployed.
2. **Everything on `cloud`.** `kma-leads --rule` runs on the daily's
   `OnSuccess=`; the canary is a weekly timer. No Modal.
3. **Every new or changed group is checked.** The 10-per-day budget was the
   model's cost bound.
4. **A 14-day memory.** Without it the last 20 runs give 29 alerts; with it 23.
5. **Size bands.** 8+ accounts push (warning), 4-7 go to IRM as info. 6 of
   the 23 backtest alerts were 8+, on 5 of 20 days.
6. **Canary in memory, never in the live daily.** The daily's `--canary` would
   put the planted group into the public dashboard and explorer counts.

## Open

- Whether flagged groups are campaigns. Nobody has labelled any; the 0.5
  share and the 8-account line rest on the backtest counts only.
- The rule never reads concealment evidence: `concealment.KEPT` is empty, no
  feature has passed its IO-archive test.
- Hashtag-sequence groups: 0 in every one of the 20 runs. The trace matched to
  Kenya's documented threat still has too little data to form a community.

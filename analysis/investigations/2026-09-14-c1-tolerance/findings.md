# C1: closing the three remaining tolerance terms, or closing C1

Per-id reproduction of the census selector sits at recall 0.641 against a 0.95
bar, after the toxic arm was ported and merged passes became scoreable
(`docs/plans/2026-09-08-finishing-the-revamp.md`). `reproduce`'s docstring lists
six tolerance terms; three of them now have evidence behind them and can be
tested rather than argued about.

## The three, and what each one is

**Pass timing (term 6).** `census_pass_times` dates a pass by its FIRST
engagement write, which is minutes after selection ran - the census fetches
before it flushes. The replay therefore stands late and can see posts the live
selector could not. Nothing records the true selection instant, so the test is a
sweep: stand progressively earlier and watch recall.

`census_runs.collected_at` does not help here, despite looking like it should.
It is stamped when the pass FINISHES, after fetching, so it is later than the
first engagement write rather than earlier.

**Empty fetches (term 4).** An object whose retweeter fetch returned nothing
writes no engagement row, so `observed_fetches` cannot see it and the replay is
charged for selecting it. `census_ttl/` records selection independently of
outcome - that is why it was captured - and has covered every pass since
2026-09-08. `--ground-truth census_ttl` scores against it.

**Rate-limit truncation (term 3).** A truncated live pass fetched a prefix of
its selection; the replay always fetches all of it, so the tail is scored as
replayed-only. `census_runs.fetched_retweeted` is the collector's own count of
what landed, and `--truncate` cuts the replayed selection to it. A prefix rather
than a sample, because both sides order by `repost_count` descending.

## Results

All three terms were instrumented, and all three turned out to be closed to
measurement with the evidence that exists. Runs at 10 passes (the 10 most
recent, where `census_ttl` coverage is best), snapshot `2026-09-13-c1-replay`.

| variant | recall | precision | Jaccard | replayed | observed |
|---|---|---|---|---|---|
| baseline (25 passes, for reference) | 0.641 | 0.587 | 0.442 | 7,561 | 6,926 |
| baseline (10 passes) | **0.751** | 0.635 | 0.524 | 2,973 | 2,514 |
| ledger ground truth | 0.095 | 0.166 | 0.064 | 2,973 | 5,197 |

### Term 4, empty fetches: the evidence does not separate the arms

The ledger switch scored far WORSE, and the reason is not subtle once the
numbers are put beside the collector's own counters. Over those 10 passes
`census_runs` records 2,795 `selected_retweeted` and 2,500
`selected_conversations`; the ledger returns 5,197. It is both arms, and
`CENSUS_TTL_SCHEMA` carries no channel column, so there is no way to take one.

Scoring a retweeted-arm policy against retweeted + conversations is scoring it
against a target set half of which it never selects from. The 0.095 is an
artefact of the ground truth, not a property of the port.

**The first run of this variant hid the problem rather than showing it.**
`observed_selections` clipped the ledger at the pass boundary, but a pass's
ledger is written at the END of that pass - usually after the next one has
started - so every window came back empty and `reproduce` fell back to
engagement writes on 10 passes of 10, reporting numbers identical to the
baseline. A switch that silently does nothing and reports "no change" is worse
than one that fails, and it took comparing `observed` counts to notice.
Fixed, with a regression test.

### Term 3, rate-limit truncation: measured, and small

`fetched_retweeted / selected_retweeted` over the same 10 passes: 0.96, 0.98,
0.99, 1.00, 0.84, 0.68, 0.96, 0.88, 0.98, 0.99. Two passes lost a real share
and the rest lost a few percent. Truncation is worth single-digit percentage
points of recall, not the 0.25 the gap needs.

### Term 6, pass timing: unmeasurable, not unmeasured

Nothing records the instant selection ran. `census_pass_times` uses the first
engagement write, which is minutes late; `census_runs.collected_at` is stamped
when the pass FINISHES, so it is later still, not earlier. The only approach is
a blind sweep of offsets, and with terms 3 and 4 accounting for so little there
is no reason to believe an offset closes a 0.25 gap.

## C1 is closed: policy replay is indicative, not exact

Recorded rather than left in between, which is what the item asked for.

**What the harness can be used for.** Comparing candidate collection policies on
direction and magnitude - edges per request, unscorable share, yield - against
the same frozen snapshot. Its one correctness property, no leakage, is intact
and independently tested, and it does not depend on the incumbent reproducing
itself per id.

**What it cannot be used for.** Any claim that a policy would have fetched
specific objects, or any per-id agreement figure presented as validation.

**What would reopen it**, in the order that matters:

1. **Record the arm in `census_ttl`.** One field beside `object_id` in
   `storage.write_census_ttl`, and term 4 becomes measurable. Everything else
   here is downstream of that one missing column.
2. Then re-measure terms 3 and 6 against a ground truth that means something.

Both need weeks of accumulation before they can be scored, which is the same
shape as this item's original premise - and that premise was already falsified
once by waiting. So this is a note about what future evidence would buy, not a
plan to wait for it.

## A note on where this ran

Three attempts died for reasons that had nothing to do with the work: three
concurrent R2 readers took each other down on DNS and connection failures; a
single local run was killed by the mac's low-memory watchdog after 90 minutes
with nothing saved; and a fan-out of eight Modal variants sat queued for hours
because each requested 16 GiB when the job peaks near 1.5 GiB - an over-request
is a scheduling penalty, not free headroom, and it exhausted the workspace
spend limit.

`02_variant.py` is the response: one variant per invocation, recorded to
`out/variants.json` the moment it finishes, so a kill costs one variant.

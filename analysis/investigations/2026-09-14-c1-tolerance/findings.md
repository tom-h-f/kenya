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

(pending)

## If the three together do not clear the bar

Close C1 explicitly rather than leave the harness between states: record that
policy replay is indicative rather than exact, and that candidate collection
policies are scored on direction rather than per-id agreement. The harness has
one correctness property - no leakage - and that property is independent of
whether the incumbent reproduces itself.

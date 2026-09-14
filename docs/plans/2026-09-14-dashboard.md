# The dashboard, and why it needs less than it was blocked on

C4 has been deferred since 2026-09-05 with two asks attached: an R2 token scoped
to both buckets, and a Cloudflare Zero Trust application. Neither was requested,
because v2 had nothing to publish. It does now -
`docs/analysis/2026-09-13-publishable-statistics.md` names four things - and the
shape of what is publishable makes one of the two asks unnecessary.

## What goes on it

Exactly the four things A7 found publishable, and nothing that reads like a
fifth:

1. **The benchmark reproduction.** `kma.iohunter_gate` against the six published
   IO datasets, 6/6 within two published SD, with Iran judged against the
   reference code's 71.31. This is the only claim on the page that rests on
   someone else's ground truth, so it goes first.
2. **The relevance classifier's measured error.** Human-validated 0.889/0.827
   weighted against the keyword gate's 0.973/0.516, with the validation sample
   size beside it.
3. **What the detector surfaces, as its own output.** The community listing at
   the 0.5 relevance floor, and the A2 adjudications with their verdict
   distribution. Labelled as what the method surfaced, never as what exists.
4. **The negative results.** Nothing adjudicated as an influence operation;
   nothing political at high confidence; and the depth finding - deepening an
   account makes it fall out of the ranking, so a ranking over a sparse corpus
   is provisional until its head is deepened.

Not on it, and the page should say so rather than leave the gap for a reader to
fill: any prevalence rate. That waits on the control arm
(`2026-09-14-control-arm.md`).

## The architecture, which is simpler than the one it was blocked on

Everything above is a small table or a short series. None of it needs a live
query against the corpus, and none of it changes faster than a re-run.

So: an **exporter** in the analysis project writes one figures file (a few tens
of KB of JSON) from a pinned snapshot, and a **static page** renders it. The
exporter runs where R2 credentials already are - tac2 or Modal - and the page
holds none.

That removes the dual-bucket R2 token from the critical path entirely. The token
existed to let a server-side dashboard read both buckets at request time; a page
that renders a pre-computed file never reads R2 at all.

What is still needed, and it is one ask rather than two:

- A Cloudflare Pages project, with a Zero Trust Access application in front of
  it if the page is not to be public.

Whether it should be public is the decision to make before building it. The four
items above are publishable in the sense of "defensible", which is not the same
as "should be on the open internet under this project's name while collection
continues to August 2027" - a public page naming surfaced accounts changes what
the collection is, and the A2 verdicts are model judgements about real people.
Recommendation: Access-gated, and revisit at publication time.

## Provenance on the page itself

Every figure carries the snapshot id and the run id it came from, because the
whole argument of this project is that a number without its collection context
is not a number. The exporter refuses to write a figure it cannot stamp.


## Built 2026-09-14, and what item 1 changed about it

`kma.figures` (exporter), `analysis/figures/recorded.yaml` (the transcribed
numbers), `analysis/figures/index.html` (the page). No deployment: see the open
decision below.

**The ranking result leads the page now, and the rest reads differently for
it.** The plan above put the benchmark reproduction first. That was written
before the depth test, which found that deepening an account removes it from the
ranking: 0 of 390 treated accounts kept a top-500 place against 22.7% for an
untreated holdout drawn from the same list by the same rule. That is the
strongest thing here - a measurement of our own method failing, under a control
arm, which no collection artifact can flatter - and it constrains how the
adjudication section can be read, so it goes first and the adjudication is
explicitly labelled as a description of the method's output.

**Two halves, and the split is the discipline.** Live figures (corpus shape,
ranking stability) are recomputed from R2 on every export, because a
transcribed number goes stale silently. Recorded figures come from
`recorded.yaml` and `export` refuses any entry missing `source` or `measured` -
a transcribed number without its provenance is not publishable.

The exporter and the check script disagreed on their first run - 391 treated
with 1 survivor against 390 with 0 - because the exporter counted accounts
deepened before the earlier snapshot, which are treated in BOTH runs and belong
to neither arm. Both now window treatment on the two runs' snapshots. Two
published numbers differing by one survivor is exactly the kind of thing that
discredits a page, so the windowing rule is stated in both places.

**The control scope reads 0 posts and that is correct.** The arm went live on
2026-09-14 and the snapshot predates its first write. It appears as a scope with
nothing in it rather than being hidden, because a reader should be able to see
it is coming.

## The one decision left

Whether the page is public or behind Cloudflare Access. The four figures are
defensible, which is not the same as belonging on the open internet under this
project's name while collection continues to August 2027: the page names a
method that surfaces real accounts, and the adjudication verdicts are model
judgements about real people. Recommendation stands - Access-gated, revisit at
publication time. Nothing is deployed until that is decided.

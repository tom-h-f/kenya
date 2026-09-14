# A7: what this project can publish, and what it cannot

Written 2026-09-13, after the v2 revamp closed. A7 asks for "a small number of
measurements that survive the criticism that they are artifacts of how the data
was collected". This is the audit: every candidate number, what it rests on, and
the verdict.

The test applied throughout is the one the objective sets - for each series,
**is it standardised or within-partition, and does its window start at
collection start rather than collection start minus the search horizon** - plus
one this project learned the hard way: **does it survive the corpus changing
shape underneath it**.

## The three failure modes, all of them observed here

1. **Composition drift.** The collector edits its own target list, so the
   baseline mix moves. Between July and August 2026 it went from 70.7% search /
   12.1% replies to 14.3% / 72.6%, and replies carry 3.2x the hate rate. The raw
   toxicity series appeared to rise 78%; standardised to the earlier mix it was
   flat then falling.
2. **Sticky readers.** Two coordination helpers unioned every historical pass
   instead of the latest - 42.7x inflation on clusters, 1.39x on edges. Fixed
   2026-09-13: the cluster helper is deleted (no correct call existed) and the
   edge helper is no longer a package export, so neither can be picked up by
   accident. Nothing in the producer path used either.
3. **A measured relationship moving because the corpus moved, not the thing.**
   The model-versus-gate disagreement rate looked like model drift and was
   collection composition; see the drift caveats in
   `plans/2026-09-13-relevance-in-production.md`.

## Publishable now

**Method reproduction.** 5 of 6 IOHunter benchmark datasets reproduce the
published unsupervised numbers, four within 0.2 Macro-F1. Iran diverges because
the benchmark disagrees with itself: IOHunter's own code on its own release
gives 71.31 where the paper says 60.83, and ours gives 71.43. Publishable as a
claim about the copy, with the Iran caveat attached, never as a Kenya result.

**The relevance gate's error rate, human-measured.** On 88 hand-labelled posts
drawn fresh and judged blind: keyword gate precision 0.973 / recall 0.516
corpus-weighted, learned classifier 0.889 / 0.827. Publishable because the
labels are human, the sample is stratified and the weighting is stated. The
honest framing is "our Kenya filter finds about 83% of Kenyan posts at about
89% precision", not "X% of the corpus is Kenyan".

**What the detector surfaces, as a description of its own output.** The
community report lists 17 communities and 328 accounts at a 0.5 relevance
floor; blind adjudication called 5 of 17 political and 12 of 17 Kenya-relevant,
against 1 and 3 for v1, with no influence operation on either side. Publishable
as "this is what the method surfaces on this corpus, and here is what a reader
made of it". NOT publishable as prevalence: there is no denominator.

**Negative results.** The refuted self-amplification discriminator (confirmed
operations self-amplify MORE than our pods, d = +0.78), the text-similarity
threshold finding (at 0.85 in this encoder most admitted pairs are unrelated
Sheng replies), and the composition-drift result itself. These are the most
defensible things here, because each is a measurement of our own method
failing, which no collection artifact can flatter.

## Not publishable, and why

**Any prevalence rate.** "X% of Kenyan election talk is toxic" needs an
unbiased denominator and nothing in the collector samples the discourse at
random. Within-partition and standardised series are the most that can be said,
and only with the reference mix stated beside them.

**Any per-account claim.** The whole pipeline is triage: detection says
accounts act together, adjudication says a reader found it worth attention.
Neither establishes that a named account is inauthentic, and the coded-term
register has documented innocent senses.

**Counts of "coordinated accounts".** The top-500 is a triage budget, not a
finding: it is 500 because that is what a human can look at. The number of
communities depends on a Leiden resolution and seed.

**Corroboration as a structural claim.** The old "corroborated clusters are
0% Kenya-referencing" claim is false and retired.

## What would make prevalence publishable

Only a control arm: a random sample of Kenyan election discourse collected
independently of the target list, against which the targeted corpus can be
compared. That is the "no control arm" gap in OBJECTIVES section 4, and it is a
collection design problem, not an analysis one.

## Standing rule for anything published

State the basis on the same line as the number: the snapshot id, whether it is
standardised and to what reference, whether the filter is the keyword gate or
the classifier, and the date. Kenya-share figures recorded before 2026-09-13
used the keyword gate at recall 0.655 and are not comparable with later ones.

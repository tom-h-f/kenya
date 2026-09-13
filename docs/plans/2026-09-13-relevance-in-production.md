# Putting the relevance classifier into production

Written 2026-09-13. The classifier is trained, measured (precision 0.918 /
recall 1.000 against Tom's labels, where the keyword gate is 0.971 / 0.655),
scored over the corpus once, and adopted by the analysis layer. Three things
are missing before it is actually in production: nothing scores new posts,
nothing outside tac2 can load the weights, and the collector's targeting gate
still runs on the keyword regex.

Order matters here - each step is useless without the one above it.

## 1. Weights somewhere every host can load

The model exists only on mike-pc. Modal and any future scorer need it.

**R2, not the HF Hub**: there is no HF token on tac2 or in `.env`, and Modal
already holds a `kenya-r2` secret, so shipping through R2 adds no credential
anywhere. `models/relevance/<slug>/` beside the other prefixes, a loader in
`kma.relevance`, and the slug already recorded on every score row.

Acceptance: a fresh process with only R2 credentials can load the model and
reproduce the B2 numbers.

## 2. A scoring job that keeps up

`modal_relevance.py`, the shape of `modal_backfill.py`: `db.pending_posts`
against the relevance prefix, score on GPU, write runs, drain until empty. On a
daily Modal schedule.

The enrich worker on tf1 keeps computing its own regex `domain` column. No CPU
inference is added there, and the two enrichments stay independent.

Acceptance: a smoke run scores a bounded slice and writes it; a full pass
leaves `pending_posts` empty; the schedule is registered.

## 3. The collector's gate reads it

`promoted_handles` already joins `hatespeech/` for its `domain` column, so this
is the same shape of query: a `relevance_view` beside `hatespeech_view`, and
`COALESCE(model call, regex domain)`. Keep the existing behaviour that a
cluster with no scored posts passes the gate - failing closed would stop
targeting whenever a scorer falls behind.

**Measured on pi0, not tf1.** pi0 has a 600 MB budget and tf1 has the headroom
to hide a query that kills it. If the join is too heavy there, write a small
per-account rollup (author_id, kenya_share, n_scored - about 331k rows) and
join that instead.

Acceptance: the gate picks the model's call where a post is scored, unit tests
cover the fallback, and the pass is measured on pi0 inside budget.

*Measured 2026-09-13, inside the collector container on pi0 with its own DuckDB
settings (600 MB limit, 2 threads):* reading and deduplicating the whole
relevance prefix - 1,281,210 rows - takes **5 s at 245 MB peak RSS**. The gate
query already scans posts and hatespeech, which is the minutes-long part; this
is the marginal cost, and it is small. The direct join stands and the
per-account rollup is not needed.

Running the FULL gate query there while the collector was mid-pass timed out on
the posts read, which is the known shape of that query on pi0 rather than
anything new.

## 4. Guardrails before it drives collection

- **Mentions.** "@SomeForcePolice [emoji]" scores 0.80 on handles alone. Strip
  mentions before scoring, as `coord2.clean_text` now does for the text trace,
  or require a few real words before trusting a positive. This matters more
  once the score decides what the collector chases.
- **Threshold.** 0.5 is where the evaluation ran, not a tuned value. Choose it
  against a precision target on the human labels and record the trade.

*Both done 2026-09-13.* Mentions are handled by retraining rather than by
stripping at serving time only: the 09-12 model was trained on raw text and
served stripped, which is a mismatch, and the 09-13 model is trained the way it
is served. On Tom's labels that moved precision 0.918 -> 0.997 for one post of
recall (1.000 -> 0.980).

The threshold sweep (`09_threshold.py`, corpus-weighted) says keep 0.5. The
model is bimodal, so every cut from 0.3 to 0.5 gives the same 0.997 / 0.980;
0.7 reaches precision 1.000 but costs recall 0.980 -> 0.702, which is 0.003 of
precision for 0.28 of recall.

## 5. Validation and drift

- A second blind human pass (100 fresh posts, `measure-eval human-sheet`)
  judging the MODEL, not the gate. The first pass validated the labeller; this
  validates what shipped.
- A standing weekly model-versus-gate disagreement rate. A jump is either drift
  or a new campaign vocabulary, and both are worth knowing.
- Retrain quarterly: the names in Kenyan politics will keep changing through
  August 2027, and the training labels are model labels whose blind spots the
  human pass is the only check on.

## 6. Restate the figures

Every Kenya-share figure recorded before 2026-09-13 used the keyword gate at
recall 0.655. Recompute the headline v2 numbers on the model and mark each
published figure with the basis it used.

## Unresolved

- Whether the collector's own `CLUSTER_MIN_KENYA_SHARE` (0.15) is still the
  right cut once the gate stops under-reporting relevance: the model calls 37.7%
  of the corpus Kenyan against the gate's 21.5%, so the same threshold is a
  stricter filter than it was.

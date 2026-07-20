# Findings: Plan A - mine + dual-label 2026 posts (2026-07-19)

> Chronological result log (labelling + reliability). For current status and
> next steps see `STATE.md`.

Goal: 1,000 confirmed hate-or-offensive 2026 Kenyan posts, dual-labelled,
plus a random control stratum for honest prevalence. Brief:
`PLAN-A-HANDOFF.md`. Class mapping 0=neither, 1=offensive, 2=hate.

## A0/A1a: corpus scored (Modal A100, model-d3-s1337)

The shipped Plan D checkpoint on the Modal volume is `model-d3-s1337` (the
median-seed d-dapt-focal run - the handoff's `model-d-dapt-focal` is the
seed-42 variant; D3 protocol says ship the median).

`12_score_corpus.py` scored all **76,197** deduped corpus texts
(`out/dapt_corpus.parquet`) in **358s (213 rows/s)** on an A100, ~$0.21.
Output `out/corpus_scored.parquet`.

Argmax distribution over the 2026 corpus: neither 66,747 / offensive 9,067 /
**hate 383** (0.5%). p_hate mean 0.020, max 0.746; **1,307 rows over
p_hate 0.20** (the Plan D deployment threshold), i.e. 1.7% of the corpus.

## A1b: candidate mining

`12_mine_candidates.py` -> `out/label_batch_001.parquet`, **2,440 rows**
(target ~2,500; near-dup dedupe + a 10-post-per-author cap took 55).

| stratum | n | mean p_hate | mean p_offensive |
|---|---|---|---|
| p_hate_top | 1,192 | 0.352 | 0.397 |
| p_offensive_top | 770 | 0.062 | 0.686 |
| random_control | 283 | 0.014 | 0.278 |
| nli_tail | 150 | 0.057 | 0.419 |
| lexicon | 45 | 0.045 | 0.396 |

Notes on the strata:
- Strata are assigned in priority order lexicon > nli_tail > p_hate_top >
  p_offensive_top > random_control so every lexicon and NLI row survives even
  when the classifier also ranks it highly; `is_lexicon` / `is_nli_tail`
  columns keep the overlap visible.
- **45 lexicon hits, not 52** - the other 7 were lost to the corpus's
  near-dup dedupe or eval-split exclusion upstream in `08_export_corpus.py`.
- **The NLI tail is sampled uniformly, not ranked by p_hate.** Ranking it
  would have selected exactly the rows the classifier already likes, killing
  the stratum's whole purpose (probing what the classifier misses). NLI tail
  size in-pool: **9,214 rows (12% of corpus)** - the documented
  over-triggering, unchanged.
- p_hate_top bottom cut is p_hate 0.211.

## A2/A3: labelling infrastructure

`prompts/label_v1.md` - Davidson-compatible classes + the NCIC ethnic-contempt
definition, the full `kma/incitement.py:LEXICON` glossary with fp_risk, the
"coded term + target + intent" rule, and 10 few-shot examples. Strict-JSONL
contract, verified against both models before any batch.

`13_label_drive.py` - 25-post chunks, two labellers via `agy`
(**Gemini 3.1 Pro (Low)**, **Claude Sonnet 4.6 (Thinking)**; `claude -p`
dropped per Tom), concurrency 4, resumable (re-run skips finished chunks -
verified), 3 attempts then park in `out/labels/<tag>/failed/`.

`14_label_merge.py` - agreement -> accepted, flags OR'd, disagreements to
`adjudication_queue_<tag>.csv` (hate-involved first), Cohen's kappa overall
and per class, per-stratum positive rates, optional blind-check sampling.

## A4: pilot - 200 rows, stratified proportional (prompt v1, 2026-07-19)

**Throughput**: 8 chunks x 2 labellers in **3.6 min wall**. Gemini mean
45s/chunk (123 posts/min at concurrency 4), Claude Sonnet mean 50s/chunk
(98 posts/min). **Zero parse failures, zero parked chunks** across 16 calls.

**Agreement**: 80.0%, **Cohen's kappa 0.637** (hate 0.519, offensive 0.668,
neither 0.651).

| | gemini | claude |
|---|---|---|
| neither | 118 | 114 |
| offensive | 66 | 63 |
| hate | 16 | 23 |

Accepted (both agree): **160 of 200**. Confirmed positives **61**, of which
**11 hate**.

| stratum | n | accepted | positive rate | hate rate |
|---|---|---|---|---|
| p_hate_top | 98 | 76 | **0.368** | 0.145 |
| p_offensive_top | 63 | 50 | **0.640** | 0.000 |
| nli_tail | 12 | 9 | 0.111 | 0.000 |
| lexicon | 4 | 3 | 0.000 | 0.000 |
| random_control | 23 | 22 | **0.000** | 0.000 |

### Gate check

- **Kappa >= 0.5: PASS** (0.637 overall; hate 0.519 is the weakest and sits
  just over the line).
- **Mined-stratum positive rate >= 40%: PASS on the mined strata combined**
  (47.6% over p_hate_top + p_offensive_top) but **p_hate_top alone is 36.8%,
  under the line**. The miner is better at finding offensive than hate.
- **Prevalence**: 0/22 accepted random-control rows were positive. With 22
  rows that bounds true prevalence loosely (95% upper bound ~13%), but it
  confirms the handoff's core premise - random sampling would not have got
  anywhere near 1,000 positives.

### Yield projection for the full 2,440-row batch

Applying pilot per-stratum accept and positive rates:

| stratum | full n | accept | pos rate | projected positives |
|---|---|---|---|---|
| p_hate_top | 1,192 | 0.78 | 0.37 | 341 |
| p_offensive_top | 770 | 0.79 | 0.64 | 391 |
| nli_tail | 150 | 0.75 | 0.11 | 12 |
| lexicon | 45 | 0.75 | 0.00 | 0 |
| random_control | 283 | 0.96 | 0.00 | 0 |
| **total** | **2,440** | | | **~744** |

**Short of the 1,000 target by ~250**, and hate specifically projects to only
~140 confirmed rows. Adjudication would recover some (many disagreements are
positive-vs-neither), but the projected **adjudication queue is ~488 rows** -
a large human load for Tom.

### The disagreement pattern (this is the actionable finding)

40 pilot disagreements, and they are not noise - they cluster on one axis:

| gemini | claude | n |
|---|---|---|
| neither | hate | 11 |
| offensive | neither | 15 |
| neither | offensive | 8 |
| hate | offensive | 5 |
| offensive | hate | 1 |

Every `neither`->`hate` case is **collective ethnic denigration without a
violence or expulsion call**, e.g. *"Kalenjin presidency is always a curse to
the nation NEVER AGAIN"* and *"this Kalejins don't know what they are doing
with government"*. Claude reads these as NCIC-sense ethnic contempt; Gemini
reads them as political criticism.

**Prompt v1 does not adjudicate this case.** It says naming an ethnic
community is not hate and that political criticism is not hate, but never
says where collective ethnic contempt without a violence call falls. That
single ambiguity plausibly accounts for most of the hate-class kappa gap
(0.519) and a good share of the 488-row projected queue.

## Prompt v2: the NCIC standard as the governing rule (2026-07-19)

Tom's call: **use NCIC rules as our rules.** `prompts/label_v2.md` makes
NCIC Act s.13 the governing test for the hate/neither boundary and states the
consequence explicitly - **a violence call is not required**; expression that
stirs up ethnic contempt, ridicule or hatred is hate on its own terms. It
enumerates the qualifying cases (collective negative attribution, declaring a
community unfit to lead or belong, collective blame, ridicule of a community)
against the exclusions (criticism of a named politician, bloc voting
analysis, condemning hate), and states the line once: *criticising a leader
who happens to belong to a community is not hate; transferring that criticism
onto the community itself is.* Four new few-shots drawn from the actual v1
disagreement cases.

Same 200 rows, same seed, relabelled - directly comparable:

| metric | v1 | v2 |
|---|---|---|
| agreement rate | 0.800 | 0.795 |
| kappa overall | 0.637 | **0.665** |
| **kappa hate** | 0.519 | **0.643** |
| kappa offensive | 0.668 | 0.720 |
| kappa neither | 0.651 | 0.630 |
| accepted | 160 | 159 |
| confirmed positives | 61 | **75** |
| **confirmed hate** | 11 | **25** |
| adjudication queue | 40 | 41 |

Per-stratum positive rate (hate rate in brackets):

| stratum | v1 | v2 |
|---|---|---|
| p_hate_top | 0.368 (0.145) | **0.586 (0.357)** |
| p_offensive_top | 0.640 (0.000) | 0.582 (0.000) |
| nli_tail | 0.111 (0.000) | 0.222 (0.000) |
| lexicon | 0.000 | 0.000 |
| **random_control** | **0.000** | **0.000** |

Reading:

- **Hate-class kappa 0.519 -> 0.643 and confirmed hate 11 -> 25.** The v1 gap
  really was the unstated collective-contempt case, not labeller noise.
- **p_hate_top now clears the 40% gate at 58.6%** (it was 36.8%, the one gate
  that failed under v1), and its hate rate more than doubled to 0.357.
- **Random control stayed at zero positives.** The broader rule did not start
  firing on benign posts - it moved the boundary where it was contested, not
  everywhere. That is the check that matters most for a widened definition.
- The queue did not shrink (41 vs 40). Disagreement moved rather than
  resolved: it is now mostly hate-vs-offensive on ethnically-tinged personal
  abuse, instead of hate-vs-neither on collective contempt. That is a
  narrower, more genuinely arguable class of case - fine for adjudication.
- One transient failure (`claude/chunk_005`, empty response) recovered on
  retry. Retry logic works; throughput 57-123 posts/min per labeller.

### Yield projection for the full 2,440-row batch (v2 rates)

| stratum | n | accept | pos rate | proj positives | proj hate |
|---|---|---|---|---|---|
| p_hate_top | 1,192 | 0.71 | 0.59 | 499 | 304 |
| p_offensive_top | 770 | 0.87 | 0.58 | 391 | 0 |
| nli_tail | 150 | 0.75 | 0.22 | 25 | 0 |
| lexicon | 45 | 0.75 | 0.00 | 0 | 0 |
| random_control | 283 | 0.96 | 0.00 | 0 | 0 |
| **total** | **2,440** | | | **~915** | **~304** |

Up from ~744 under v1, and hate from ~140 to **~304**. Still ~85 short of
1,000 on agreed rows alone, but the projected 500-row adjudication queue
(256 hate-involved) will contribute more once Tom rules on them - **1,000 is
reachable from this single batch.**

## LLM adjudication instead of human (2026-07-19)

Tom dropped human adjudication. Replacement: `15_adjudicate.py` sends disputed
rows to a **third model (Claude Opus 4.6) blind** - it never sees the other
two verdicts or rationales, because anchoring it on their arguments would make
it a referee of rhetoric rather than an independent third vote. Majority of
three wins; if all three differ, median severity wins.

Tested on the pilot's 41 disputes: 35 resolved by majority, 6 by median
severity. **Opus sided with Gemini 24 times vs Sonnet 11** - the same-family
tilt I was worried about did not appear, and it tilted the other way. Pilot
then reads 200/200 labelled, 103 positives, 35 hate.

That result stands, but the adjudicator was not used on the full run - see
below.

## The full run: quota wall, then single-labeller (2026-07-19/20)

`13_label_drive.py --tag full` on all 2,440 rows:

- **Gemini 3.1 Pro: all 98 chunks, clean.**
- **Claude Sonnet 4.6: 9 of 98, then agy quota exhausted** ("Individual quota
  reached, resets in 3h58m"). 89 chunks parked. The quota is shared across all
  non-Gemini models - GPT-OSS 120B was blocked too - so Opus adjudication was
  equally unavailable.

Options put to Tom: wait ~4h and resume (the driver is idempotent, ~20 min of
work remaining), switch the second labeller to Gemini 3.5 Flash (one family,
correlated blind spots), or ship Gemini-only. **Tom chose Gemini-only.**

`14_label_merge.py` gained a single-labeller mode that refuses to fake the
missing metrics: agreement and kappa are written as `null`, no adjudication
queue is produced, and every row carries `label_source=single_labeller_gemini`.

### Result: `out/labels_2026_full_final.parquet`, 2,440 rows

| | count |
|---|---|
| neither | 1,344 |
| offensive | 782 |
| **hate** | **314** |
| **confirmed positives** | **1,096** |

**The 1,000-positive target is met** (1,096), with 314 hate.

Per-stratum positive rate (hate rate in brackets):

| stratum | n | positive | hate |
|---|---|---|---|
| p_offensive_top | 770 | 0.660 | 0.012 |
| p_hate_top | 1,192 | 0.445 | 0.242 |
| lexicon | 45 | 0.289 | 0.111 |
| nli_tail | 150 | 0.193 | 0.053 |
| **random_control** | **283** | **0.057** | **0.014** |

**Corpus prevalence, measured (the random control's whole purpose):
~5.7% offensive-or-hate, ~1.4% hate.** The mined strata run 8-12x that rate,
which is the miner working as intended - and confirms the handoff's arithmetic
that random sampling for 1,000 positives would have needed ~18,000 rows
labelled.

Note the single-labeller rates run below the pilot's dual-labeller accepted
rates (p_hate_top 0.445 vs 0.586). Those are not comparable: the pilot figure
was over rows where two models agreed, which selects for easy cases.

### Splits - `out/gold_2026_ids.json`

| split | n | neither | offensive | hate |
|---|---|---|---|---|
| train | 1,962 | 924 | 741 | 297 |
| gold (random_control) | 283 | 267 | 12 | 4 |
| challenge (lexicon + nli_tail) | 195 | 153 | 29 | 13 |

Gold is the random control only - the one stratum no model selected, so the
only split where a score says something about the corpus rather than about the
miner. Challenge (rule-based miners, not classifier-selected) is non-circular
but not a random sample: use it for coded-term recall, never for prevalence.

## Status: batch complete, with a standing caveat

**Read this before training on these labels or quoting any number from them.**

1. **No reliability estimate exists for this dataset.** One labeller, no
   second opinion, no adjudication, no human check. The pilot's kappa 0.665
   measured a two-labeller design that was not used for the full run and does
   not validate these labels. The blind-check gate (>= 85% overall / >= 70%
   hate) was never run - `14_label_merge.py --blind-check 100` still writes
   the CSV and key whenever someone wants to close that loop.
2. **The gold set inherits the same weakness.** A model scored on it is being
   measured against Gemini 3.1 Pro's judgement. It cannot detect anything that
   model is systematically wrong about - and the pilot showed the labellers
   disagreed on 20% of rows, so there is real judgement variance in here that
   is now invisible.
3. **Cheapest fix if this matters later**: the 89 parked Sonnet chunks are
   still queued on disk. Re-running `13_label_drive.py --tag full` after a
   quota reset picks up exactly where it stopped (~20 min), and
   `14_label_merge.py --tag full` then yields the full dual-labeller design
   with kappa, retroactively - no relabelling wasted.

Next round (out of scope here): merge these 2,440 rows with the 48k
2013-era set and continue fine-tuning d-dapt-focal, holding out gold. The
`flags` column is already populated for the multi-task head.

## Dual-labeller resume + the asymmetry finding (2026-07-20)

agy quota cleared; `13_label_drive.py --tag full` resumed and took Sonnet
from 9 to 38 of 98 chunks before the quota re-exhausted (29 ok, 60 parked,
11.2 min, 65 posts/min, resets in ~4h47m). **950 of 2,440 rows now carry two
independent labels** - more than enough for a reliability estimate, so the
merge was run on that subset rather than waiting ~10h for full coverage.

| metric | value |
|---|---|
| rows labelled by both | 950 |
| agreement | 0.7526 |
| **Cohen's kappa** | **0.590** (hate 0.578, offensive 0.636, neither 0.578) |
| disagreements | 235 (178 involve hate) |

### The finding: the two labellers disagree *directionally* on hate

| | Sonnet hate | Sonnet neither | Sonnet offensive |
|---|---|---|---|
| **Gemini hate** | 217 | 25 | 7 |
| **Gemini neither** | **120** | 397 | 26 |
| **Gemini offensive** | 26 | 31 | 101 |

- Gemini not-hate / Sonnet hate: **146**
- Gemini hate / Sonnet not-hate: **32**
- **Asymmetry: 4.56x**

Per-stratum hate rate, Gemini vs Sonnet: p_hate_top **0.313 vs 0.460**,
lexicon **0.111 vs 0.178**, nli_tail 0.053 vs 0.053.

**Gemini is systematically more conservative about hate than Sonnet**, and
120 of the disagreements are a full two steps apart (neither vs hate), not
boundary quibbling.

This does not prove Gemini is wrong - Sonnet may over-call. It does prove
the single-labeller decision was **materially consequential, not neutral**:
the full batch was labelled by the more conservative of the two, and round 2
trained on it and regressed on coded incitement (see `findings.md`). The
mechanism proposed there now has direct supporting evidence.

### Blind check rebuilt to arbitrate this

`18_blind_check.py` (new) detects a dual-labeller file and switches from
coded-weighted sampling to **disagreement arbitration**. Sheet:
`out/blind_check_coded.csv`, 120 rows, columns post_id/text/human_label only,
shuffled, verified to leak nothing.

| pool | n | question it answers |
|---|---|---|
| split_gem_soft | 30 | Gemini not-hate vs Sonnet hate - **the axis** |
| split_gem_hard | 12 | the reverse, so we do not assume a winner |
| agree_hate | 18 | when both say hate, are they right? |
| agree_coded | 24 | agreed lexicon/NLI rows |
| random_agreed | 36 | calibration - catches over-flagging |

`score` reports which labeller the human sided with, and specifically what
share of the 30 split_gem_soft rows the human calls hate. **>50% means the
training labels are too conservative -> prompt v3 + relabel.** Scorer verified
end-to-end with synthetic answers.

## Full dual coverage via the Cursor CLI (2026-07-20)

agy's non-Gemini quota caps at ~30 chunks/window, so the second opinion was
never going to finish there. `13_label_drive.py` now dispatches to either
`agy` or the Cursor `agent` CLI (`LABELLERS` maps name -> (cli, model)),
which draw on **separate quotas** - the durable fix.

Ran `cursor-sonnet-4.5` over **all 98 chunks**, not just the 60 agy missed:
mixing Sonnet 4.6 and 4.5 into one labeller bucket would hide which model
produced which label and make kappa uninterpretable. 98 ok, 0 parked,
27.2 min, 90 posts/min. The 38 agy Sonnet-4.6 chunks survive as a third
opinion where they exist.

Cursor CLI notes: needs `--trust` headlessly, and emits a prose preamble plus
```json fences regardless of instruction. `parse_response` now skips non-`{`
lines - structural validation (every post_id back exactly once, enums) is
unchanged, so this loosens transport, not standards.

### Full-batch reliability: 2,440 rows, both labellers

| metric | 950-row subset (Sonnet 4.6) | **full 2,440 (Sonnet 4.5)** |
|---|---|---|
| agreement | 0.7526 | **0.7992** |
| kappa overall | 0.590 | **0.674** |
| kappa hate | 0.578 | **0.666** |
| disagreements | 235 | 490 (217 hate-involved) |

Kappa 0.674 is substantial agreement and close to the pilot's 0.665 - the
lower subset figure was a sampling artefact (that subset was p_hate_top-heavy).

### The asymmetry replicates exactly: 4.56x

| | Cursor hate | Cursor neither | Cursor offensive |
|---|---|---|---|
| **Gemini hate** | 275 | 20 | 19 |
| **Gemini neither** | **120** | 994 | 230 |
| **Gemini offensive** | 58 | 43 | 681 |

- Gemini not-hate / Cursor hate: **178**
- Gemini hate / Cursor not-hate: **39**
- **Asymmetry 4.56x** - identical to the independent 950-row estimate against
  a *different* model version (Sonnet 4.6). Two Claude models, two CLIs, same
  answer. This is now a replicated finding, not an artefact.

Hate rate by stratum, Gemini vs Cursor: p_hate_top **0.242 vs 0.339**,
p_offensive_top 0.012 vs 0.034, random_control 0.014 vs 0.025, lexicon
0.111 vs 0.133.

Whole-batch label counts: Gemini 314 hate / 782 offensive / 1,344 neither;
Cursor **453 hate** / 930 offensive / 1,057 neither.

**Conclusion unchanged and now firmer: the training labels came from the
markedly more conservative of two models.** Whether that conservatism is
correct is a question only a human can settle - which is what the blind
check is for.

### Blind check rebuilt on full pools

`out/blind_check_coded.csv`, 120 rows drawn from complete pools: 30
split_gem_soft (of 178 available), 12 split_gem_hard (of 39), 18 agree_hate,
24 agree_coded, 36 random_agreed. Verified to leak nothing; scorer verified
end-to-end. `18_blind_check.py` now discovers labeller columns dynamically
(excluding `label_source`) so it works with either second labeller.

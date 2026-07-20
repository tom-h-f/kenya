# Findings: hate-speech fine-tune (2026-07-18)

> Chronological result log (Plan D + round 2). For current status and next
> steps see `STATE.md`.

## Results

| model | test macro-F1 | hate F1 | hate P / R |
|---|---|---|---|
| TF-IDF + logreg baseline | 0.509 | 0.357 | 0.26 / 0.57 |
| twitter-xlm-roberta-base, 4 epochs | **0.543** | **0.421** | 0.34 / 0.54 |

Run was killed at 96% then resumed from the epoch-3 checkpoint (`--resume`).
Val macro-F1 by epoch: 0.468 -> 0.541 -> 0.555 -> 0.560; epoch 4 selected as
best. Val loss rose throughout (0.82 -> 1.05) while F1 climbed - expected with
weighted CE on noisy labels; F1 is the selection metric, not loss.

Threshold sweep (`out/03_hate_thresholds.csv`): p_hate >= 0.95 gives 0.67
precision at 0.23 recall (107 flags on 4,805 test rows); p_hate >= 0.5 gives
0.38 precision at 0.51 recall. Labels themselves are noisy (~3 annotators,
offensive/hate boundary blurry), so ceiling is limited - Davidson-style
datasets typically top out around 0.6-0.7 macro-F1 even with big models.

## B3/C3 ablations (2026-07-19, Colab T4, batch 64, v2 splits, 1 seed)

| variant | test macro-F1 | unan macro-F1 | unan hate P/R |
|---|---|---|---|
| b3-all (reference) | 0.5545 | 0.5759 | 0.39 / 0.73 |
| **b3-agree60** (drop tie rows) | **0.5626** | **0.5857** | 0.37 / **0.79** |
| b3-weighted (agreement weights) | 0.5569 | 0.5639 | 0.37 / 0.69 |
| c3-afrihate (+18k AfriHate swa) | 0.5567 | 0.5834 | 0.37 / 0.72 |
| **c4-combo (agree60 + afrihate)** | 0.5580 | **0.5972** | 0.36 / **0.81** |

Verdict: **c4-combo is the canonical recipe** - clear winner on the unanimous
(trusted-label) test, +2.1pt over reference, hate recall 0.81. It trails
agree60 slightly on the FULL test, which still contains the noisy 2-of-3
labels: AfriHate pulls the model away from this dataset's annotation quirks,
so it loses on noise and wins on clean labels - the desirable direction.
Other lessons: dropping tie rows beats keeping or down-weighting them;
agreement-weighting actively hurts (0.5639, worst). Margins ~1pt at 1 seed
individually, but ordering consistent across both tests. Multi-seed
confirmation in the Plan D matrix. Round-1 v1-split numbers not comparable.
Unan triage curve (combo): p_hate>=0.75 -> P 0.50 / R 0.68; >=0.90 -> 0.54 /
0.49.

## Domain transfer to 2026 corpus (spot-check, 14 known-flagged posts)

`out/04_spotcheck_flagged.csv` - scored the manipulation-sweep flagged set:

- 9/14 scored **offensive**, 5 neither, 0 hate. p_hate ~0.01 even on
  "tunaua nyoka na mayai yake" (coded dehumanisation + violence call).
- Benign-looking rows in the set (press-freedom post, wantam banter) correctly
  scored neither - model is not spraying false positives.

Interpretation: training data is 2013-era, mostly English/explicit tribe-name
hate. The 2026 coded Swahili/Sheng incitement register (nyoka, goons, mafeelings
metaphors) is out of distribution -> lands in "offensive", not "hate".

## Plan D (afro-xlmr-large + DAPT) - in progress

### D0: DAPT corpus export (2026-07-19, tac2)

`08_export_corpus.py` from R2 latest X posts: 118,417 pulled -> 117,528
after clean/min-len -> 81,313 after exact dedupe -> 76,197 after MinHash
near-dedupe (heavy retweet duplication). 4 rows dropped for eval-split
overlap. `out/dapt_corpus.parquet`, copied to Drive.

### Tooling landed (2026-07-19, all smoke-tested on tac2)

- `02_train.py` extended: `--grad-accum/--label-smoothing/--focal-gamma/
  --llrd/--seed/--warmup-ratio/--patience/--grad-checkpoint`. Defaults
  unchanged (b3/c4 invocations bit-identical). Focal replaces class weights;
  LLRD via create_optimizer override (28 param groups on base-size smoke).
- `09_dapt.py`: MLM continue-pretraining, stock-baseline gate (gain >= 0.10
  in holdout loss), 100-step timing probe with >8h abort, per-500-step Drive
  checkpoints, auto-resume (verified). `prediction_loss_only=True` required -
  MLM eval OOMs otherwise (250k-vocab logit accumulation).
- `10_push_hf.py`: private HF push, `--dry-run` verified.
- `run_d_batch.sh` stage-1 ladder: d-base / d-dapt / d-dapt-ls / d-dapt-focal
  (batch 8 x accum 8, lr 1e-5, 5 epochs, warmup 0.06). Deletes checkpoints
  after eval (Drive quota). Stage-2 (llrd on best) + stage-3 (seeds 1337/2027)
  scripts get authored at decision time.
- Notebooks: `colab_d_dapt.ipynb`, `colab_d_finetune.ipynb`. HF_TOKEN via
  Colab secret. Primary execution path is now the `colab` CLI from tac2
  (T4 session + Drive mount + detached nohup runs), notebooks are backup.
- transformers v5 gotcha (bit on T4): `from_pretrained` dtype="auto" loads
  afro-xlmr-large's fp16 weights -> "Attempting to unscale FP16 gradients"
  under fp16 AMP. Fixed: `dtype=torch.float32` in 02_train.py + 09_dapt.py.

### Execution pivot: Colab free tier -> Modal (2026-07-19)

Colab free T4 killed the full DAPT run twice (preemption at ~step 1200, then
daily GPU quota exhausted at ~step 1300; checkpoint-resume worked both
times). Pivoted to Modal A100 via `modal_train.py` (volume-backed out/,
detached spawns). A100 smoke: 20-step sample gain +0.33, 3x T4 throughput.
Full DAPT rerun fresh on A100 (batch 32 x accum 2 = same eff 64, no
grad-checkpoint). Ladder retuned to batch 32 x accum 2 accordingly (no
d-variant had started - comparability intact).

### D1: DAPT result (2026-07-19, Modal A100, seed 42)

74,673 train / 1,524 holdout tweets, 2 epochs, eff batch 64, 22 min.
Holdout MLM loss **2.7721 -> 1.6581** (perplexity 16.0 -> 5.2), gain
**1.114** vs a gate of 0.10. Domain adaptation clearly worked.

### D2: stage-1 ladder (2026-07-19, Modal A100, seed 42, canonical recipe
agree60 + AfriHate, 5 epochs, eff batch 64, lr 1e-5)

All numbers on `test_unanimous` (2,264 rows; class support neither 2,054 /
offensive 135 / **hate 75**) unless labelled full. Hate P/R/F1 at argmax.

| variant | unan macro-F1 | full macro-F1 | hate P/R/F1 | offensive F1 | neither F1 | acc |
|---|---|---|---|---|---|---|
| c4-combo (prior best) | 0.5972 | 0.5580 | 0.36/0.81/- | - | - | - |
| d-base (stock large) | 0.6091 | 0.5688 | 0.397/0.827/0.537 | 0.370 | 0.920 | 0.849 |
| d-dapt | 0.6315 | 0.5832 | 0.466/0.827/0.596 | 0.375 | 0.923 | 0.855 |
| d-dapt-ls | 0.6373 | 0.5741 | 0.416/0.853/0.559 | 0.421 | 0.932 | 0.868 |
| **d-dapt-focal** | **0.7024** | **0.5897** | 0.586/0.773/0.667 | 0.478 | 0.963 | 0.923 |

Three findings, in order of size:

1. **+6.7pt unan over d-dapt - but see the control below: the cause was
   dropping class weights, NOT the focal term.** The `--focal-gamma` flag
   changes two things at once (adds focal, removes class weighting), so
   this row was confounded as originally run.
2. **DAPT works, worth +2.2pt** (d-dapt 0.6315 vs d-base 0.6091, hate F1
   0.537 -> 0.596). Smaller than the MLM gain suggested, but real and in
   the right direction. Stock afro-xlmr-large *alone* (d-base 0.6091)
   already beats c4-combo, so most of Plan D's headline gain is model
   size/family; DAPT and focal add on top.
3. **Label smoothing 0.05 is roughly neutral** (+0.6pt unan, -0.9pt full).
   Not worth carrying.

**Promotion gates vs c4-combo** (>= 0.62 unan, hate R >= 0.80 at P >= 0.36,
full >= 0.55): d-dapt, d-dapt-ls and d-dapt-focal all clear the macro-F1
and full-test gates. On hate recall, d-dapt (0.827) and d-dapt-ls (0.853)
clear it at argmax; d-dapt-focal reads 0.773 at argmax - **but that is an
operating-point artefact, not a ceiling**. Sub-argmax threshold sweep
(`11_hate_sweep.py`, out/eval-d-dapt-focal-unan_full_sweep.csv):

| threshold | flagged | hate P | hate R |
|---|---|---|---|
| argmax | 99 | 0.586 | 0.773 |
| 0.26 | 134 | 0.455 | 0.813 |
| 0.20 | 158 | 0.405 | 0.853 |
| 0.16 | 176 | 0.381 | 0.893 |

At **equal recall to c4-combo (0.81), focal's precision is 0.455 vs
c4-combo's 0.36** - a 9.5pt precision gain at the same catch rate, i.e.
~25% fewer false flags per real hit. Gates met; the deployment choice is
threshold 0.26 for recall-first triage, argmax for precision-first review.

**Caveat that gates the whole table: hate support is 75 rows.** A 4-row
swing moves recall 5pt, so single-seed differences under ~3pt (e.g. ls vs
dapt) are noise. Focal's +6.7pt is well outside that, DAPT's +2.2pt is not.
D3 multi-seed is required before any of this is quotable.

### D2 stage 2: LLRD on the winner (2026-07-19, Modal A100, seed 42)

| variant | unan macro-F1 | full macro-F1 | hate P/R/F1 | offensive F1 | acc |
|---|---|---|---|---|---|
| d-dapt-focal | **0.7024** | **0.5897** | 0.586/0.773/0.667 | 0.478 | 0.923 |
| d-dapt-focal-llrd | 0.6498 | 0.5626 | 0.517/0.600/0.556 | 0.440 | 0.903 |

Layer-wise LR decay 0.9 **hurt, -5.3pt unan**, and cost 17pt of hate recall.
Well outside the ~3pt noise band, so this is a real effect, not a seed
wobble. Reading: at lr 1e-5 the lower layers were not overfitting in the
first place, so decaying them to ~1e-6 just starved the model of the
adaptation it needed - the DAPT weights are already domain-tuned and want
to keep moving. **LLRD is not in the shipped recipe.**

Winner: **d-dapt-focal** = DAPT model + canonical recipe + focal gamma 2.0,
no class weights, no label smoothing, no LLRD.

### D3: multi-seed confirmation (2026-07-19, Modal A100, d-dapt-focal x 3 seeds)

| seed | unan macro-F1 | full macro-F1 | hate F1 | hate P | hate R | offensive F1 |
|---|---|---|---|---|---|---|
| 42 | 0.7024 | 0.5897 | 0.6667 | 0.586 | 0.773 | 0.478 |
| 1337 (median, shipped) | 0.6976 | 0.5951 | 0.6460 | 0.605 | 0.693 | 0.483 |
| 2027 | 0.6638 | 0.5902 | 0.6000 | 0.537 | 0.680 | 0.443 |
| **mean ± sd** | **0.6879 ± 0.0210** | **0.5917 ± 0.0030** | 0.6375 ± 0.0341 | - | 0.7156 ± 0.0505 | - |

**Seed 42 was the lucky one.** The honest headline is **0.688 ± 0.021 unan**,
not 0.7024 - still +9.1pt over c4-combo (0.5972) and clearing the 0.62 gate
even at mean minus one sd (0.667). Full-test is rock steady (± 0.003);
all the variance sits in the two rare classes, as the 75-row hate support
predicted.

**Gate verdict, stated honestly:**
- unan macro-F1 >= 0.62: **PASS** (0.688 ± 0.021)
- full test >= 0.55: **PASS** (0.592 ± 0.003)
- hate recall >= 0.80 at precision >= 0.36: **FAILS at argmax**
  (mean recall 0.716 ± 0.051), **PASSES on a tuned threshold**. Median seed
  (1337) sweep: threshold 0.20 gives R 0.840 / P 0.384; threshold 0.26 gives
  R 0.773 / P 0.464. So the gate is met at thr 0.20 with precision (0.384)
  just above c4-combo's 0.36 - a much thinner margin than seed 42 suggested
  (that run gave P 0.455 at R 0.813).

Net: the model is a solid, reproducible win on overall quality, and roughly
a wash with c4-combo specifically on the hate-recall-at-fixed-precision
trade. Anyone quoting "0.70" or "+9.5pt precision at equal recall" is
quoting the best seed - do not.

Shipped checkpoint: **seed 1337** (median by unan macro-F1, per protocol -
picking the best seed would be selecting for test-set luck).

### D-control: was it focal, or was it dropping class weights? (2026-07-19)

`--focal-gamma` also disables class weighting, so `d-dapt -> d-dapt-focal`
changed two things at once. `d-dapt-nowt` isolates them: plain unweighted CE,
no focal, everything else identical.

| loss | unan | full | hate P/R/F1 | offensive F1 | neither R | acc |
|---|---|---|---|---|---|---|
| weighted CE (`d-dapt`) | 0.6315 | 0.5832 | 0.466/0.827/0.596 | 0.375 | 0.872 | 0.855 |
| **plain CE (`d-dapt-nowt`)** | **0.6981** | 0.5904 | 0.607/0.720/0.659 | 0.473 | 0.957 | 0.921 |
| focal, no weights (`d-dapt-focal`) | 0.7024 | 0.5897 | 0.586/0.773/0.667 | 0.478 | 0.958 | 0.923 |

**Verdict: the entire gain came from removing class weights.** Plain CE
0.6981 vs focal 0.7024 - a 0.4pt difference, five times smaller than the
seed sd (2.1pt). The focal term contributes nothing measurable here.

The mechanism is visible in neither-recall: 0.872 -> 0.957. On 2,054 benign
unanimous-test posts, inverse-frequency weighting produced **263 false
flags; plain CE produces 88**. The 11x penalty on the majority class was
buying hate recall (0.827 vs 0.720) at the cost of ~175 extra false
positives and 6.7pt of macro-F1 - a bad trade even for recall-first triage,
since the same recall is reachable by lowering the hate threshold, which
costs far less precision elsewhere.

**Restated finding: for this dataset, inverse-frequency class weighting is
actively harmful; drop it and tune the decision threshold instead.** That is
a more useful and more transferable result than the focal story - it also
retro-explains why c4-combo (class-weighted) sat at hate P 0.36.

Practical note: the shipped model keeps focal only because it is the variant
with 3-seed validation. Plain CE is statistically tied and simpler; if
anything is retrained from here, use plain CE + threshold tuning and skip
focal.

### D5: packaged (2026-07-19)

- Classifier (seed 1337, median): **`tom-h-f/kenya-hatespeech-afroxlmr`**
  (private, HF). Model card carries the limitations - especially "not a
  coded-incitement detector" and "do not re-add class weights".
- DAPT encoder: **`tom-h-f/kenya-dapt-afroxlmr`** (private, HF) - the
  starting point for any future round.
- Copy on Drive: `~/Drive/Colab/hatespeech-finetune/out/model-d/`.
- Licence CC BY-NC-SA (inherited from AfriHate).
- NOT wired into the kma pipeline - tf1 is a 3.8GB-RAM CPU box and this is
  2.26GB in fp32; quantisation decision still pending.

### D4: 2026 transfer spot-check - THE headline (2026-07-19, seed 1337)

Same 14 known-flagged manipulation-sweep posts as round 1
(`out/04_spotcheck_d.csv` vs round-1 `out/04_spotcheck_flagged.csv`).

| | round-1 model | Plan D (d3-s1337) |
|---|---|---|
| argmax labels | 8 offensive / 6 neither / **0 hate** | 6 offensive / 8 neither / **0 hate** |
| p_hate mean | 0.0122 | **0.0609** (5x) |
| p_hate max | 0.1080 | **0.4048** |
| posts over p_hate 0.20 | 0 | 1 |
| posts over p_hate 0.10 | 1 | 3 |

Per-post movement (p_hate, round-1 -> Plan D):

| post (abridged) | r1 | D | x |
|---|---|---|---|
| "Nyoka ni gachagua na wamunyoro **retards**" | 0.0058 | **0.4048** | 70x |
| "...Kenya's number tribalist..." | 0.0061 | 0.1190 | 20x |
| **"tunaua nyoka na mayai yake"** | 0.0077 | 0.0673 | 9x |
| "kutoa wantam, next ni kutoa **madoadoa**" | 0.0020 | 0.0190 | 10x |
| "nyoka na mayai yake lazima iende" | 0.0055 | 0.0156 | 3x |

**Verdict: real movement in the right direction, not a solve.** Every post
moved up in p_hate (11 of 14 materially; the 3 flat ones are benign). The
most explicit coded post - snake metaphor plus an ethnic target plus a slur -
went 70x and now sits at 0.40, i.e. it would flag at the deployment
threshold of 0.20. But the operationally important ones still do not:
*"tunaua nyoka na mayai yake"* ("we kill the snake and its eggs") reaches
only 0.067, and *madoadoa* - the 2007-08 ethnic-cleansing term - only 0.019.
Both remain far under any usable threshold.

Note also two posts moved offensive -> neither. That is the class-weight
removal cutting both ways: better precision on in-distribution data, more
conservatism on out-of-distribution coded speech. For recall-first triage
this is a genuine tension; threshold is the only lever we have until 2026
labels exist.

**Conclusion: Plan D delivers a materially better classifier (+9.1pt unan,
reproducible across seeds) but NOT yet a coded-incitement detector.** The
gap is training data, not architecture: the model has now READ the 2026
corpus (DAPT, perplexity 16 -> 5.2) but has never been SHOWN a labelled
2026 coded-incitement example. That is exactly Plan A - see
`PLAN-A-HANDOFF.md`. The joint lexicon+NLI rule in `kma/incitement.py`
remains the primary coded-incitement signal; this model is the
surface-offensiveness tier and a corpus-wide ranking signal.

## Conclusion

Model is a useful **third triage signal** (surface-offensiveness tier + cheap
corpus-wide scoring) but does NOT replace the lexicon+NLI joint rule for coded
incitement. To make it the primary detector: label a few hundred 2026 corpus
posts (NLI-tail + flagged + random benign) and continue fine-tuning - the
infrastructure here supports that directly (`00_prep.py` splits, `02_train.py
--model out/model`).

## Round 2: training on Plan A's 2026 labels (2026-07-20, Modal A100)

1,662 labelled 2026 rows added. Model selection on `val2026` (300 rows).
Three test sets: `gold` (283 random-control), `challenge` (195 lexicon/NLI),
`test_unanimous` (2,264, 2013-era human labels).

| test set | variant | macro-F1 | hate R | hate P | off F1 | neither R |
|---|---|---|---|---|---|---|
| gold | r2-baseline (shipped d3-s1337) | 0.4278 | 0.000 | 0.000 | 0.333 | 0.925 |
| gold | **r2-mixed** | **0.5850** | 0.250 | 1.000 | 0.381 | 0.985 |
| gold | r2-continue | 0.4346 | 0.000 | 0.000 | 0.333 | 0.989 |
| challenge | r2-baseline | 0.4526 | 0.077 | 0.333 | 0.429 | 0.725 |
| challenge | **r2-mixed** | **0.6383** | 0.154 | 1.000 | 0.738 | 0.922 |
| challenge | r2-continue | 0.5722 | 0.154 | 0.500 | 0.582 | 0.935 |
| unanimous | r2-baseline | **0.6976** | 0.693 | 0.605 | 0.483 | 0.963 |
| unanimous | r2-mixed | 0.6712 | 0.613 | 0.568 | 0.464 | 0.959 |
| unanimous | r2-continue | 0.6203 | 0.600 | 0.634 | 0.279 | 0.989 |

`r2-continue` (two-stage, 2026-only, lr 5e-6) **catastrophically forgot**:
offensive F1 on the 2013 set collapsed 0.483 -> 0.279. The two-stage bet is
dead; mixing with oversampling is the right shape.

`r2-mixed` looks like a triumph: +15.7pt gold, +18.6pt challenge, and only
-2.6pt on the 2013 set (inside the 2.1pt seed sd). **It is not. See below.**

### The circularity check - and it fails

Gold and challenge are labelled by Gemini 3.1 Pro. `r2-mixed` was trained on
Gemini 3.1 Pro labels from the same batch and prompt. So those gains measure
**agreement with the labeller**, which training on the labeller's output is
guaranteed to improve. The only human-grounded check available is the
14 known-flagged manipulation-sweep posts (`04_spotcheck_*.csv`):

| post (abridged) | round-1 | Plan D | **r2-mixed** |
|---|---|---|---|
| "Nyoka ni gachagua na wamunyoro **retards**" | 0.0058 | 0.4048 | **0.3264** |
| **"tunaua nyoka na mayai yake"** | 0.0077 | 0.0673 | **0.0105** |
| "nyoka na mayai yake lazima iende" | 0.0055 | 0.0156 | **0.0023** |
| "kutoa wantam, next ni kutoa **madoadoa**" | 0.0020 | 0.0190 | **0.0038** |
| p_hate mean over all 14 | 0.0122 | 0.0609 | **0.0289** |
| argmax labels | 8 off / 6 neither | 6 off / 8 neither | **4 off / 10 neither** |

**13 of 14 posts moved DOWN in p_hate. The canonical coded-incitement post -
"we kill the snake and its eggs" - fell 6x, back to round-1 levels.** The
model became more conservative on exactly the register Plan A was meant to
teach it.

### What this means

**Round 2 improved agreement with the labeller, not detection of coded
incitement.** The most probable mechanism: Gemini is conservative about
coded hate - it labelled only 1.4% of the random control as hate, and the
pilot needed prompt v2 to lift hate kappa from 0.519 to 0.643, i.e. the
hate boundary was the labeller's weakest axis all along. Training on those
labels transferred that conservatism into the model. We taught it the
labeller's blind spot.

This is precisely the risk `findings-plan-a.md` flagged: one labeller, no
reliability estimate, and a gold set that "cannot detect anything that model
is systematically wrong about". It is no longer hypothetical.

**Do not ship r2-mixed.** The shipped model stays `d3-s1337`.

### What would settle it

1. **The 100-row blind check** (`14_label_merge.py --tag full
   --blind-check 100`) - now the critical experiment, not hygiene. If Tom
   disagrees with Gemini's `neither`/`offensive` calls on coded rows, the
   labels are the problem and the prompt's hate definition needs widening
   for coded speech (v3), followed by a relabel.
2. **Resume the 89 parked Sonnet chunks** when agy quota resets - gives the
   independent second opinion the full run never got, plus kappa.
3. Consider a **coded-speech-weighted blind check**: sampling uniformly from
   accepted rows puts ~94% benign rows in front of a human. Sampling the
   lexicon/NLI strata harder tests the axis that actually failed.

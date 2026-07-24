# STATE - Kenya hate-speech classifier

Single source of truth for where this investigation stands. Updated
2026-07-25. If this disagrees with any other doc, this wins.

Reading order for someone new: this file, then `README.md` (scripts + how to
run), then `findings.md` (Plan D + round 2 results) and `findings-plan-a.md`
(labelling results). Historical planning docs are in `archive/`.

---

## One-paragraph status

Opus v4 labels (1,500; human-validated) trained under the settled mixed
recipe. Initial Opus ×5 run gained challenge but lost −3.4pt on 2013
unanimous. A seed-1337 **extra-repeat sweep** found **Opus ×3** as the
Pareto point: challenge **0.614** (+7.2pt) and unanimous **0.690** (−0.8pt,
inside ~2.1pt seed noise). Hard-flag oversample raised challenge further
but re-hurt unanimous. Hate-threshold sweeps and coded-14 scoring are
banked. Production remains `d3-s1337` until a **3-seed confirm** of mix ×3
lands. Remaining 940 labels and flag heads stay deferred.

## Current decision

- Replace production `d3-s1337`: **NO** until mix ×3 3-seed confirm.
- Provisional research recipe: **AfriHate ×1 + Opus train ×3** (not ×5).
- Opus ×5: **rejected** as ship candidate (unan regression past seed noise).
- Opus ×3 + hard-flag oversample: **reject for ship** (unan −2.6pt); keep as
  challenge-maximising experiment only.
- Finish remaining 940 Opus labels: **DEFER**.
- Flag-head pilot: **NOT APPROVED**.
- Existing shipped model remains the production candidate.

---

## Round-3 Opus ×5 (measured 2026-07-24)

Recipe: DAPT + 2013 + AfriHate ×1 + Opus ×5, focal-gamma 2.0,
val=`val2026_opus-v4`, seeds 1337/1338/1339. App `ap-C1wCWkA3GD6F6kbhgKaesM`.

| eval set | baseline `d3-s1337` | r3-opus mean ± sd | Δ |
|---|---|---|---|
| `challenge_opus-v4` | 0.541 | 0.610 ± 0.012 | +0.069 |
| `test_unanimous` | 0.698 | 0.664 ± 0.006 | −0.034 |
| old `gold` | 0.428 | 0.414 ± 0.011 | −0.014 |

**Verdict:** do not promote.

## Follow-up: mix / threshold / errors (2026-07-24)

### Mix extra-repeat sweep (seed 1337 only)

App `ap-aovETPE2nGy7XhQRKb9R2v`. Same recipe; Opus repeat ∈ {1,2,3} plus
×3+hardflags×2. ×5 = prior `r3-opus-s1337`.

| variant | challenge | unan | Δ chal | Δ unan |
|---|---|---|---|---|
| baseline | 0.541 | 0.698 | — | — |
| Opus ×1 | 0.579 | 0.680 | +0.038 | −0.018 |
| Opus ×2 | 0.585 | 0.685 | +0.044 | −0.013 |
| **Opus ×3** | **0.614** | **0.690** | **+0.072** | **−0.008** |
| Opus ×5 | 0.597 | 0.670 | +0.056 | −0.027 |
| ×3 + hardflags | 0.630 | 0.672 | +0.089 | −0.026 |

**Pareto:** Opus ×3. Next: 3-seed confirm (`run_r3_mix3_seeds.sh`).

### Hate threshold (full sweep, `11_hate_sweep.py`)

On `val2026_opus-v4` (50 hate / 250):

| model | best-F1 thr | F1 | P | R | R≥0.80 thr | P@R80 |
|---|---|---|---|---|---|---|
| baseline | 0.42 | 0.45 | 0.40 | 0.50 | 0.28 | 0.27 |
| r3-opus-s1337 | **0.24** | **0.69** | 0.56 | 0.90 | **0.28** | **0.58** |

On `test_unanimous`, baseline still has the better hate-F1 operating curve;
Opus models trade some 2013 hate precision for 2026 val recall. Deploy with
an explicit threshold, not argmax, if using an Opus checkpoint.

### Error-driven cleanup

- 27 challenge errors shared by all three ×5 seeds
  (`out/r3_hard_misses_challenge.csv`); 4 hate misses; **none in train**.
- No automatic label flips (labels human-validated).
- Hard-flag train oversample pack n=82 on Modal volume; helped challenge,
  hurt unan when added on top of ×3.

### Coded-14 spotcheck

Opus gold on all 14: 0 hate / 8 offensive / 6 neither. Mean p_hate:
baseline 0.070 → opus-s1337 0.032; argmax hate still 0. Taxonomy treats
most coded menace without a protected-group target as offensive.

Caveats:
- Challenge labels are Opus v4 and human-validated.
- No Opus-labelled gold in the partial set.
- Mix ×3 numbers above are **1-seed** until 3-seed confirm finishes.

## What is banked

| asset | where | number |
|---|---|---|
| Shipped classifier `d3-s1337` | HF `tom-h-f/kenya-hatespeech-afroxlmr` (private); Drive `out/model-d/`; Modal vol | unan macro-F1 **0.688 ± 0.021** (3 seeds), full 0.592 |
| Round-3 Opus ×5 (not shipped) | Modal `model-r3-opus-s{1337,1338,1339}/` | challenge 0.610 ± 0.012; unan 0.664 ± 0.006 |
| Mix sweep seed-1337 | Modal `model-r3-mix-r{1,2,3,3hard}-s1337/` | Pareto = Opus ×3 |
| Threshold + coded-14 | Modal `sweep-*`, `spotcheck_coded14_*`, `r3_threshold_summary.csv` | see above |
| DAPT encoder | HF `tom-h-f/kenya-dapt-afroxlmr`; Modal `dapt-afro-xlmr/` | perplexity 16.0 -> 5.2 |
| Opus v4 partial labels | `out/labels_2026_opus-v4-partial.parquet` | 1,500; neither 676 / offensive 558 / hate 266 |
| Opus v4 splits | `out/{train2026,val2026,challenge}_opus-v4.parquet` | 1,055 / 250 / 195 |
| Prompt v4 | `prompts/label_v4.md` | Kenya 2027 context + aliases |

## Settled by ablation - do not relitigate

- **Class weighting is harmful here**: -6.7pt, 3x false positives on benign
  posts. Use plain CE + threshold tuning. Focal loss adds nothing once
  weights are removed (0.6981 vs 0.7024, inside seed noise).
- **LLRD hurts** (-5.3pt). **Label smoothing neutral.**
- **Mix, never two-stage**: 2026-only continuation catastrophically forgot
  (offensive F1 0.483 -> 0.279 on 2013 data).
- **DAPT**: large LM gain, ~+2pt classification (inside 1-seed noise). Keep,
  don't over-claim.
- **Seed sd on unan macro-F1 ~2.1pt.** Nothing smaller is a result.
- **Opus oversample ×5 overfits 2026 vs 2013**; ×3 is the measured better
  tradeoff on seed 1337 (pending 3-seed).

## The open problem, precisely

Opus-validated 2026 labels improve challenge fit. The remaining question is
whether Opus ×3 holds across seeds with unanimous inside noise of shipped.
If yes, promote with a tuned hate threshold for triage. If not, keep
`d3-s1337` and treat Opus checkpoints as a 2026 shadow model. Coded
incitement without a protected-group target remains mostly `offensive` under
the validated taxonomy, so p_hate alone will not catch the 14 known coded
posts.

---

## Roadmap

1. **In flight / next:** 3-seed confirm for Opus ×3 (`run_r3_mix3_seeds.sh`).
2. If confirm passes: decide promote vs shadow-deploy; set hate threshold
   from val sweep (~0.24–0.28 for Opus).
3. Optional: sample remaining 940 for coverage.
4. Flag heads only after measured human-positive support is adequate.

---

## Infrastructure notes

- **GPU = Modal** (`modal_train.py`, A100, volume `hatespeech-finetune`).
- Labelling: `13_label_drive.py`; manual `out/opus_v4_full_manual/`.
- Train path: `23_opus_partial_merge.py`, `24_prep_opus_v4.py`,
  `run_r3_opus_batch.sh`, `run_r3_mix_sweep.sh`, `run_r3_threshold_spot.sh`,
  `run_r3_mix3_seeds.sh`.
- **v5 transformers gotcha**: `from_pretrained(dtype=torch.float32)`.

## Pending / deferred

- Mix ×3 3-seed confirm (promote gate).
- Remaining ~940 Opus v4 labels.
- Flag-head pilot.
- Deployment threshold/quantisation on whatever is promoted.

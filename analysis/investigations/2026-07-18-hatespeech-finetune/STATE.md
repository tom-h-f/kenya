# STATE - Kenya hate-speech classifier

Single source of truth for where this investigation stands. Updated
2026-07-25 (mix ×3 3-seed confirm). If this disagrees with any other doc,
this wins.

Reading order for someone new: this file, then `README.md` (scripts + how to
run), then `findings.md` (Plan D + round 2 results) and `findings-plan-a.md`
(labelling results). Historical planning docs are in `archive/`.

---

## One-paragraph status

Opus v4 labels (1,500; human-validated). Opus ×5 rejected (unan −3.4pt).
Mix sweep + **3-seed confirm** of **Opus ×3**: challenge **0.612 ± 0.010**
(+7.1pt vs shipped) and unanimous **0.683 ± 0.009** (−1.5pt, inside ~2.1pt
seed noise). Stop-gate cleared on the challenge/collapse rule. Production
still `d3-s1337` until an explicit promote + hate-threshold choice; Opus ×3
is the approved research / optional-replace candidate. Remaining 940 labels
and flag heads stay deferred.

## Current decision

- Opus ×3 (AfriHate ×1 + Opus train ×3): **APPROVED candidate** (3-seed).
- Replace production `d3-s1337` immediately: **NO** — needs explicit promote
  plus deployment threshold (val best-F1 ~0.24 / R≥0.80 ~0.28 on Opus).
- Opus ×5: **rejected** (unan past seed noise).
- Opus ×3 + hard-flag oversample: **reject for ship**.
- Finish remaining 940 Opus labels: **DEFER**.
- Flag-head pilot: **NOT APPROVED**.
- Shipped weights remain `d3-s1337` until promote.

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

**Pareto:** Opus ×3 → confirmed below.

### Mix ×3 3-seed confirm (measured 2026-07-25)

App `ap-mMx1qzqwGZaX6U9159ZE6r`. Recipe: AfriHate ×1 + Opus ×3. Seeds
1337/1338/1339. Artifacts: `model-r3-mix3-s{1337,1338,1339}/`.

| eval set | baseline | mix ×3 mean ± sd | Δ |
|---|---|---|---|
| `challenge_opus-v4` | 0.541 | **0.612 ± 0.010** | **+0.071** |
| `test_unanimous` | 0.698 | 0.683 ± 0.009 | −0.015 |
| old `gold` | 0.428 | 0.421 ± 0.041 | −0.007 |

Per-seed challenge / unan: 0.614/0.690, 0.601/0.685, 0.621/0.673.

**Verdict:** stop-gate **PASS** (clear challenge gain; unan drop inside seed
noise). Not auto-shipped — promote is an operator decision with threshold
tuning.

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
## What is banked

| asset | where | number |
|---|---|---|
| Shipped classifier `d3-s1337` | HF `tom-h-f/kenya-hatespeech-afroxlmr` (private); Drive `out/model-d/`; Modal vol | unan macro-F1 **0.688 ± 0.021** (3 seeds), full 0.592 |
| **Opus ×3 candidate (approved)** | Modal `model-r3-mix3-s{1337,1338,1339}/` | challenge **0.612 ± 0.010**; unan **0.683 ± 0.009** |
| Round-3 Opus ×5 (rejected) | Modal `model-r3-opus-s{1337,1338,1339}/` | challenge 0.610 ± 0.012; unan 0.664 ± 0.006 |
| Mix sweep seed-1337 | Modal `model-r3-mix-r{1,2,3,3hard}-s1337/` | led to ×3 |
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
- **Opus oversample ×5 overfits 2026 vs 2013**; **×3** is the confirmed
  better tradeoff (3-seed).

## The open problem, precisely

Opus ×3 clears the statistical stop-gate. What remains is product choice:
promote `r3-mix3` (with hate thr ~0.24–0.28) or keep `d3-s1337` and shadow
the Opus model on 2026 traffic. Coded incitement without a protected-group
target remains mostly `offensive` under the validated taxonomy, so p_hate
alone will not catch the 14 known coded posts.

---

## Roadmap

1. **Operator:** promote Opus ×3 vs keep shipped + shadow.
2. If promote: push best seed (or seed-mean ensemble) to HF; set hate
   threshold from val sweep (~0.24–0.28).
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

- Explicit promote / HF push for Opus ×3.
- Remaining ~940 Opus v4 labels.
- Flag-head pilot.
- Deployment threshold/quantisation on whatever is promoted.

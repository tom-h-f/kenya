# STATE - Kenya hate-speech classifier

Single source of truth for where this investigation stands. Updated
2026-07-25 (Opus ×3 **promoted**). If this disagrees with any other doc,
this wins.

Reading order for someone new: this file, then `README.md` (scripts + how to
run), then `findings.md` (Plan D + round 2 results) and `findings-plan-a.md`
(labelling results). Historical planning docs are in `archive/`.

---

## One-paragraph status

**Production candidate: Opus ×3 seed 1337** (`r3-mix3-s1337`), pushed to HF
`tom-h-f/kenya-hatespeech-afroxlmr`. 3-seed: challenge **0.612 ± 0.010**
(+7.1pt), unanimous **0.683 ± 0.009** (−1.5pt, inside seed noise). Default
hate decision threshold for triage: **0.28** (val R≥0.80, P≈0.58). Prior
`d3-s1337` remains on the Modal volume as rollback. **Model work for this
investigation is closed** — next work is wiring scores into the live monitor
/ desk brief, not further training. Remaining 940 labels and flag heads stay
deferred.

## Current decision

- Opus ×3 seed 1337: **PROMOTED** (HF private repo above).
- Rollback: Modal `model-d3-s1337` / previous HF revision if needed.
- Hate threshold (deploy): **0.28** on `p_hate` (not argmax); val best-F1
  alternative 0.24 if maximizing F1 over precision.
- Further classifier ablations / oversample sweeps: **STOP**.
- Finish remaining 940 Opus labels / flag heads: **DEFER** (not blocking).
- Next product work: score live posts with this model; feed Phase 5 desk
  brief / incitement monitoring — not more fine-tunes.

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

**Verdict:** stop-gate **PASS**. **Promoted 2026-07-25:** seed 1337 → HF
`tom-h-f/kenya-hatespeech-afroxlmr`; deploy hate thr **0.28**.

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
| **Production Opus ×3 `r3-mix3-s1337`** | HF `tom-h-f/kenya-hatespeech-afroxlmr`; Modal `model-r3-mix3-s1337/` | challenge 0.612 ± 0.010; unan 0.683 ± 0.009; thr 0.28 |
| Rollback `d3-s1337` | Modal `model-d3-s1337/` (prior HF revision) | unan 0.688 ± 0.021 |
| Opus ×3 other seeds | Modal `model-r3-mix3-s{1338,1339}/` | banked, not shipped |
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

Classifier optimization for this batch is **closed**. Open product work is
operational: score the live Kenyan stream with Opus ×3 + thr 0.28, surface
hate/offensive into the desk brief / monitoring path, and only reopen
labelling if production errors show a systematic taxonomy miss (especially
coded menace labelled `offensive`).

---

## Roadmap

1. ~~Promote Opus ×3~~ **done** (HF + thr 0.28).
2. Wire model into live scoring (`12_score_corpus` / monitor / `kma`).
3. Feed Phase 5 desk brief with hate/offensive flags.
4. Optional later: sample remaining 940 only if live errors demand it.
5. Flag heads: still deferred.

---

## Infrastructure notes

- **GPU = Modal** (`modal_train.py`, A100, volume `hatespeech-finetune`).
- Labelling: `13_label_drive.py`; manual `out/opus_v4_full_manual/`.
- Train path: `23_opus_partial_merge.py`, `24_prep_opus_v4.py`,
  `run_r3_opus_batch.sh`, `run_r3_mix_sweep.sh`, `run_r3_threshold_spot.sh`,
  `run_r3_mix3_seeds.sh`.
- **v5 transformers gotcha**: `from_pretrained(dtype=torch.float32)`.

## Pending / deferred

- Wire promoted model into live monitor / desk brief.
- Remaining ~940 Opus v4 labels (only if live errors demand).
- Flag-head pilot.
- Quantisation measurement if edge deploy is required.

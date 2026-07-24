# STATE - Kenya hate-speech classifier

Single source of truth for where this investigation stands. Updated
2026-07-24. If this disagrees with any other doc, this wins.

Reading order for someone new: this file, then `README.md` (scripts + how to
run), then `findings.md` (Plan D + round 2 results) and `findings-plan-a.md`
(labelling results). Historical planning docs are in `archive/`.

---

## One-paragraph status

Round-3 trained on **1,500 Opus v4 labels** (1,100 automated Claude Code +
400 manual; prompt `label_v4.md`) mixed with the settled 2013 recipe. On the
Opus-labelled challenge set it gains **+6.9pt macro-F1** (0.541 -> 0.610 ±
0.012). On 2013 `test_unanimous` it regresses **−3.4pt** (0.698 -> 0.664 ±
0.006), past seed noise (~2.1pt). Old gold is flat (−1.4pt). Production
weights stay `d3-s1337`. Opus-partial is a research / 2026-taxonomy candidate
only. Remaining 940 posts and flag heads stay deferred.

## Current decision

- Replace production `d3-s1337`: **NO**. Unanimous regression (−3.4pt) exceeds
  seed noise; challenge gain is partly circular (Opus labels on challenge).
- Use Opus v4 partial as training signal for further experiments: **YES**,
  with the single-labeller caveat and no Opus gold coverage.
- Finish remaining 940 Opus labels before any further claim: **DEFER**. Not
  required for the measured go/no-go above.
- Flag-head pilot: **NOT APPROVED**. Partial-set support is still thin
  (dehumanisation 43 / violence 44 / coded 58; ethnic_targeting = hate count).
- Existing shipped model remains the production candidate.

---

## Round-3 result (measured 2026-07-24)

Recipe: DAPT `out/dapt-afro-xlmr` + 2013 HateSpeech_Kenya + AfriHate + Opus
`train2026_opus-v4` ×5, no class weights / focal-gamma 2.0 to match shipped
comparator, val=`val2026_opus-v4`, seeds 1337/1338/1339. Modal app
`ap-C1wCWkA3GD6F6kbhgKaesM`. Artifacts on volume `hatespeech-finetune`:
`model-r3-opus-s{1337,1338,1339}/`, `eval-r3-{baseline,opus-s*}_*_metrics.json`.

| eval set | baseline `d3-s1337` | r3-opus mean ± sd | Δ |
|---|---|---|---|
| `challenge_opus-v4` (195, Opus labels) | 0.541 | **0.610 ± 0.012** | **+0.069** |
| `test_unanimous` (2013) | 0.698 | 0.664 ± 0.006 | −0.034 |
| old `gold` (Gemini-era labels) | 0.428 | 0.414 ± 0.011 | −0.014 |

Stop-gate from `docs/plans/2026-07-24-opus-v4-train.md`: promote only if unan
beats shipped by >~2.1pt **or** clear challenge gain without 2013 collapse.
Challenge gain is clear and stable; unanimous drop is real (above seed noise)
but not catastrophic forgetting. Net: **do not promote**.

Caveats:
- Challenge labels are Opus; primary metric measures fit to Opus taxonomy.
- No Opus-labelled gold in the partial set (0/283 prior gold IDs labelled).
- Single labeller (`claude-opus-code` / model `opus`); operator accepted Opus
  over human on the calibration disagreements.

## What is banked

| asset | where | number |
|---|---|---|
| Shipped classifier `d3-s1337` | HF `tom-h-f/kenya-hatespeech-afroxlmr` (private); Drive `out/model-d/`; Modal vol | unan macro-F1 **0.688 ± 0.021** (3 seeds), full 0.592 |
| Round-3 Opus candidate (not shipped) | Modal vol `model-r3-opus-s{1337,1338,1339}/` | challenge 0.610 ± 0.012; unan 0.664 ± 0.006 |
| DAPT encoder | HF `tom-h-f/kenya-dapt-afroxlmr`; Modal vol `dapt-afro-xlmr/` | corpus perplexity 16.0 -> 5.2 |
| Opus v4 partial labels | `out/labels_2026_opus-v4-partial.parquet`; report `out/21_opus_partial_1500_report.json` | 1,500 rows; neither 676 / offensive 558 / hate 266; 24.1% changed vs Gemini |
| Opus v4 splits | `out/{train2026,val2026,challenge}_opus-v4.parquet` | 1,055 / 250 / 195; gold unavailable |
| 2026 label batch (prior) | `out/labels_2026_full.parquet` (dual), `out/labels_2026_full_final.parquet` (single) | 2,440 rows, dual-labelled, kappa 0.674 |
| Round-2 splits | `out/{train2026,val2026,gold,challenge}.parquet` | 1,662 / 300 / 283 / 195 |
| Corpus prevalence (measured) | random control stratum | **5.7% positive, 1.4% hate** |
| Calibrated taxonomy set | `out/blind_check_coded_calibration.csv` | 120 rows; protected-target boundary adjudicated |
| Prompt-v3 heldout | `out/heldout_v3_{human,scored}.csv`, `out/20_heldout_report.json` | 93 rows; v3 improves ~10pt but fails gates |
| Prompt v4 | `prompts/label_v4.md` | Kenya 2027 context + aliases; taxonomy from v3 |

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

## The open problem, precisely

Round-2 `r2-mixed` moved 13 of 14 known coded posts down in p_hate under old
Gemini labels. Opus v4 relabelling (partial) changed 24.1% of labels on the
1,500 overlap and, after mixed retraining, fits the Opus challenge set better
(+6.9pt) while giving back ~3.4pt on 2013 unanimous. The remaining gap is not
"more of the same recipe": either recover 2013 with mix/oversample tuning, get
an independent (non-Opus) challenge reference, or explicitly accept a
2026-taxonomy tradeoff for monitoring. Prompt-v3 promotion gates and the
independent human gate from the old roadmap remain unmet for full-corpus
claims.

---

## Roadmap if the investigation resumes

1. Optional: finish or sample the remaining 940 for coverage, not as a
   promotion requirement.
2. If pursuing a shippable Opus model: rebalance mix / `--extra-repeat` to
   claw back unanimous without killing challenge gain; re-run 3 seeds.
3. Build a small independent human challenge set that is **not** Opus-prelabelled.
4. Flag heads only after measured human-positive support is adequate.
5. Deployment remains separate and should keep using `d3-s1337` until a
   candidate clears the unan seed-noise bar or an explicit product tradeoff
   is recorded.

---

## Infrastructure notes

- **GPU = Modal** (`modal_train.py`, A100, volume `hatespeech-finetune`
  mounted at out/). `uv run modal run --detach modal_train.py --cmd "..."
  --spawn`. Free credits cover this. HF push via Modal secret `huggingface`.
- **Labelling** (`13_label_drive.py`) supports `agy`, Cursor, and Claude CLIs
  (`claude-opus-code`). Manual path: `out/opus_v4_full_manual/{inbox,outbox}`.
- Merge / prep: `23_opus_partial_merge.py`, `24_prep_opus_v4.py`,
  `run_r3_opus_batch.sh`.
- **v5 transformers gotcha**: `from_pretrained(dtype=torch.float32)` required
  or fp16 AMP crashes on afro-xlmr's fp16 weights.
- Colab is abandoned (free-tier preemption); notebooks kept as backup only.

## Pending / deferred

- Remaining ~940 Opus v4 labels (spend-limited); not blocking the no-promote
  decision above.
- Flag-head pilot.
- 89 agy Sonnet-4.6 chunks (38 done) remain parked; no need to resume.
- Opus labels on the 93-row heldout are retained as provenance, not an
  independent reference.
- Deployment remains separate and should use the shipped model only after
  threshold/quantisation measurement.

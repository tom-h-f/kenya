# STATE - Kenya hate-speech classifier

Single source of truth for where this investigation stands. Updated
2026-07-22. If this disagrees with any other doc, this wins.

Reading order for someone new: this file, then `README.md` (scripts + how to
run), then `findings.md` (Plan D + round 2 results) and `findings-plan-a.md`
(labelling results). Historical planning docs are in `archive/`.

---

## One-paragraph status

We have a shipped 3-class (neither/offensive/hate) classifier at **0.688 ±
0.021 macro-F1**. Prompt v4 + Claude Code Opus produced a new working label
set of **1,100 rows** (first 44 chunks of the 2,440 batch). The operator
reviewed Opus vs human calibration disagreements and preferred Opus. Full
2,440 labelling was stopped by Claude Code spend limits and is **not
required**. Next work, if any, is rebuilding train/val splits from the 1,100
and retraining with the settled plain-CE mixed recipe.

## Current decision

- Working labels: **`out/labels_2026_opus-v4-partial.parquet`** (1,100 rows,
  Opus / prompt v4 / `claude-opus-code`).
- Completing the remaining 1,340 of the 2,440 batch: **NOT REQUIRED**.
- Flag-head pilot: still **NOT APPROVED** until rare-flag support is measured
  on this set and judged sufficient.
- Shipped model remains production until a retrain on the 1,100 is evaluated.

---

## What is banked

| asset | where | number |
|---|---|---|
| Shipped classifier `d3-s1337` | HF `tom-h-f/kenya-hatespeech-afroxlmr` (private); Drive `out/model-d/`; Modal vol | unan macro-F1 **0.688 ± 0.021** (3 seeds), full 0.592 |
| DAPT encoder | HF `tom-h-f/kenya-dapt-afroxlmr`; Modal vol `dapt-afro-xlmr/` | corpus perplexity 16.0 -> 5.2 |
| 2026 label batch | `out/labels_2026_full.parquet` (dual), `out/labels_2026_full_final.parquet` (single, used for round 2) | 2,440 rows, dual-labelled, kappa 0.674 |
| Round-2 splits | `out/{train2026,val2026,gold,challenge}.parquet` | 1,662 / 300 / 283 / 195 |
| Corpus prevalence (measured) | random control stratum | **5.7% positive, 1.4% hate** |
| Calibrated taxonomy set | `out/blind_check_coded_calibration.csv` | 120 rows; protected-target boundary adjudicated |
| Prompt-v3 heldout | `out/heldout_v3_{human,scored}.csv`, `out/20_heldout_report.json` | 93 rows; v3 improves ~10pt but fails gates |
| Opus v4 partial labels | `out/labels_2026_opus-v4-partial.parquet` (= merge of 1,100); report `out/21_opus_partial_1100_report.json` | neither 476 / offensive 384 / hate 240; 27.2% label movement vs prior Gemini-final |

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

Prompt v4 + Opus is now the accepted labelling standard for this investigation.
The 1,100-row partial set is mined-stratum heavy (`p_hate_top` 905 / `nli_tail`
150 / `lexicon` 45) and contains **no** `random_control` or `p_offensive_top`
rows, so it cannot estimate corpus prevalence. Versus the prior Gemini-final
labels on the same IDs, 27.2% of class labels moved (mostly neither↔offensive
and some hate boundary shifts). Retrain quality on this set still needs to be
measured; coded-incitement under-detection may persist.

---

## Roadmap if the investigation resumes

1. Rebuild train/val/gold/challenge splits from
   `labels_2026_opus-v4-partial.parquet`, mixed with 2013 data as before.
2. Run the settled plain-CE mixed recipe over three seeds on Modal.
3. Evaluate the 14 known coded posts and 2013 regression set; compare to the
   shipped `d3-s1337` numbers.
4. Only if rare-flag support is adequate, reconsider flag heads.
5. Completing the remaining 1,340 of the 2,440 batch is optional, not a gate.

---

## Infrastructure notes

- **GPU = Modal** (`modal_train.py`, A100, volume `hatespeech-finetune`
  mounted at out/). `uv run modal run --detach modal_train.py --cmd "..."
  --spawn`. Free credits cover this. HF push via Modal secret `huggingface`.
- **Labelling** (`13_label_drive.py`) supports `agy`, Cursor, and Claude CLIs.
  Cursor runs in read-only `ask` mode because print mode otherwise has write
  tools; parsing remains strict and resumable.
- **v5 transformers gotcha**: `from_pretrained(dtype=torch.float32)` required
  or fp16 AMP crashes on afro-xlmr's fp16 weights.
- Colab is abandoned (free-tier preemption); notebooks kept as backup only.

## Pending / deferred

- Remaining 1,340 of the Opus v4 full batch: deferred; not a gate.
- 89 agy Sonnet-4.6 chunks (38 done) remain parked; no need to resume.
- Opus labels on the 93-row heldout are retained as provenance, not an
  independent reference.
- Deployment remains separate and should use the shipped model only after
  threshold/quantisation measurement.

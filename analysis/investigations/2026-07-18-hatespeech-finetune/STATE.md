# STATE - Kenya hate-speech classifier

Single source of truth for where this investigation stands. Updated
2026-07-21. If this disagrees with any other doc, this wins.

Reading order for someone new: this file, then `README.md` (scripts + how to
run), then `findings.md` (Plan D + round 2 results) and `findings-plan-a.md`
(labelling results). Historical planning docs are in `archive/`.

---

## One-paragraph status

We have a shipped 3-class (neither/offensive/hate) classifier at **0.688 ±
0.021 macro-F1**, reproducible across three seeds and pushed to HF. It still
under-detects coded incitement. The human calibration resolved the taxonomy:
`hate` requires an identifiable protected-group target; generic abuse or
violence is `offensive`, with separate flags. Prompt v3 improved substantially
on a fresh 93-row challenge-weighted set, but **failed its predeclared
promotion gates**. Gemini v3 reached 0.806 exact / 0.786 macro-F1 with hate
P/R 0.809/0.630; Cursor v3 reached 0.774 / 0.757 with hate P/R 0.600/0.889.
Neither clears 0.85 exact plus 0.70 hate precision and recall. The reference
was Opus-prelabelled then human-validated, not independent or blind, so these
numbers may contain anchoring bias. **Do not relabel 2,440 rows or train flag
heads from the current targets.**

## Current decision

- Full prompt-v3 relabel: **NOT APPROVED**.
- Flag-head pilot: **NOT APPROVED**; human positive support is only 6
  dehumanisation / 6 violence / 27 protected-targeting / 8 coded-language.
- Existing shipped model remains the production candidate.
- If work continues, use the 28 v3 inter-labeller disagreements for prompt-v4
  error analysis, then run a genuinely independent human gate.

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

Round-2 `r2-mixed` moved 13 of 14 known coded posts down in p_hate. The old
Gemini labels were directionally conservative: 178 Gemini-not-hate/Cursor-hate
rows versus 39 in reverse. Human calibration then exposed a second problem:
the v2 prompt contradicted itself by calling targetless coded violence `hate`
while defining hate as protected-group attack. Prompt v3 fixes that boundary
and raises held-out exact agreement from 0.710 to 0.806 for Gemini and 0.677
to 0.774 for Cursor. The remaining errors are asymmetric: Gemini v3 is precise
but misses hate; Cursor v3 catches hate but over-calls it. Their agreement is
only 65/93, though those 65 agreed rows match the validated reference 92.3%.

---

## Roadmap if the investigation resumes

1. Inspect the 28 v3 Gemini/Cursor disagreements and both models' flag errors.
2. Draft prompt v4 only if the errors form a correctable rule, not case-by-case
   exceptions.
3. Build a new independent human set. Do not prelabel it with a model.
4. Require 0.85 exact agreement and hate precision/recall >= 0.70 for every
   labeller that will generate training labels.
5. Require measured human-positive support before any flag head. The current
   rare-flag counts are insufficient for a defensible pilot.
6. Only after those gates pass: relabel, rebuild splits, run the settled
   plain-CE mixed recipe over three seeds, then evaluate the 14 known coded
   posts and 2013 regression set.

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

- 89 agy Sonnet-4.6 chunks (38 done) remain parked; no need to resume.
- Opus labels on the 93-row heldout are retained as provenance, not an
  independent reference.
- Deployment remains separate and should use the shipped model only after
  threshold/quantisation measurement.

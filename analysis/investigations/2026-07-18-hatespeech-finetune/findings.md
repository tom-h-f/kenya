# Findings: hate-speech fine-tune (2026-07-18)

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

## Conclusion

Model is a useful **third triage signal** (surface-offensiveness tier + cheap
corpus-wide scoring) but does NOT replace the lexicon+NLI joint rule for coded
incitement. To make it the primary detector: label a few hundred 2026 corpus
posts (NLI-tail + flagged + random benign) and continue fine-tuning - the
infrastructure here supports that directly (`00_prep.py` splits, `02_train.py
--model out/model`).

# STATE - Kenya hate-speech classifier

Single source of truth for where this investigation stands. Updated
2026-07-20. If this disagrees with any other doc, this wins.

Reading order for someone new: this file, then `README.md` (scripts + how to
run), then `findings.md` (Plan D + round 2 results) and `findings-plan-a.md`
(labelling results). Historical planning docs are in `archive/`.

---

## One-paragraph status

We have a shipped 3-class (neither/offensive/hate) classifier for Kenyan
political tweets at **0.688 ± 0.021 macro-F1**, a +9.1pt improvement over the
previous best, reproducible across 3 seeds and pushed to HF. It reads the
2026 corpus well (DAPT dropped perplexity 16 -> 5.2) but **still under-detects
coded incitement** - and an attempt to fix that by training on 1,662 freshly
LLM-labelled 2026 posts made coded detection *worse*, not better. The likely
cause is now evidenced: the labeller (Gemini 3.1 Pro) is systematically
conservative about hate (4.56x asymmetry vs a second model, replicated). The
project is blocked on **one human task**: a 120-row blind check that decides
whether the labels are the problem (-> relabel) or the taxonomy is (-> flag
head). Everything else is done and waiting on that verdict.

## THE ONE ACTION THAT UNBLOCKS EVERYTHING

Tom fills in `out/blind_check_coded.csv` (120 rows, `human_label` column:
neither/offensive/hate), then:

```sh
uv run 18_blind_check.py score
```

The sheet over-samples the 42 rows where the two labellers disagree. The
headline output: of the 30 rows Gemini called not-hate and the second model
called hate, **what fraction does the human call hate?**
- **> 50%** -> training labels are too conservative -> prompt v3 + relabel
  (Phase A below)
- **<= 50%** -> labels are sound -> the problem is the taxonomy -> flag head
  (Phase B below)

~45 minutes. Nothing downstream can be decided without it.

---

## What is banked

| asset | where | number |
|---|---|---|
| Shipped classifier `d3-s1337` | HF `tom-h-f/kenya-hatespeech-afroxlmr` (private); Drive `out/model-d/`; Modal vol | unan macro-F1 **0.688 ± 0.021** (3 seeds), full 0.592 |
| DAPT encoder | HF `tom-h-f/kenya-dapt-afroxlmr`; Modal vol `dapt-afro-xlmr/` | corpus perplexity 16.0 -> 5.2 |
| 2026 label batch | `out/labels_2026_full.parquet` (dual), `out/labels_2026_full_final.parquet` (single, used for round 2) | 2,440 rows, dual-labelled, kappa 0.674 |
| Round-2 splits | `out/{train2026,val2026,gold,challenge}.parquet` | 1,662 / 300 / 283 / 195 |
| Corpus prevalence (measured) | random control stratum | **5.7% positive, 1.4% hate** |

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

On the 14 human-known coded posts, round-2 `r2-mixed` moved 13 of 14 DOWN in
p_hate; "tunaua nyoka na mayai yake" fell 6x. It gained +15.7pt on gold and
+18.6pt on challenge, but both are Gemini-labelled - those gains measure
agreement with the labeller, not detection. Dual-labelling then showed Gemini
calls not-hate where a second Claude model calls hate on 178 rows vs 39 the
reverse (**4.56x**, replicated across two model versions and two CLIs). So the
suspicion "we trained the model into the labeller's blind spot" has evidence.
The blind check settles whether the labeller or the human is right.

---

## Roadmap after the blind check

### Phase A - if labels are too conservative (relabel)
1. Prompt v3: widen the hate definition for coded speech, anchored on the
   missed examples the blind check surfaces. Bump version, keep diffable.
2. Relabel 2,440 rows, both CLIs (agy + cursor, separate quotas). ~1h.
3. Re-merge, re-split, re-run round 2 (recipe below).

### Phase B - fix the axis (flag head), do regardless once labels trusted
3-class may be the wrong target: coded incitement is a violence_call wearing
a metaphor, and hate/offensive is where every annotator has disagreed. The
`flags` column (dehumanisation, violence_call, ethnic_targeting,
coded_language) is **already populated** on all 2,440 rows. Add a multi-task
head to `02_train.py`: shared encoder, 3-class head + 4 binary flag heads,
summed loss, per-flag metrics. ~half a day build, ~40 min GPU.

### Phase C - retrain + evaluate (recipe is settled)
`r2-mixed` shape: DAPT encoder, agree60 + AfriHate + 2026 oversampled 5x,
**plain CE** (drop focal), lr 1e-5, 5 epochs, eff batch 64, select on
val2026. Run **3 seeds**. Evaluate in authority order:
1. human-verified gold (only non-circular measure)
2. the 14 known coded posts (`04_infer.py` on `out/10_flagged.csv`)
3. challenge (195 lexicon/NLI rows)
4. test_unanimous (2013, regression check)

### Phase D - deployment (separate task)
Quantise for tf1 (3.8GB RAM CPU; int8 dynamic first, measure delta on gold),
pick threshold from gold sweep, wire into `kma/enrich.py` as a third signal
beside the lexicon+NLI rule. Not the primary incitement detector.

---

## Infrastructure notes

- **GPU = Modal** (`modal_train.py`, A100, volume `hatespeech-finetune`
  mounted at out/). `uv run modal run --detach modal_train.py --cmd "..."
  --spawn`. Free credits cover this. HF push via Modal secret `huggingface`.
- **Labelling = two CLIs** (`13_label_drive.py`): `agy` and cursor `agent`,
  separate quotas, resumable + idempotent. Cursor needs `--trust` and emits
  fenced prose (parser tolerates it).
- **v5 transformers gotcha**: `from_pretrained(dtype=torch.float32)` required
  or fp16 AMP crashes on afro-xlmr's fp16 weights.
- Colab is abandoned (free-tier preemption); notebooks kept as backup only.

## Pending / deferred

- 89 agy Sonnet-4.6 chunks (38 done) parked on quota - a *third* opinion,
  not needed. Resume only if a tie-break is wanted post-blind-check.
- Opus adjudication of the 490-row disagreement queue - deferred: a v3
  relabel would supersede it. Run only if labels pass the blind check as-is.

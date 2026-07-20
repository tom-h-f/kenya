# Round-2 plans: Kenya hate/offensive/neither classifier

Goal: classify tweets by Kenyan citizens on politics/society/culture into
hate / offensive / neither, with incitement sub-flags, well enough to serve as
primary triage for the 2027-election monitor. Round-1 result (this dir):
twitter-xlm-roberta-base on 2013-era data = test macro-F1 0.543, and coded
2026 Swahili/Sheng incitement scores "offensive" not "hate" - data era is the
binding constraint. Four workstreams below; A/B/C feed D.

Dependency order: B (cheap, immediate) -> A (labels) -> C (external data) -> D
(big train). A and C can run in parallel.

---

## Plan A - LLM labelling pipeline (~3-5k 2026 posts)

### Why not a local Swahili model as labeller
Checked (2026-07): swahili-gemma-1b (Crane AI, conversational 1B),
AfroLlama-8B, UlizaLlama-7B. All small; AfroBench shows even 27-70B open
models trail frontier models on Swahili understanding tasks, and label quality
is the entire point of this exercise. Swahili specialisation belongs in the
*encoder we train* (afro-xlmr, Plan D), not the labeller. Labellers = Claude
Code + Antigravity CLI dual-annotation (see A4). Local-model role, if any:
cheap third-opinion ensemble later - not round 1.

### A1. Sampling frame (script `05_sample.py`)
Pull from R2 posts via `kma.db`, then stratify. Target 5,000 rows:
- ~1,000 NLI-tail: `latest_incitement` max NLI score >= 0.85, no lexicon hit
- ~200 all lexicon hits (only ~50 joint-flagged exist - take every hit)
- ~800 current-model high `p_offensive`/`p_hate` (round-1 model scores corpus
  first - dogfood + hard-example mining)
- ~500 posts by coordination-cluster / flagged authors (manipulation-sweep set)
- ~2,500 random from corpus (calibration + honest class prior; expect mostly
  neither - that is the point)
Constraints: dedupe near-copies (copypasta) by normalised text hash + existing
embeddings cosine > 0.95; cap 10 posts/author; strip RTs to original text;
keep `platform_post_id`, `author_handle`, `created_at`, `text` only.
Output: `out/label_batch_001.parquet`.

### A2. Label schema (one row per post, JSONL from API -> parquet)
```
post_id, text, created_at,
label:            hate | offensive | neither
flags:            dehumanisation, violence_call, ethnic_targeting,
                  coded_language            (booleans, any subset)
target_group:     free text or null (e.g. kikuyu, luo, "politicians")
confidence:       high | medium | low
rationale:        one sentence, quotes the operative phrase
labeller:         model id
prompt_version:   semver of the prompt file
```
Keep `prompts/label_v1.md` in-repo; bump version on any edit; labels always
carry their prompt version - re-labelling is diffable.

### A3. Prompt design (`prompts/label_v1.md`)
- Class definitions: Davidson-compatible (so merges stay coherent) plus NCIC
  Act hate-speech definition (ethnic contempt, incitement) for the hate class.
- Kenya 2026-27 context block: election cycle, ethnic-bloc politics, the
  coded-term glossary from `kma/incitement.py` LEXICON (nyoka, madoadoa,
  watajua hawajui, kumira kumira, goons, mbogi...) with fp_risk notes -
  instructs "coded term alone is not hate; coded term + target + intent is".
- Language note: text may be English/Swahili/Sheng code-mix; translate
  mentally, judge meaning not language.
- 8-10 few-shot examples covering: explicit hate, coded hate, political
  criticism that is NOT hate (the NLI failure mode), vulgar-but-not-targeted
  (offensive), benign coded-term use, Sheng banter.
- Output: strict JSON via tool-use schema. Temperature 0.
### A4. Harness: dual CLI-agent labelling (DECIDED 2026-07-18)
Labellers = **Claude Code (headless) + Antigravity CLI (Gemini)**, both label
ALL 5k posts. Cross-model agreement replaces single-model self-consistency
and gives a free inter-annotator metric.
- Work queue: `out/chunks/chunk_NNN.jsonl`, 50 posts each (post_id + text
  only). Same chunks fed to both labellers.
- Driver `06_label_drive.py`: per chunk, invokes labeller command template
  with the prompt file + chunk, expects strict-JSON lines back, validates
  (label enum, flags consistent: neither => no violence_call), writes
  `out/labels/<labeller>/chunk_NNN.jsonl`; failed/invalid chunks re-queued.
  State file per labeller - resumable, idempotent, overnight-safe.
- Claude side concrete: `claude -p @prompts/label_v1.md @chunk --output-format
  json` (exact flags finalised at build time). Antigravity side: same prompt +
  chunk contract, invocation template filled in with Tom (headless/scripted
  mode); tool differences absorbed by the driver, output contract identical.
- Merge `07_label_merge.py`: rows where both agree on class -> accepted
  (keep both flag sets, OR the booleans, record per-labeller confidence);
  class disagreement -> adjudication queue. Report cross-model kappa overall
  and per class - if kappa < 0.5 something is broken (prompt or a labeller);
  stop and inspect before proceeding.

### A5. Human QA gate (blocks Plan D) (REVISED for dual-labeller)
- Tom adjudicates the **disagreement queue** (expect a few hundred rows) -
  human label wins, stored as `label_source=human`.
- Plus **100-row blind check** of randomly sampled AGREED rows (no model
  columns shown) - catches shared blind spots (two LLMs can agree wrongly on
  coded Sheng). Gate: >= 85% human agreement with the agreed-set labels AND
  hate-row agreement >= 70%; below gate -> fix prompt, bump version, relabel.
- Human-touched rows (adjudicated + blind-checked) seed the **2026 gold test
  set** - excluded from all training forever.

---

## Plan B - Clean existing 48k labels

Annotator count columns (`hate_speech, offensive_language, neither`) are an
unused asset: they encode agreement.

### B1. Agreement features (`00_prep.py` upgrade)
- `n_votes = sum(counts)`, `agreement = max(counts)/n_votes`,
  `is_unanimous = agreement == 1.0`.
- Distribution report first (how many unanimous / 2-of-3 / split rows) -
  decides thresholds; measure before choosing.

### B2. Filters
- Language ID (fasttext lid.176 or lingua): keep en/sw/mixed, drop clear
  other-language rows (Malay "masai" collisions etc.). Log drop count -
  expect low hundreds.
- Near-duplicate scan across the whole set (MinHash on 5-gram shingles) so
  variants of one tweet cannot straddle train/test.

### B3. Label-noise experiment matrix (small, on current base model)
Fixed evaluation: unanimous-only test split + (once Plan A lands) 2026 gold.
Train variants, 1 seed each, compare macro-F1:
1. all rows as-is (round-1 reference)
2. unanimous + 2/3-majority only, split/tie rows dropped
3. all rows, sample weight = agreement
4. tie rows relabelled by the Plan A LLM pipeline (cheap: only ~hundreds of
   rows), then variant 2 + relabels
Winner's recipe becomes the canonical `clean.parquet` for Plan D. Each variant
is a ~40-80 min MPS run or minutes on Colab - run as batch overnight.

---

## Plan C - Sibling datasets

### C1. Candidates (vet in this order)
| dataset | what | why |
|---|---|---|
| **AfriHate** (HF `afrihate/afrihate`, paper 2501.08284) | 15 African langs, native-speaker annotated, classes hate/abusive/neutral + target attributes | Swahili subset maps 1:1 to our taxonomy AND our flags (targets: ethnicity/politics/...). Top priority. |
| ~~XtremeSpeech~~ | ~20k Kenya passages | DROPPED 2026-07-19: access friction (request form), AfriHate covers the need. Revisit only if AfriHate ablation disappoints. |
| **HateDay** (2411.15462) | day-representative Twitter sample incl Swahili | Realistic class prior; useful for calibration split more than training. |
| Davidson 2017 (US English) | the original schema source | Probably skip: US slang, wrong domain; test-only ablation if bored. |
| "State of NLP in Kenya" survey (2410.09948) | catalogue | Mine bibliography for anything Kenya-specific missed above. |

### C2. Per-dataset intake checklist (script `07_siblings.py` + notes in
`siblings.md`)
1. License + redistribution terms recorded (some require citation/forms).
2. Label mapping table -> {hate, offensive, neither} written down explicitly;
   anything unmappable (e.g. "abusive to individual") documented.
3. Language filter: keep sw + Kenya-relevant en; tag `lang`.
4. Dedupe against our 48k and against each other (same MinHash pass as B2).
5. Add `source` column; never mix without provenance.
### C3. Keep-or-drop rule
Ablation on the current base model: train (cleaned 48k + candidate) vs
(cleaned 48k). Candidate stays only if Kenya-test macro-F1 does not drop and
hate-F1 improves OR coverage of coded-term challenge set improves. External
data that helps generic Swahili but hurts Kenya specificity gets sampled down
(cap at ~1x the Kenya data size), not fully dropped.

---

## Plan D - afro-xlmr-large on Colab A100

### D1. Base model + data
- Primary: `Davlan/afro-xlmr-large` (560M, XLM-R-large adapted to 17 African
  langs incl Swahili). Fallback comparison: `cardiffnlp/twitter-xlm-roberta-
  large` (twitter domain, weaker Swahili) - one run each, keep winner.
- Training corpus: canonical clean 48k (Plan B winner) + accepted siblings
  (Plan C) + 4.5k LLM labels (Plan A, minus gold). All with `source`,
  `agreement`/`confidence` weights, flags where present.

### D2. Stage 0 - domain-adaptive pretraining (biggest single lever)
Continue MLM on our own unlabelled 109k-post 2026 corpus (+ optionally the
sibling texts) before any classification: 1-2 epochs, seq 128, bf16,
~1-2h on A100 (measure with 200-step timing run first). This is what teaches
the encoder 2026 Sheng/political register that no labelled set covers.
Save as `afro-xlmr-large-kenya2026` -> base for all D3 runs. Gate: MLM loss on
held-out corpus posts drops materially vs stock model; downstream A/B one run
with vs without - keep only if it wins (it usually does: Gururangan et al.
don't-stop-pretraining result).

### D3. Fine-tune recipe
- **Multi-task head**: 3-class softmax + 4 sigmoid flag outputs, shared
  encoder. Loss = CE(class, weights) + 0.5 * masked-BCE(flags) - flags only
  supervised on rows that have them (LLM labels + AfriHate targets); mask
  elsewhere. Flags force the encoder to represent *why* something is hate -
  helps the small hate class and directly outputs the triage dimensions
  incitement.py cares about.
- Loss options, ablate in this order (one change at a time): weighted CE
  (reference) -> + label smoothing 0.05 -> focal loss gamma=2 (instead of
  weights) -> + agreement/confidence sample weights.
- Optimisation: lr 1e-5, layer-wise LR decay 0.9, warmup 6%, weight decay
  0.01, effective batch 64 (32 x grad-accum 2), bf16, max_len 128 (95th pct
  tweet len - verify on merged corpus), 5 epochs, early stop patience 2 on
  Kenya-val macro-F1, best-checkpoint select.
- Curriculum finish: after merged-corpus training, 1 extra epoch at lr 5e-6 on
  Kenya-only data (48k-clean + 2026 labels) so external data cannot dominate
  final decision boundaries.
- 3 seeds for the final recipe; report mean +/- sd; ship median-val-F1 seed
  (or logit-ensemble all 3 if inference budget allows).

### D4. Evaluation protocol (fixed before any D3 run)
1. **Kenya-clean test**: unanimous-label test split from Plan B (never
   trained on).
2. **2026 gold**: human-verified rows from Plan A QA (grow to ~500 over
   rounds).
3. **Coded-term challenge set**: ~100 hand-built contrastive pairs - each
   LEXICON term in (a) genuinely inciting context, (b) benign/reclaimed
   context. Metric: paired accuracy (both sides right). This is the direct
   test of the round-1 failure.
4. Metrics: macro-F1 (headline), hate recall at precision >= 0.6 (triage
   operating point), per-flag AUPRC, calibration ECE after temperature
   scaling on val.
Promotion gate vs round-1 model: +5 macro-F1 pts on (1), beats it on (2), and
>= 60% paired accuracy on (3).

### D5. Colab mechanics
- A100 40GB runtime; `pip install` cell (same deps as here + `wandb` optional).
- Data up: single `merged_corpus.parquet` + splits, uploaded to Drive;
  notebook mounts Drive; checkpoints -> Drive every epoch (Colab preemption
  survival); `--resume` already supported.
- Timing: measure 100 steps before committing to ablation count; A100 bf16 on
  560M @ bs64/len128 historically ~5-8 it/s -> full epoch on ~70k rows =
  ~3-5 min. If so, whole ablation matrix (~10 runs) fits one session. Measure
  first - do not trust these numbers.
- Model out: push each accepted checkpoint to private HF hub repo
  (`tomfrench/kenya-hatespeech-*`); tac2/tf1 pull from there.
- Keep every run's config + metrics in one `runs.jsonl` (append-only) for the
  writeup.

### D6. Deployment (after promotion gate)
- Score full corpus on tac2 MPS or Colab; persist to NEW R2 prefix
  `hatespeech/` (same pattern as `incitement/` - do NOT reuse labels/).
- Add `kma/hatespeech.py` scorer + `latest_hatespeech` db helper; wire into
  enrich worker as optional pass (tf1 is CPU + 3.8GB RAM: needs the 560M model
  quantised (int8 onnx) or nightly batch scoring on tac2 instead - decide by
  measuring tf1 inference speed, don't assume).
- Triage rule v2: replace/augment joint lexicon+NLI with
  `p_hate >= threshold_from_D4` OR (flag violence_call AND coded_language),
  validated on the same 100-row sheet method as incitement.py round 1.

---

## Sequencing + effort

1. **B1-B2** (half day, local): agreement stats, langid, dedupe, canonical
   test split. Unblocks everything.
2. **A1-A4** (1 day build + batch turnaround): sample, prompt, harness, 5k
   labels. **A5** human QA (~1-2h of Tom's time) - hard gate.
3. **C** in parallel with A batch wait (1 day): AfriHate + XtremeSpeech intake
   + ablations queued.
4. **B3** overnight ablation batch (local MPS or first Colab session).
5. **D** (1-2 Colab sessions): stage-0 DAPT, ablation matrix, final 3-seed
   run, eval report, HF push.
6. **D6** deployment + corpus scoring + triage-rule validation.

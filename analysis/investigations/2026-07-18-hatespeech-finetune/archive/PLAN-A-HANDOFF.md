# Plan A handoff: mine + dual-label 1,000 positive 2026 posts

Self-contained brief for the executing agent. Read fully before coding.
Companion docs in this dir: `plans-round2.md` (Plan A original design - this
file supersedes its A1/A4 sections for execution), `findings.md` (Plan D
results), `PLAN-D-HANDOFF.md` (the classifier this feeds).

## Mission

Produce **1,000 confirmed hate-or-offensive 2026 Kenyan posts**, dual-labelled
by two independent LLM agents, plus a random control stratum for honest
prevalence. This is the missing ingredient identified by Plan D: every model
so far trains on 2013-era labels and mis-reads the 2026 coded register
(*"tunaua nyoka na mayai yake"* scores p_hate ~0.01).

Note the target is a **yield of positives**, not a count of labelled rows.
Random sampling will not get there - see the arithmetic in A1.

## Context you must know

- Project: kenya-monitor-2027, X/Twitter monitor for the 2027 Kenyan election.
  Posts in R2, DuckDB access via `kma.db` (package `analysis/src/kma/`).
  Work from `analysis/investigations/2026-07-18-hatespeech-finetune/`, on
  `master`. Scripts are standalone PEP 723 (`uv run <script>`).
- **Corpus (measured 2026-07-19)**: 118,417 latest X posts; 109,057 have
  incitement scores. Of those: **52 lexicon hits total** (tiny - take all),
  **16,520 rows with max NLI score >= 0.9** (~15% of corpus - NLI
  over-triggers on ethnic-bloc horse-race talk, documented failure mode; it
  is a weak miner, not a label).
- **The good miner is now Plan D's classifier**: `out/model-d-dapt-focal`
  (afro-xlmr-large + DAPT + focal loss). On the 2013-era unanimous test:
  unan macro-F1 0.7024, hate P 0.586 / R 0.773 at argmax, offensive F1 0.478.
  **Its 2026 transfer is unmeasured** (Plan D's D4 spot-check was pending at
  handoff time) - so treat its scores as a *ranking signal for candidates*,
  never as labels. If D4 has since landed, read it first: a model that still
  refuses to fire on coded posts needs the lexicon/NLI strata weighted up.
- Class mapping everywhere: 0=neither, 1=offensive, 2=hate.

## Environment map

- This box (tac2): M4 Pro 24GB. Fine for mining SQL, chunking, driving the
  labellers. Not for GPU inference over 118k posts.
- **Modal** is the GPU path (`modal_train.py` in this dir, app
  `hatespeech-finetune`, volume of the same name mounted at `out/`).
  `uv run modal run modal_train.py --cmd "..."`, or `--detach ... --spawn`
  for long jobs. A100, ~$2.10/h, Starter plan has $30/mo free credits.
  The d-dapt-focal model dir already lives on that volume.
- R2 creds come from the monorepo-root `.env` via `kma.db` (auto-loaded).
- **agy CLI verified working headless (2026-07-19)**:
  `agy -p '<prompt text>' --model "Gemini 3.1 Pro (Low)"` returned clean
  JSONL, no prose, no code fences, in 9s for a 2-item test. Other models:
  `agy models`. Use `--print-timeout` (default 5m) for big chunks and
  `--dangerously-skip-permissions` only if a chunk needs file reads (prefer
  passing chunk text inline in the prompt - no file permissions needed).

## Deliverables (in order)

### A1. Candidate mining - `12_mine_candidates.py`

The yield arithmetic that drives everything: offensive+hate prevalence in
this corpus is unknown but plausibly 5-15%. Labelling 1,000 random posts
would return ~50-150 positives, so **do not sample randomly for the bulk**.
Mine ranked candidates and over-sample, then confirm by labelling.

Steps:
1. Score the full corpus with d-dapt-focal on Modal - reuse
   `04_infer.py:predict` over the cleaned corpus text (`08_export_corpus.py`
   already produces `out/dapt_corpus.parquet`, 76,197 deduped texts; extend
   it or write the scoring inline). Write
   `out/corpus_scored.parquet` (post_id, text, p_neither, p_offensive,
   p_hate). A100, a few minutes, ~$1.
2. Build strata (target ~2,500 candidates, tune after the pilot in A2):
   - **~1,200** top-ranked by `p_hate` (descending)
   - **~800** top-ranked by `p_offensive` among rows not already taken
   - **52** all lexicon hits (`latest_incitement`, `len(lexicon_hits)>0`)
   - **~150** NLI tail: max(dehumanisation, violence_call, othering) >= 0.9
     AND > political_criticism_score, sampled, excluding rows already taken
     (this stratum tests whether NLI catches things the classifier misses)
   - **~300 RANDOM control** - uniform from the corpus, NOT model-selected
3. Constraints (reuse existing helpers): dedupe near-copies with
   `00_prep.py:near_dup_clusters` (MinHash, already used by
   `08_export_corpus.py`); cap **10 posts per author**; keep only
   `post_id, author_handle, created_at, text, stratum, p_hate, p_offensive`.
4. Output `out/label_batch_001.parquet` + `out/12_mine_report.json` with
   per-stratum counts.

**The random control stratum is not optional.** It is the only way to
measure (a) true prevalence and (b) the miner's precision. Model-mined rows
are biased toward what the model already believes - excellent as training
data (hard-example mining), useless as a measure of anything.

### A2. Pilot - 200 rows, measure before scaling

Run the full A3 pipeline on 200 candidates first (stratified proportionally).
Report: per-stratum positive rate, cross-labeller agreement (Cohen's kappa),
seconds per chunk, and any parse failures. **Decision point - do not scale
past this without checking:**
- Positive rate in the mined strata below ~40% -> the miner is weak; raise
  the p_hate cut, or D4 says the model does not transfer and lexicon/NLI
  strata need weighting up. Recompute how many candidates 1,000 positives
  needs and say the new number out loud.
- Kappa below 0.5 -> something is broken (prompt or a labeller). Stop and
  inspect; do not label 2,500 rows with a broken prompt.

### A3. Dual-agent labelling - `prompts/label_v1.md` + `13_label_drive.py`

**Prompt** (`prompts/label_v1.md`, versioned - bump on any edit, labels
always carry their prompt version so relabels are diffable):
- Class definitions: Davidson-compatible (so the 48k merges coherently) plus
  the NCIC Act hate-speech definition (ethnic contempt, incitement).
- Kenya 2026-27 context block: election cycle, ethnic-bloc politics, and the
  coded-term glossary from `kma/incitement.py:LEXICON` (nyoka, madoadoa,
  watajua hawajui, kumira kumira, goons, mbogi) with its `fp_risk` notes.
  State explicitly: **a coded term alone is not hate; coded term + target +
  intent is.**
- Language note: English/Swahili/Sheng code-mix; judge meaning, not language.
- 8-10 few-shot examples covering: explicit hate, coded hate, political
  criticism that is NOT hate (the NLI failure mode), vulgar-but-not-targeted
  (offensive), benign coded-term use, Sheng banter.
- Output contract: **strict JSONL, one object per input, no prose, no code
  fences** (verified achievable with agy). Schema per row:
  `{post_id, label: hate|offensive|neither, flags: [dehumanisation,
  violence_call, ethnic_targeting, coded_language], target_group: str|null,
  confidence: high|medium|low, rationale: str}` - rationale must quote the
  operative phrase.

**Driver** (`13_label_drive.py`): chunk to `out/chunks/chunk_NNN.jsonl`
(**25 posts/chunk** - agy print mode defaults to a 5m timeout; keep chunks
small enough to land well inside it), feed identical chunks to both
labellers, validate output (label enum; flag consistency - `neither` implies
no `violence_call`; every input id present), write
`out/labels/<labeller>/chunk_NNN.jsonl`. Per-labeller state file, resumable,
idempotent, safe to re-run. Re-queue failed/invalid chunks up to 2 times,
then park them in `out/labels/failed/`.

Labellers (independent model families - that is the point):
- **agy**: `agy -p "$(cat prompts/label_v1.md)\n\n$(cat chunk)" --model
  "Gemini 3.1 Pro (Low)" --print-timeout 10m`
- **Claude Code headless**: `claude -p ... --output-format json`

Run chunks with concurrency ~4 and **measure throughput in the pilot** -
per-tool rate limits are unknown; do not assume.

### A4. Merge + QA - `14_label_merge.py`

- Both labellers agree on class -> **accepted**; OR the flag booleans, keep
  both confidences and rationales.
- Class disagreement -> `out/adjudication_queue.csv` for Tom. Human label
  wins, stored as `label_source=human`.
- Report **Cohen's kappa** overall and per class. Prioritise the hate-class
  disagreements in the queue - they matter most and are fewest.
- **100-row blind check** of randomly sampled AGREED rows, model columns
  hidden - catches shared blind spots (two LLMs can agree wrongly on coded
  Sheng). Gate: >= 85% human agreement overall AND >= 70% on hate rows.
  Below gate -> fix prompt, bump version, relabel.
- Outputs: `out/labels_2026_v1.parquet` (all rows, with stratum + provenance)
  and a headline count of confirmed hate + offensive. **That count is the
  deliverable** - if it is short of 1,000, mine another batch from the next
  rank band rather than loosening the labelling standard.

### A5. Split discipline - read this before anyone trains on it

- Human-touched rows (adjudicated + blind-checked) + **the entire random
  control stratum** become the **2026 gold test set**. Excluded from training
  forever. Record the ids in `out/gold_2026_ids.json`.
- Model-mined rows are training data only. Anyone reporting a test score on
  mined rows is measuring the miner, not the model.
- Everything carries `prompt_version` and `labeller`; relabels are diffs.

## Explicitly out of scope

- Retraining Plan D with these labels (a later round; it also adds the
  multi-task flag head - the `flags` field exists for that).
- Wiring anything into `kma/enrich.py` or scoring the corpus in production.
- The coded-term challenge set (needs this data first).

## Working agreements

- Sub-minute smoke of every script before any long run; pilot before scale.
- Measure, do not estimate: print per-chunk timing, positive rates, kappa.
  Numbers without a stratum/labeller/prompt-version label are worthless.
- Every long run resumable and idempotent (re-running a chunk must be safe).
- Update a `findings-plan-a.md` as results land, not at the end.
- Caveman comms with Tom; commit only when he asks.
- Tom adjudicates. Do not silently resolve class disagreements by rule.

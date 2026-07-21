# Opus v4 Relabelling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Relabel the 2,440-post 2026 batch with Claude Opus 4.6 Thinking using a versioned, Kenya-specific prompt and produce auditable comparison reports without overwriting prior labels.

**Architecture:** Keep the existing resumable chunk runner and single-labeller merge path. Add prompt v4 as a modular static reference, strengthen response and provenance validation, and add a small reporting utility for calibrated-set and full-batch comparisons. Run a calibration check and stratified pilot before resuming the isolated full tag.

**Tech Stack:** Python 3.12, pandas, scikit-learn, pytest, PEP 723 scripts via `uv`, `agy` Claude Opus 4.6 Thinking.

---

### Task 1: Add the versioned Kenya context prompt

**Files:**
- Create: `analysis/investigations/2026-07-18-hatespeech-finetune/prompts/label_v4.md`
- Create: `analysis/investigations/2026-07-18-hatespeech-finetune/test_prompt_v4.py`

**Step 1: Write the failing prompt-contract tests**

Load `label_v4.md` and assert that it contains:

```python
REQUIRED_CONTEXT = {
    "William Ruto": ("Kasongo", "Zakayo", "Sugoi"),
    "election slogans": ("Wantam", "Tutam"),
    "institutions": ("IEBC", "NCIC"),
    "languages": ("Kiswahili", "Sheng", "code-switching"),
}

def test_v4_has_required_kenyan_context() -> None:
    prompt = Path(__file__).with_name("prompts").joinpath("label_v4.md").read_text()
    for terms in REQUIRED_CONTEXT.values():
        assert all(term in prompt for term in terms)

def test_v4_aliases_are_not_automatic_hate_evidence() -> None:
    prompt = read_prompt()
    assert "political alias" in prompt
    assert "not a protected group" in prompt
    assert "term alone is not hate" in prompt
```

Also assert the strict JSONL output fields and the calibrated invariant that
`hate` is equivalent to protected-group targeting.

**Step 2: Verify the tests fail**

Run:

```bash
uv run --with pytest pytest -q test_prompt_v4.py
```

Expected: failure because `prompts/label_v4.md` does not exist.

**Step 3: Write prompt v4**

Preserve v3's taxonomy and output schema. Organise v4 into these explicit
sections:

1. task and governing boundary;
2. neutral Kenya 2027 political context;
3. actors, institutions, parties, coalitions, places, aliases, and slogans;
4. Kenyan English, Kiswahili, Sheng, and code-switching;
5. coded ethnic, exclusion, dehumanisation, and violence language;
6. flags and consistency rules;
7. hard positive, hard negative, quotation, sarcasm, and ambiguous examples;
8. strict JSONL contract.

State that `Kasongo` and `Zakayo` commonly identify Ruto in this corpus but
are not ethnic labels. Apply the same rule to parties, coalitions, supporters,
protest movements, and regional political blocs. Do not encode disputed
political claims as facts.

**Step 4: Verify prompt tests**

Run:

```bash
uv run --with pytest pytest -q test_prompt_v4.py
```

Expected: all prompt-contract tests pass.

**Step 5: Commit**

```bash
git add analysis/investigations/2026-07-18-hatespeech-finetune/prompts/label_v4.md \
  analysis/investigations/2026-07-18-hatespeech-finetune/test_prompt_v4.py
git commit -m "feat(analysis): add Kenya-aware label prompt"
```

### Task 2: Strengthen response validation and provenance

**Files:**
- Modify: `analysis/investigations/2026-07-18-hatespeech-finetune/13_label_drive.py`
- Modify: `analysis/investigations/2026-07-18-hatespeech-finetune/test_13_label_drive.py`

**Step 1: Write failing validation tests**

Add table-driven tests proving `parse_response` rejects:

- missing `confidence`, `rationale`, or `target_group`;
- invalid confidence values;
- `hate` without `ethnic_targeting`;
- `ethnic_targeting` on a non-hate label;
- non-null `target_group` without `ethnic_targeting`.

Add a test for a `run_manifest` helper:

```python
manifest = module.run_manifest(
    tag="opus-v4-pilot",
    prompt_path=tmp_path / "label_v4.md",
    labellers=["claude-opus-4.6"],
    input_name="/source/batch.parquet",
    rows=120,
)
assert manifest["prompt_version"] == "v4"
assert manifest["prompt_sha256"] == hashlib.sha256(prompt_bytes).hexdigest()
assert manifest["labellers"]["claude-opus-4.6"]["model"] == "Claude Opus 4.6 (Thinking)"
```

**Step 2: Verify focused tests fail**

Run:

```bash
uv run --with pytest pytest -q test_13_label_drive.py
```

Expected: new validation and manifest tests fail.

**Step 3: Implement minimal validation and manifest writing**

Add `VALID_CONFIDENCE`, required-field checks, and cross-field consistency
checks to `parse_response`. Add `run_manifest(...)` and write
`out/labels/<tag>/manifest.json` before invoking any model. Include tag, UTC
creation time, input path, row count, prompt filename and SHA-256, labeller
CLI/model mapping, chunk size, and concurrency.

Do not include credentials, environment values, or full prompt text in the
manifest.

**Step 4: Verify runner tests**

Run:

```bash
uv run --with pytest pytest -q test_13_label_drive.py
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add analysis/investigations/2026-07-18-hatespeech-finetune/13_label_drive.py \
  analysis/investigations/2026-07-18-hatespeech-finetune/test_13_label_drive.py
git commit -m "feat(analysis): validate Opus label runs"
```

### Task 3: Add Opus relabelling reports

**Files:**
- Create: `analysis/investigations/2026-07-18-hatespeech-finetune/21_opus_relabel.py`
- Create: `analysis/investigations/2026-07-18-hatespeech-finetune/test_21_opus_relabel.py`

**Step 1: Write failing unit tests**

Define and test pure helpers:

```python
def score_reference(df: pd.DataFrame, label_col: str, reference_col: str) -> dict: ...
def label_movement(current: pd.Series, previous: pd.Series) -> dict: ...
def flag_counts(flags: pd.Series) -> dict[str, int]: ...
```

Assert exact agreement, macro-F1, hate precision/recall/F1, a complete 3x3
movement matrix, and exploded flag counts on small fixtures. Assert duplicate
or mismatched ID sets raise `ValueError`.

**Step 2: Verify tests fail**

Run:

```bash
uv run --with pytest pytest -q test_21_opus_relabel.py
```

Expected: failure because the module does not exist.

**Step 3: Implement the reporting CLI**

Provide:

```text
make-calibration --source out/blind_check_coded_calibration.csv
score-calibration --tag <tag> --labeller claude-opus-4.6
compare-full --tag <tag> --labeller claude-opus-4.6 --baseline <parquet>
```

`make-calibration` writes a driver-compatible parquet with
`stratum=calibration`. `score-calibration` joins by string `post_id` and writes
`out/21_opus_calibration_<tag>.json`. `compare-full` reports label and flag
counts, confidence counts, label movement, changed-row counts by stratum, and
the source manifests to `out/21_opus_full_<tag>.json`.

**Step 4: Verify report tests**

Run:

```bash
uv run --with pytest pytest -q test_21_opus_relabel.py
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add analysis/investigations/2026-07-18-hatespeech-finetune/21_opus_relabel.py \
  analysis/investigations/2026-07-18-hatespeech-finetune/test_21_opus_relabel.py
git commit -m "feat(analysis): report Opus relabelling"
```

### Task 4: Run calibration and pilot checkpoints

**Files:**
- Generate: `analysis/investigations/2026-07-18-hatespeech-finetune/out/labels/opus-v4-calibration/**`
- Generate: `analysis/investigations/2026-07-18-hatespeech-finetune/out/labels/opus-v4-pilot/**`
- Generate: `analysis/investigations/2026-07-18-hatespeech-finetune/out/21_opus_calibration_opus-v4-calibration.json`

**Step 1: Build the calibration input**

Run:

```bash
uv run 21_opus_relabel.py make-calibration \
  --source out/blind_check_coded_calibration.csv
```

Expected: 120 unique rows written.

**Step 2: Run the 120-row Opus calibration**

Run:

```bash
uv run 13_label_drive.py --input opus_v4_calibration.parquet \
  --tag opus-v4-calibration --prompt-version v4 \
  --labellers claude-opus-4.6 --concurrency 2
```

Expected: five complete chunks, zero parked chunks.

**Step 3: Score and inspect calibration**

Run:

```bash
uv run 21_opus_relabel.py score-calibration \
  --tag opus-v4-calibration --labeller claude-opus-4.6
```

Inspect every mismatch involving `hate`, political aliases, quotation, or
condemnation. Fix only generalisable prompt errors; if the prompt changes,
increment the prompt revision/tag and rerun this checkpoint.

**Step 4: Run a stratified full-batch pilot**

The full source parquet is an ignored local artifact in the main checkout:

```bash
SOURCE=/Users/tom/Code/Misc/kenya-monitor-2027/analysis/investigations/2026-07-18-hatespeech-finetune/out/labels/full/batch.parquet
uv run 13_label_drive.py --input "$SOURCE" --pilot 100 \
  --tag opus-v4-pilot --prompt-version v4 \
  --labellers claude-opus-4.6 --concurrency 2
uv run 14_label_merge.py --tag opus-v4-pilot --prompt-version v4 \
  --labellers claude-opus-4.6 --blind-check 30
```

Expected: 100 labels, no structural failures, a 30-row blinded review sheet,
and no overwritten `full` artifacts.

**Step 5: Commit checkpoint reports**

```bash
git add -f analysis/investigations/2026-07-18-hatespeech-finetune/out/21_opus_calibration_opus-v4-calibration.json \
  analysis/investigations/2026-07-18-hatespeech-finetune/out/14_merge_report_opus-v4-pilot.json
git commit -m "test(analysis): validate Opus v4 pilot"
```

### Task 5: Relabel and merge all 2,440 posts

**Files:**
- Generate: `analysis/investigations/2026-07-18-hatespeech-finetune/out/labels/opus-v4-full/**`
- Generate: `analysis/investigations/2026-07-18-hatespeech-finetune/out/labels_2026_opus-v4-full.parquet`
- Generate: `analysis/investigations/2026-07-18-hatespeech-finetune/out/14_merge_report_opus-v4-full.json`
- Generate: `analysis/investigations/2026-07-18-hatespeech-finetune/out/21_opus_full_opus-v4-full.json`
- Modify: `analysis/investigations/2026-07-18-hatespeech-finetune/STATE.md`
- Modify: `analysis/investigations/2026-07-18-hatespeech-finetune/findings-plan-a.md`

**Step 1: Run the isolated full tag**

Run:

```bash
SOURCE=/Users/tom/Code/Misc/kenya-monitor-2027/analysis/investigations/2026-07-18-hatespeech-finetune/out/labels/full/batch.parquet
uv run 13_label_drive.py --input "$SOURCE" \
  --tag opus-v4-full --prompt-version v4 \
  --labellers claude-opus-4.6 --concurrency 2
```

Expected: 98 complete chunks covering exactly 2,440 unique IDs and zero
parked chunks. Resume the same command after transient failures.

**Step 2: Merge the single-labeller output**

Run:

```bash
uv run 14_label_merge.py --tag opus-v4-full --prompt-version v4 \
  --labellers claude-opus-4.6 --blind-check 100
```

Expected: 2,440 accepted single-labeller rows with the reliability caveat.

**Step 3: Compare with the prior full labels**

Run:

```bash
BASELINE=/Users/tom/Code/Misc/kenya-monitor-2027/analysis/investigations/2026-07-18-hatespeech-finetune/out/labels_2026_full_final.parquet
uv run 21_opus_relabel.py compare-full --tag opus-v4-full \
  --labeller claude-opus-4.6 --baseline "$BASELINE"
```

Expected: complete ID match and a JSON report with movement, stratum, flags,
confidence, and provenance.

**Step 4: Update investigation status**

Record the exact Opus model, prompt hash, calibration metrics, full class/flag
distribution, movement from the prior labels, and the single-labeller
limitation. Do not describe model-assisted calibration as independent.

**Step 5: Run the full test suite**

Run:

```bash
uv run --with pytest pytest -q
```

Expected: all tests pass.

**Step 6: Commit the completed relabel**

```bash
git add analysis/investigations/2026-07-18-hatespeech-finetune/STATE.md \
  analysis/investigations/2026-07-18-hatespeech-finetune/findings-plan-a.md
git add -f analysis/investigations/2026-07-18-hatespeech-finetune/out/labels/opus-v4-full \
  analysis/investigations/2026-07-18-hatespeech-finetune/out/14_merge_report_opus-v4-full.json \
  analysis/investigations/2026-07-18-hatespeech-finetune/out/21_opus_full_opus-v4-full.json
git commit -m "feat(analysis): relabel corpus with Opus v4"
```

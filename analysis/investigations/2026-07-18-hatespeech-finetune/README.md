# Kenya hate-speech classifier

3-class (neither / offensive / hate) classifier for Kenyan political tweets,
built to complement the lexicon+NLI incitement rule in `kma/incitement.py`.

**For current status and what to do next, read `STATE.md`.** This file is the
map of scripts and how to run them.

## Docs

- `STATE.md` - where the project is, the open question, the roadmap. Start here.
- `findings.md` - Plan D (DAPT + afro-xlmr-large) and round-2 results.
- `findings-plan-a.md` - the 2026 labelling effort and reliability analysis.
- `archive/` - superseded planning briefs.

## Scripts (standalone PEP 723; `uv run <script>`)

Pipeline order. Numeric prefix = rough sequence; scripts import each other by
prefix (e.g. `03_eval` calls `04_infer.predict`).

| script | does |
|---|---|
| `_common.py` | shared: label maps (0=neither/1=offensive/2=hate), device, splits |
| `00_prep.py` | clean + stratified split of `HateSpeech_Kenya.csv`; MinHash near-dup helpers |
| `01_baseline.py` | TF-IDF + logreg macro-F1 floor |
| `02_train.py` | fine-tune. Flags: `--model --extra-data(,) --extra-repeat --agreement-min --focal-gamma --no-class-weights --llrd --seed --val-split --grad-accum --no-base-train` |
| `03_eval.py` | test report, confusion png, hate-threshold sweep |
| `04_infer.py` | score a CSV; `predict()` is the pipeline seam |
| `07_siblings.py` | AfriHate Swahili intake |
| `08_export_corpus.py` | R2 -> `dapt_corpus.parquet` (tac2, needs R2 creds) |
| `09_dapt.py` | domain-adaptive MLM pretraining |
| `10_push_hf.py` | push a model dir to private HF |
| `11_hate_sweep.py` | sub-argmax hate-threshold sweep |
| `12_score_corpus.py` / `12_mine_candidates.py` | score full corpus, mine label candidates |
| `13_label_drive.py` | dual-CLI (agy + cursor) labelling, resumable |
| `14_label_merge.py` | merge labellers, kappa, adjudication queue, blind-check sheet |
| `15_adjudicate.py` | third-model blind adjudication of disagreements |
| `16_gold_split.py` | gold / challenge / train split assignment |
| `17_prep_round2.py` | Plan A labels -> round-2 train/val/gold/challenge parquets |
| `18_blind_check.py` | the human gate: `make` a sheet, `score` it |
| `modal_train.py` | run any of the above on a Modal A100 |
| `run_*.sh` | batch recipes (d = Plan D ladder, r2 = round 2) |

## GPU runs (Modal)

```sh
uv run modal run modal_train.py --cmd "python 09_dapt.py --full"          # sync
uv run modal run --detach modal_train.py --cmd "bash run_r2_batch.sh" --spawn  # detached
```

Volume `hatespeech-finetune` holds all parquets and model outputs. Local
smokes run on MPS (sample mode, sub-minute); real training goes to Modal.

## Class mapping

0 = neither, 1 = offensive, 2 = hate. Verified from annotator-count means,
**not** the Davidson paper ordering.

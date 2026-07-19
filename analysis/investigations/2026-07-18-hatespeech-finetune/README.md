# 2026-07-18 hate-speech fine-tune

Fine-tunes `cardiffnlp/twitter-xlm-roberta-base` on `HateSpeech_Kenya.csv`
(repo root, untracked) into a 3-class classifier: **neither / offensive /
hate**. Training-only investigation; `04_infer.py::predict` is the seam for
later `kma` pipeline wiring (see `kma/incitement.py` notes on the joint
lexicon+NLI rule this is meant to upgrade).

## Dataset

48,076 tweets, Davidson-style annotation (~3 annotator votes per row in the
`hate_speech` / `offensive_language` / `neither` count columns).

Class mapping - verified from annotator-count means, NOT the Davidson paper
ordering:

| Class | meaning   | rows   |
|-------|-----------|--------|
| 0     | neither   | 36,352 |
| 1     | offensive | 8,543  |
| 2     | hate      | 3,181  |

Caveats:
- Provenance unconfirmed (looks like 2013-election-era Kenya tweets):
  vocabulary drift vs the 2026 corpus - spot-check transfer before trusting.
- `USERNAME_n` placeholders are normalised to `@user` in prep.
- A few off-topic rows (e.g. Malay tweets caught by a "masai" keyword
  collision) left in as noise.

## Run (local, M4 Pro / MPS)

```sh
cd analysis/investigations/2026-07-18-hatespeech-finetune
uv run 00_prep.py          # clean + stratified 80/10/10 split
uv run 01_baseline.py      # TF-IDF + logreg macro-F1 floor
uv run 02_train.py         # sub-minute smoke run (1k rows, 1 epoch)
uv run 02_train.py --full  # real run -> out/model
uv run 03_eval.py          # test report, confusion png, threshold sweep
uv run 04_infer.py --csv some_posts.csv --column text
```

## Run (Google Colab, T4/A100)

Upload this directory plus `HateSpeech_Kenya.csv` into the same folder, then:

```
!pip install -q pandas pyarrow scikit-learn torch transformers accelerate matplotlib
!python 00_prep.py && python 01_baseline.py
!python 02_train.py --full
!python 03_eval.py
```

Device (CUDA/MPS/CPU) is autodetected; fp16 enabled on CUDA only.

## Outputs (`out/`)

- `train|val|test.parquet` - splits (seed 42, stratified)
- `01_baseline.json` - floor macro-F1
- `model/` (+ `model-sample/` from smoke) - checkpoint + `train_log.json`
- `03_metrics.json`, `03_confusion.png`, `03_errors.csv`,
  `03_hate_thresholds.csv` - precision/recall per flag threshold for triage

Primary metric: macro-F1 (11:3:1 class imbalance; weighted cross-entropy in
training).

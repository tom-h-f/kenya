# Phase 2 - Semantic / narrative layer

Status: **core done** (2026-07-07), classification + deltas remaining. Delivers
the existing README TODO.

## Done

- `kma/db.py`: `embeddings_source()`, `latest_embeddings()`.
- `kma/semantic.py`: `embed_new` (incremental, persists to R2
  `embeddings/platform=x/model=<slug>/dt=/run=.parquet`, 768d normalized),
  `search` (cosine k-NN via DuckDB `vss`), `assign_topics` (PCA-50 + HDBSCAN),
  `topic_summary` (tf-idf terms).
- Model: `paraphrase-multilingual-mpnet-base-v2`. Validated on live data: search
  handles code-mixed English/Swahili/Sheng well ("Kasongo"/"wantam" resolve to
  Ruto-opposition posts). 1,000 posts embedded so far; incremental skip verified.
- `notebooks/narratives.py` (search + topics), exports clean.

## Done - classification + deltas (encoder-only stack)

- `kma/classify.py`: `classify_new` (sentiment `cardiffnlp/twitter-xlm-roberta-
  base-sentiment` + emotion `MilaNLProc/xlm-emo-t`, persisted to R2 `labels/`,
  incremental) and `stance` (zero-shot `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`,
  target-based, live). Runs on **MPS** via chunked batches + `mps.empty_cache()`
  with a per-chunk CPU fallback (naive full-batch MPS OOMs). ~16 posts/s.
- `kma/db.py`: `labels_source`, `latest_labels`.
- `kma/deltas.py`: `slice_sentiment(dimension)` for region / lang / community.
  **Community = the EXPERIMENTAL tribe proxy** (location -> historically-dominant
  county community; `TRIBE_DISCLAIMER`, aggregate-only, never per-person).
  Validated: sentiment resolves per region; stance direction correct on coded
  terms ("wantam"/"Kasongo").
- `notebooks/narratives.py` extended: sentiment/emotion, region + community
  deltas (with disclaimer), interactive stance.
- Tokenizer deps added: `sentencepiece`, `tiktoken`, `protobuf`.

Data caveat found: X's `lang` field mislabels Sheng/Swahili (tags "tl"/"in"/
"ht"), so trust `lang` slices less than region; consider our own language ID.

## Done - topic tuning

`assign_topics` now uses **UMAP -> HDBSCAN** (BERTopic recipe) and `topic_summary`
uses **c-TF-IDF** with an English + Swahili/Sheng stopword list. On the full 18,321
embeddings: mcs=60 gives 43 coherent, distinctive topics at 44% outliers (was 89%
noise / 2 topics with PCA). Clusters are clean (e.g. `maandamano, nchi, wananchi,
amani, haki, serikali`; `wantam, kasongo, tutam`) and surfaced off-topic
promotional clusters (insurance/reinsurance, crypto/finance) worth flagging to
Phase 3. Determinism: `_embeddings_with_text` now `ORDER BY platform_post_id`
(UMAP is input-order sensitive; without the sort, runs disagreed).

## Remaining

- **embedding-atlas** viz over persisted embeddings + labels.
- Optional: temporal deltas (narrative/sentiment trends over `created_at`).

## Why

Track what narratives spread, their sentiment/emotion/stance, and how they shift
over time and across region/language. This is the substrate for characterising
what a coordinated cluster (Phase 3) is actually pushing.

## Persistence (hybrid)

Embeddings are expensive, so **persist** them; everything else computes live.

- New R2 prefix:
  `embeddings/platform=x/model=<model>/dt=YYYY-MM-DD/run=<utc-ts>.parquet`
- Columns: `platform_post_id`, `model`, `dim`, `embedding` (`list<float>`),
  `embedded_at`.
- Incremental: embed only `platform_post_id`s not already present. Add a
  `embeddings_source()` / `latest_embeddings()` helper in `kma/db.py`.

## Deliverables

- `analysis/src/kma/semantic.py`:
  - `embed_new(con, model, batch) -> int` - find un-embedded posts, encode,
    write a new Parquet run to `embeddings/`.
  - `search(con, query, k)` - k-NN over embeddings via DuckDB `vss` (HNSW).
  - `topics(con)` - cluster embeddings into narrative topics.
  - `classify(con)` - sentiment + emotion + target-stance per post.
- `analysis/notebooks/narratives.py` (marimo) - topics over time, sentiment
  trend, stance toward tracked principals, region/language slices.
- New deps via `uv`: `sentence-transformers`, `bertopic`, `hdbscan`;
  DuckDB `vss` extension.

## Model choices - MEASURE, don't guess

- **Embeddings.** Kenyan content is English + Swahili + Sheng (code-mixed).
  Candidate: `paraphrase-multilingual-mpnet-base-v2`. `DECISION:` confirm by
  nearest-neighbour sanity check on a 500-post sample before committing; try at
  least one alternative. Run locally on tac2 first; benchmark throughput.
- **Sentiment / emotion.** Multilingual classifier; verify it handles Swahili/
  Sheng, not just English. `DECISION:` pick after a small labelled spot-check.
- **Stance.** Target-based toward the principals in `targets.yaml`. Consider an
  LLM few-shot pass if encoder models underperform on code-mixed text.

## Deltas

- Time buckets on `created_at`; narrative/sentiment trends per topic.
- Region: normalise free-text author `location` to region; slice by region.
- Language: slice by `lang` (+ Phase 0 fields).
- **Tribe: flagged, not core.** Ethnicity is not in the data and inferring it
  from names/language is unreliable and sensitive. If attempted at all, treat as
  a clearly-caveated experiment using region + language proxies, never a headline
  metric. `DECISION:` confirm whether to include this at all.

## Visualise

`embedding-atlas` (from README) over the persisted embeddings + topic/sentiment
labels.

## Verify (small scale first)

Embed ~500 posts, inspect nearest neighbours and a handful of topics for face
validity before the full-corpus run and before wiring the notebook.

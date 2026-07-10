# Phase 4 - Story discovery, trusted-media triage, targeted tracking

Status: **core done**. Delivers proactive claim-level surfacing on top of the
existing embeddings/coordination stack. Adds `kma/stories.py` + a marimo notebook
+ a collector handoff, following the established phase pattern (one module + one
notebook, reading/writing R2 Parquet via DuckDB).

## Why

Specific false stories/campaigns keep appearing in the wild. Nothing in the
pipeline isolated a **discrete claim**: everything was an ad-hoc `search()` or a
broad unsupervised topic. This phase asks *which claims are circulating right now,
are they corroborated by trusted media, and where did they start* - and hands a
flagged story back to the collector so it chases the origin + spread.

**Key insight that makes corroboration cheap:** the five trusted Kenyan outlets
(`StandardKenya`, `citizentvkenya`, `NationAfrica`, `KTNNewsKE`, `ntvkenya`) are
already collected as timeline accounts and already embedded in the same 768d
space. A viral claim with **no semantically-similar trusted-outlet post** is a
strong *"unverified / likely-fabricated"* triage signal - no new ingestion needed.
Two fact-checkers (`PesaCheck`, `AfricaCheck`) are added to the collector as the
strongest truth signal; their corroboration strengthens as their timelines
backfill.

## Corroboration-gap caveat (read first)

**A corroboration gap is a triage flag, NOT proof of falsity.** Trusted outlets lag
breaking news and often tweet only headlines, so a real story can show a gap for
hours. The pipeline **always surfaces the nearest trusted post** so a human judges
the gap rather than trusting the scalar. Absence of trusted coverage != false.

Two further caveats, kept visible in code + notebook:

- **Capture is a sample, not a census:** the earliest *collected* post is not
  necessarily patient-zero, and spread (retweeters/repliers) is bounded by what the
  snowball census reached.
- **Fact-checker signal is weak** until `PesaCheck`/`AfricaCheck` timelines
  backfill.
- The `story_suspicion_index` is triage for a human, **never** an auto-label.

## Pipeline (`kma/stories.py`)

1. **`candidate_stories(con, days, tau, min_size)`** - claim-level clusters:
   connected components of the cosine >= `tau` graph over recent post embeddings
   (same primitive as `coordination.content_clusters`, joined to `latest_posts` and
   filtered to the last `days`, at a lower story-level `tau ~= 0.80` - paraphrases,
   not verbatim copypasta). Before clustering, retweets/same-author dupes AND
   low-information posts (bare links, one-liners, pure emoji: `< MIN_CONTENT_WORDS`
   real words) are dropped; after clustering, degenerate **chaining blobs** (a
   component that is both large, `> MAX_COHERENT_AUTHORS`, and dispersed, cohesion
   `< MIN_COHESION`) are rejected. Both are needed - without them, low-info posts
   single-linkage-chain thousands of unrelated claims into one 6k-author component
   that swallows small stories. Keeps components with >= `min_size` distinct authors
   (default 3).
2. **`corroboration(con, stories, days)`** - per story: max cosine of its centroid
   (renormalised mean member embedding) to any `TRUSTED_SOURCES` post that also
   **shares the story's claim vocabulary** - `>= MIN_SHARED_TERMS` salient words and,
   when the story names entities, at least one entity (proper-noun proxy, see
   `_entity_terms`). This lexical+entity gate makes the signal **claim-level, not
   topic-level**: a fabricated "Ruto motorcade crash" embeds near real accident
   coverage (~0.65) but shares no entities with it, so that coverage no longer masks
   the gap. Returns the nearest gated trusted post (handle/text/sim); none passing =>
   `corrob_sim 0.0` (maximal gap).
3. **`story_scorecard(con, stories, corrob)`** - transparent weighted
   percentile-rank index (mirrors `coordination.scorecards` / `STORY_WEIGHTS`):
   `corroboration_gap` (0.25), `amplifier_botness` (0.25, via
   `authenticity.authenticity_score`), `coordination_overlap` (0.20, share of
   authors in `latest_coordination_clusters`), `reach` (0.20, distinct authors
   carrying the claim), `source_concentration` (0.10, posts per distinct author). A
   gap alone never flags a story - amplification, coordination and reach must stack
   with it. (`burst_recency` was dropped: over the recent window every claim cluster
   is time-tight, so it carried no signal; `reach` replaced it - a wide multi-account
   push is what lifts a coordinated story like the SACCO case above 2-account noise.)
   Attaches c-TF-IDF `keywords` + top `hashtags`.
4. **`origin(con, story)`** - earliest-seen member posts + author authenticity +
   coordination-cluster membership (first-mover view; bounded by capture).
5. **`spread(con, story)`** - amplifiers (retweeters from the engagement census +
   repliers via `conversation_id`) and a post-volume timeline.
6. **`persist_stories(con, scorecard, min_index)`** - writes scored stories to
   `stories/platform=x/dt=YYYY-MM-DD/run=<utc-ts>.parquet` (mirrors
   `persist_clusters`). This is the collector handoff.

`notebooks/stories.py` (marimo) renders: the ranked candidate-story table with a
corroboration verdict + nearest trusted post, a per-story drill-down (origin,
amplifiers, volume timeline), the corroboration panel with the caveat, and a
"flag for targeted collection" button.

## Collector handoff (targeted collection)

Mirrors the coordination-cluster promotion path exactly
(`storage.clusters_view` -> `adaptive.cluster_accounts`/`promote` ->
`scheduler._adaptive_targets`):

- `collector/config/targets.yaml`: `PesaCheck` + `AfricaCheck` added to
  `x.accounts` (their timelines are collected + embedded; run a timeline/backfill
  pass once to seed history).
- `collector/src/kenya_monitor/storage.py`: `stories_view(platform)` reads the
  `stories/` prefix.
- `collector/src/kenya_monitor/adaptive.py`: `flagged_story_keywords(con,
  stories_view, min_index)` returns the keywords/hashtags of the latest
  flagged-stories run above `STORY_FLAG_MIN_INDEX` (default 0.6);
  `promote()` folds them in alongside `bursting_hashtags`, tagged
  `source="story-flag"`.
- `collector/src/kenya_monitor/scheduler.py`: passes `storage.stories_view("x")`
  into `promote()`.
- Snowball needs no change: `runner.hot_objects` already censuses the most-amplified
  objects, so once a flagged story's keywords are promoted and its posts collected,
  its viral posts get retweeter+reply census automatically (traces spread).

## Persistence

New R2 prefix: `stories/platform=x/dt=YYYY-MM-DD/run=<utc-ts>.parquet`. Columns:
`story_id`, `size`, `n_posts`, `keywords`, `hashtags`, `representative_text`,
`representative_post_id`, `member_post_ids`, `corrob_sim`, `corroboration_gap`,
component + `story_suspicion_index`, `computed_at`. Readers in `kma/db.py`:
`stories_source()` / `latest_stories()`.

## Ground-truth eval (`kma/eval.py`, `run_eval.py`)

A regression harness anchored on real disinfo that circulated on X. Each
`GroundTruthCase` (matched by claim-text `ILIKE` + confirmed author handles) is
walked through the pipeline stage-by-stage and the first drop-out is reported:
**collection** (not in corpus) -> **embedding** -> **clustering** (sub-threshold or
swallowed by a blob) -> **scoring** (below the top-`TOP_FRACTION` triage cut).
`expect="surface"` cases gate a green run; `expect="known-limitation"` cases are
tracked but not required. Run: `cd analysis && uv run python run_eval.py`.

The two seed cases and how the fixes above were derived from them:
- **`sacco-savings-borrow`** (surface): a distorted "government to borrow SACCO
  savings" claim pushed by ~19 accounts. A trusted outlet (`ntvkenya`) covered the
  topic, so its corroboration gap is legitimately low - it is surfaced by
  reach/botness/coordination, not the gap. Post-fix it ranks ~58/456 (top ~13%,
  inside the 15% cut).
- **`ruto-motorcade-crash`** (known-limitation): see below.

Every threshold in the fixes (`MIN_CONTENT_WORDS`, `MAX_COHERENT_AUTHORS`,
`MIN_COHESION`, `MIN_SHARED_TERMS`, `STORY_WEIGHTS`) was chosen by measuring this
harness against live R2, not guessed.

## Known limitation - small, no-coverage fabrications

The fabricated **Ruto motorcade crash** (~3 Jul 2026, no reputable outlet reported
it) is *not* reliably surfaced, and this is structural, not a bug:
- Only ~2 accounts carried it, so it is below `min_size` (and even at `min_size=2`
  it has no reach/botness/coordination signal to lift it in an amplification-ranked
  list).
- Its topic overlaps real accident coverage and its only entity ("Ruto") is
  ubiquitous, so entity-gated corroboration only *partially* isolates it (gap rises
  but not to maximal).

Catching this class needs a **dedicated small-cluster high-gap view** (list 2+ author
clusters ranked by claim-level gap + recency, with a stricter rare-entity gate),
separate from the amplification scorecard - deferred. The harness tracks this case so
the day a fix lands, it flips green.

## Decisions locked

- Story clustering = connected-components at `tau ~= 0.80` with a low-information
  pre-filter and a chaining-blob reject (cheap, deterministic, reuses the trusted
  `content_clusters` primitive; not UMAP/HDBSCAN).
- Corroboration = claim-level: trusted outlets' **X posts only** for v1, gated by
  shared salient terms + entities (RSS/news-site ingestion deferred).
- Trusted set = 5 media handles + fact-checkers `PesaCheck`, `AfricaCheck`.
- Scorecard weights favour amplification + reach over the corroboration gap alone
  (the gap is a noise magnet for tiny uncorroborated clusters).

## Verify (small scale first)

1. `cd analysis && uv sync`; ensure embeddings exist (incl. trusted-outlet posts).
2. `candidate_stories(con, days=14, min_size=4)` - eyeball coherent claim-level
   clusters; tune `tau` around 0.80.
3. `corroboration` on a story trusted media covered (high sim) vs a fabricated
   cluster (low sim) - sanity-check both directions; set the "no coverage"
   threshold empirically here, do not guess it.
4. `story_scorecard` - organic breaking news ranks low; a bot-amplified
   uncorroborated cluster ranks high.
5. `origin` / `spread` on one story - plausible first-seen author + retweeters.
6. `persist_stories` writes one run, then `adaptive.promote(..., dry_run=True)`
   surfaces the story's keywords as `story-flag` entries.
7. `uv run marimo edit notebooks/stories.py` end-to-end against live R2.

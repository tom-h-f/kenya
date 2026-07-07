# Phase 1 - Account authenticity / bot scoring

Status: **implemented + verified** (2026-07-07). Heuristic + anomaly, per the
"heuristic + anomaly" decision.

## Verified (6,405 live accounts)

- Known-real principals score low on `suspicion` (Ruto 0.01, Kalonzo 0.004,
  Karua 0.05, media handles ~0.15; 90th pct = 0.30).
- Top-suspicious accounts are 8-90 days old, 11-63 tweets/day, following >>
  followers - the expected bot profile.
- Nuance: `anomaly_rank` flags mega-influencers (Ruto etc.) as statistical
  outliers, so it must be read alongside `suspicion`, not alone. Documented in
  the module + notebook.

Provisional / to calibrate: heuristic `WEIGHTS` and the `tweet_rate/50`,
`age/365` scale constants are hand-set; tune against data. `duplicate_text_ratio`
only fires for prolific accounts (single-post accounts can't repeat).

## Why

Bot / sockpuppet ranking is the highest-confidence signal we can build on data
we already hold. Account age and follower/following ratio are the hardest
signals for a cheap account farm to fake, so a transparent feature score gives a
defensible triage list even without labelled training data.

## Data

`authors/` (deduped) + behaviour from `latest_posts`. No ground-truth labels, so
this is **triage/ranking, not a trained classifier** - state that in output.

## Deliverables

- `analysis/src/kma/db.py` - add `latest_authors(con, platform="*")` deduping
  `authors_source` by `platform, platform_user_id ORDER BY collected_at DESC`
  (mirror of existing `latest_posts`; the tf1 server already has this view).
- `analysis/src/kma/authenticity.py` - feature + score builders returning DuckDB
  relations:
  - `author_features(con) -> relation` - one row per author with the columns
    below.
  - `authenticity_score(con) -> relation` - features + a `suspicion` score and a
    per-feature contribution breakdown.
- `analysis/notebooks/authenticity.py` (marimo) - ranked most-suspicious
  accounts with the feature breakdown; sanity panel for known-real principals.

## Features

Profile (from `latest_authors`):
- `account_age_days` = `now() - created_at`
- `followers_following_ratio`, `log_follower_following`
- `tweet_rate` = `tweet_count / account_age_days`
- `listed_ratio` = `listed_count / greatest(followers_count, 1)`
- `verified`, `blue`
- `default_profile_image` (profile_image_url matches the X default pattern)
- `empty_bio` (`bio = ''`)
- handle features: trailing-digit count, char entropy

Behaviour (from `latest_posts`, joined on `author_id = platform_user_id`):
- posts/day in the collected window
- reply ratio, repost ratio (`is_repost`), quote ratio (`is_quote`, Phase 0)
- duplicate-text rate (share of an author's posts with identical normalised text)
- posting-burst score (max posts in any short window from `created_at`)

## Scoring

`DECISION (needs your input):` two options, not exclusive -
1. **Transparent weighted heuristic** (default). Documented weights, age +
   follower/following ratio dominant, each feature normalised to [0,1]. Fully
   explainable; no training needed.
2. **Unsupervised anomaly rank** (isolation forest over the feature vector) as a
   second lens that finds odd combinations the hand weights miss.

Recommendation: ship (1) first, add (2) as an extra column. Exact weights and
thresholds to be **measured** against the data, not guessed.

## Verify

- Eyeball top-ranked accounts against their live profiles.
- Known-real principals (`WilliamsRuto`, `RailaOdinga`, media handles) must score
  low; if they score high the features are miscalibrated.
- Notebook runs end-to-end against live R2 (direct or via tf1 quack).

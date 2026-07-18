# Data model reference

How the data is physically stored, what each column means, and how the
analysis code reads it. Pairs with the collection side
([../collection/README.md](../collection/README.md), which documents how the
data is *produced*) and the code map ([code-map.md](code-map.md), which
documents how it is *consumed*).

Ground truth for the writer schemas is
`collector/src/kenya_monitor/storage.py`; for the analysis-written prefixes it
is the module that writes them (named per prefix below). The read helpers are
all in `analysis/src/kma/db.py`.

## Storage principles

- **Immutable, append-only Parquet.** Every collection or enrichment pass
  writes one new `run=<utc-ts>.parquet` file. Nothing is ever updated in
  place; there is no database.
- **Hive partitioning.** Paths encode partition columns
  (`platform=x/dt=YYYY-MM-DD/...`) so DuckDB can prune by them.
- **Latest state is a read-time reconstruction.** A post, author, or label can
  appear in many runs (re-collected over time). "Current value" is recovered
  in SQL, never stored:

  ```sql
  QUALIFY row_number() OVER (
      PARTITION BY <entity key> ORDER BY <collected_at|labeled_at|...> DESC
  ) = 1
  ```

  Every `latest_*` helper in `db.py` is exactly this pattern. Keeping all
  snapshots is deliberate: `metrics/` and repeated `posts/` rows are how
  engagement-over-time is measured.
- **`zstd` compression, `union_by_name=true`.** Readers tolerate schema drift
  across runs (a column added later reads as NULL in older files).

## Prefixes at a glance

| Prefix | Written by | Partition key | Latest-key (dedup) | Read helper |
|---|---|---|---|---|
| `posts/` | collector | `platform`, `type`, `dt` | `platform, platform_post_id` | `posts_source`, `latest_posts` |
| `authors/` | collector | `platform`, `dt` | `platform_user_id` | `authors_source`, `latest_authors` |
| `metrics/` | collector | `platform`, `dt` | (kept, not deduped) | `metrics_source` |
| `engagements/` | collector | `platform`, `dt` | `platform, platform_post_id, platform_user_id, kind` | `engagements_source`, `latest_engagements` |
| `follows/` | collector | `platform`, `dt` | (edge rows) | `follows_source`, `latest_follows` |
| `embeddings/` | analysis (`semantic.py`) | `platform`, `model`, `dt` | `platform_post_id` (by `embedded_at`) | `embeddings_source`, `latest_embeddings` |
| `labels/` | analysis (`classify.py`) | `platform`, `dt` | `platform_post_id` (by `labeled_at`) | `labels_source`, `latest_labels` |
| `incitement/` | analysis (`incitement.py`) | `platform`, `dt` | `platform_post_id` (by `scored_at`) | `incitement_source`, `latest_incitement` |
| `coordination/` | analysis (`coordination.py`) | `platform`, `kind`, (`channel`, `method`), `dt` | per kind | `coordination_source`, `latest_coordination_edges/clusters` |
| `stories/` | analysis (`stories.py`) | `platform`, `dt` | `stable_story_id` (by `computed_at`) | `stories_source`, `latest_stories` |

`type` for `posts/` is one of `search`, `timeline`, `replies`, `hydrated`
(see the collection docs for which method writes which).

## Schemas

### posts/ (`POST_SCHEMA`)

One row per collected snapshot of a post. Re-collection produces multiple rows;
`latest_posts` keeps the newest per `platform_post_id`.

| Column | Type | Notes |
|---|---|---|
| `platform` | string | `x` today |
| `platform_post_id` | string | tweet id; the join key everywhere |
| `author_id` | string | -> `authors.platform_user_id` |
| `author_handle` | string | denormalised for convenience |
| `text` | string | post body |
| `created_at` | timestamp us, UTC | **event** time (when posted) |
| `url` | string | canonical post URL |
| `source_query` | string | keyword/handle that surfaced it |
| `lang` | string | platform-detected; unreliable for Swahili/Sheng (see quirks) |
| `in_reply_to_id` | string | parent post id (reply latency needs this) |
| `in_reply_to_user_id` | string | target account of a reply |
| `is_repost` / `repost_of_id` | bool / string | native retweet marker + source id |
| `is_quote` / `quoted_post_id` | bool / string | quote-tweet marker + source id |
| `conversation_id` | string | thread root; groups a reply tree |
| `like/reply/repost/quote/view_count` | int64 | counts *at collection time* |
| `hashtags` / `cashtags` / `mentions` / `urls` | list<string> | extracted entities |
| `source_label` | string | client ("Twitter for Android", ...) |
| `place_name` / `lat` / `lon` | string / float64 | geotag (rarely populated) |
| `has_media` / `media_count` / `media_urls` | bool / int64 / list<string> | media |
| `collected_at` | timestamp us, UTC | **collection** time; the dedup ordering key |

Two time columns matter and are different: `created_at` drives all temporal
analysis; `collected_at` only orders snapshots for latest-state.

### authors/ (`AUTHOR_SCHEMA`)

One row per author snapshot; `latest_authors` keeps newest per
`platform_user_id`.

| Column | Type | Notes |
|---|---|---|
| `platform_user_id` | string | account id |
| `handle` / `display_name` / `bio` / `location` | string | profile text; `location` feeds `deltas.py` region/community proxy |
| `followers_count` / `following_count` / `tweet_count` / `listed_count` | int64 | audience-size features (Phase 1) |
| `verified` / `blue` | bool | legacy verified vs paid blue |
| `created_at` | timestamp us, UTC | account birth; account-age and cohort features |
| `profile_image_url` | string | default-avatar heuristic |
| `collected_at` | timestamp us, UTC | dedup ordering key |

### metrics/ (`METRIC_SCHEMA`)

Repeated engagement-count snapshots for a *subset* of posts (the collector
tracks the hottest posts over time). **Not deduped** on read: every snapshot is
a point on the engagement-over-time curve. Columns: `platform_post_id`, the
five counts, `collected_at`.

### engagements/ (`ENGAGEMENT_SCHEMA`)

Snowballed retweeter/replier incidence. Columns: `platform_post_id`,
`platform_user_id`, `kind` (`retweet`), `collected_at`. **Incidence only:
there is no action timestamp** (the platform does not expose when a retweet
happened), so these rows feed only untimed coordination channels.

### follows/ (`FOLLOW_SCHEMA`)

Directed follow edges: `follower_id`, `followed_id`, `collected_at`. Produced
by the BFS follow crawl seeded from suspicious accounts, so edge coverage is
conditioned on the crawl frontier (never compare follow density to the whole
corpus; compare within the crawled set).

### embeddings/ (`semantic.py`)

`platform_post_id`, `model` (slug of the encoder), `dim` (768),
`embedding` (list<float>, L2-normalised), `embedded_at`. Model is
`paraphrase-multilingual-mpnet-base-v2` (English/Swahili/Sheng in one space).
`latest_embeddings` dedups per post by `embedded_at`; the vector column is cast
to `FLOAT[768]` for DuckDB `array_cosine_similarity`.

### labels/ (`classify.py`)

`platform_post_id`, `sentiment`, `sentiment_score`, `emotion`,
`emotion_score`, `labeled_at`. Sentiment is XLM-RoBERTa
(positive/neutral/negative); emotion is xlm-emo-t. Stance is *not* here (it is
target-specific and computed live).

### incitement/ (`incitement.py`)

`platform_post_id`, `lexicon_hits` (list<string>), `lexicon_categories`
(list<string>), `dehumanisation_score`, `violence_call_score`,
`othering_score`, `political_criticism_score`, `model`, `scored_at`. Written
to its **own prefix on purpose** (see quirks). Zero-shot NLI over the same
mDeBERTa model used for stance.

### coordination/ and stories/

Written by analysis when a run is persisted (see `persist_edges`,
`persist_clusters`, `persist_stories`). `coordination/` splits by
`kind=edges|clusters` (edges further by `channel`, `method`); `stories/` keeps
one row per member post with the scored story columns. These are the
collector-handoff prefixes: `monitor adapt` reads the latest cluster/story run
to promote targets.

## Read paths

Both return a normal DuckDB connection; every `kma` query helper takes it.

- **`connect()`** - local DuckDB with `httpfs` + an R2 secret from the repo
  `.env` (`R2_ACCESS_KEY_ID/SECRET/ACCOUNT_ID`). Reads R2 directly. This is
  what notebooks and scripts use by default.
- **`connect_quack(name="kenya")`** - attaches the remote **quack** DuckDB
  server on tf1 (`QUACK_HOST`, `QUACK_TOKEN`). Queries run server-side against
  R2, so no R2 credentials are needed locally. Exposes `posts`,
  `latest_posts`, `metrics` views. Server impl: `server/duckdb_server.py`.

The `*_source(...)` helpers return a `read_parquet(...)` glob string usable
directly in a SQL `FROM` clause; the `latest_*(...)` helpers return a deduped
relation. Prefer `latest_*` unless you specifically want every snapshot (e.g.
engagement trajectories from `metrics_source`).

## Operational quirks (learned the hard way)

These are not in the schema but will bite anyone reading the data:

- **`created_at` is `datetime64[us]`, not `[ns]`.** Integer-nanosecond
  arithmetic on it is wrong. Convert with a tz-aware subtraction, e.g.
  `(pd.to_datetime(ts, utc=True) - epoch).dt.days`, not `.astype("int64")`.
- **Consolidated embedding run file + HTTP timeout.** A one-off
  `run=kenya_purge_*` file holds ~300 MB of embeddings and can exceed DuckDB's
  default 30 s HTTP timeout. Set `http_timeout=300000` (and `http_retries=5`)
  on the connection before large embedding reads.
- **`coordination/` and `stories/` may be empty.** They are only populated
  when an analysis run explicitly persists (and past runs may have been
  purged). `latest_coordination_*` / `latest_stories` then return nothing;
  recompute live (`coordination.build_layers` + `clusters`,
  `stories.candidate_stories`) rather than assuming persisted state exists.
- **`incitement/` is separate from `labels/` by necessity.** `latest_labels`
  dedups on `platform_post_id` alone; if incitement rows were written under
  `labels/`, a newer incitement-only row would *shadow* the sentiment/emotion
  row for that post. Keep them in distinct prefixes.
- **`lang` under-reports Swahili.** In practice ~0% of posts are labelled `sw`
  while ~8% of `en`-labelled posts carry Swahili function words. Do not slice
  by `lang`; use the multilingual embeddings or a dedicated language ID.
- **`metrics/` covers only tracked posts.** Engagement-velocity work runs on
  the ~1k hottest posts the collector re-samples, not the whole corpus. Ratio
  outliers (likes vs followers) can still be computed corpus-wide from
  `latest_posts` counts.
- **Capture is a sample, not a census.** Absence of a co-action is not evidence
  of absence; the earliest *collected* post is not necessarily patient-zero.
  This caveat is quoted verbatim in the analysis modules and should ride along
  with any finding.

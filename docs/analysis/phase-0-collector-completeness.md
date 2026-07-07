# Phase 0 - Collector completeness

Status: **implemented + live-verified** (2026-07-07).

## Why

Coordination detection (Phase 3) needs near-complete capture and the structured
signals tweets carry. The collector previously sampled (`min_faves:10`, ~20
posts/keyword/window) and stored only inline `text` for hashtags/mentions/links.

## What changed

New `Post` fields (native on twscrape `Tweet`, no regex):
`hashtags`, `cashtags`, `mentions`, `urls`, `quoted_post_id`, `is_quote`,
`conversation_id`, `in_reply_to_user_id`, `source_label`, `place_name`, `lat`,
`lon`, `has_media`, `media_count`, `media_urls`.

- `collector/src/kenya_monitor/collectors/base.py` - `Post` dataclass fields.
- `collector/src/kenya_monitor/storage.py` - `POST_SCHEMA` columns (lists as
  `list<string>`, geo as `double`).
- `collector/src/kenya_monitor/collectors/x.py` - `_to_post` mapping +
  `_mentions` / `_urls` / `_media_urls` helpers.
- `collector/src/kenya_monitor/config.py` - `SEARCH_MIN_FAVES` default `0`.
- `x.py search()` - `if min_faves:` so `0`/`None` both mean no floor.

Backward compatible: reads use `union_by_name=true`, so old Parquet files get
nulls for the new columns.

## Verified

13-post live scrape of `#KenyaDecides2027`: hashtags 13/13, conversation_id
13/13, source_label 13/13, mentions 11/13, in_reply_to_user_id 11/13, media
10/13. Geo empty (expected). `min_faves:0` returned low-engagement posts.

## Remaining / to benchmark

- **Window limit.** `SEARCH_WINDOW_LIMIT` still defaults 20. With `min_faves:0`
  many more posts qualify, so 20 is now the binding cap. Benchmark run
  cost/volume on pi0, then raise via env. Do not guess the value.
- **`urls` = 0 on first sample.** Confirm on a larger, link-heavy sample that
  `tw.links` maps correctly and this was just a media-only sample.
- **Archival before retention.** `MAX_AGE_DAYS=14` drops old posts. Add a step
  (or R2 lifecycle copy) to a durable prefix so longitudinal Phase 3 analysis
  survives the 14-day window.
- **Rate-limit / IP risk.** New volume runs on pi0 (residential IP), one
  scraping account. Watch for throttling after raising limits.

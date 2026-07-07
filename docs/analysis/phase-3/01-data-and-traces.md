# 01 - Data and behavioural traces

A behavioural trace is a shared action two accounts can both perform on the same
object. Each trace type becomes one coordination channel (one multiplex layer).

## Trace -> field mapping

| Channel | Shared action object | Source field(s) | Dedup on |
|---|---|---|---|
| co-retweet | the original tweet retweeted | `repost_of_id` | `latest_posts` |
| co-reply | the tweet replied to | `in_reply_to_id` | `latest_posts` |
| text-similarity | a near-duplicate content cluster | Phase 2 `embeddings` | `latest_embeddings` |
| fast co-share | co-retweet within `delta` seconds | `repost_of_id` + `created_at` | `latest_posts` |
| co-hashtag | a hashtag (or hashtag set) | `hashtags[]` | `latest_posts` |
| co-URL | a normalised outbound URL | `urls[]` | `latest_posts` |
| co-mention | a mentioned account | `mentions[]` | `latest_posts` |
| handle/image | shared profile pattern | `handle`, `profile_image_url` | `latest_authors` |

## Data-availability matrix (critical constraint)

`hashtags` / `urls` / `mentions` only exist on posts collected AFTER the Phase 0
change; the current 18,321 posts predate it. Sequencing follows from this:

| Channel | On current 18k? | Wave |
|---|---|---|
| co-retweet | yes (`repost_of_id`) | A |
| co-reply | yes (`in_reply_to_id`) | A |
| text-similarity | yes (embeddings exist) | A |
| fast co-share | yes | A |
| handle/image | yes (`authors`) | A |
| co-hashtag | no - needs new data | B |
| co-URL | no - needs new data | B |
| co-mention | no - needs new data | B |

Wave A is fully buildable now. Wave B unlocks as the updated collector accumulates
posts with the structured fields; track coverage with
`count(*) FILTER (WHERE len(hashtags) > 0)` over `latest_posts`.

## Account universe and dedup

- Accounts = distinct `author_id` in `latest_posts` (platform x). Reuse
  `kma.db.latest_posts`, `latest_authors`, `latest_embeddings`.
- Everything keys off the deduped latest-state rows (one row per post/author),
  not raw snapshots.

## Time semantics

- Co-action timing uses the tweet's `created_at` (authoring time), not
  `collected_at` (scrape time). `delta` windows are on `created_at`.
- `created_at` is second-resolution, adequate for CooRTweet-style windows
  (default 10s; we sweep).
- **Sampling caveat.** Even at `min_faves:0`, twscrape capture is not a census;
  we see a sample of each account's activity, so absence of a co-action is not
  evidence of absence. This bounds recall, not precision, and must be stated in
  every output.
- **Retention.** `MAX_AGE_DAYS=14` drops old posts; coordination that spans more
  than the rolling window needs the Phase 0 archival step (see phase-0 spec) or
  analysis is limited to a 14-day horizon.

## Normalisation notes (Wave B)

- URLs: lowercase host, strip `utm_*`/`fbclid`/tracking params, drop trailing
  slash. Shortener expansion (t.co already expanded by twscrape `links`) - verify
  on live data; otherwise treat shortened and expanded as distinct (documented
  limitation).
- Hashtags: casefold. Consider both the co-hashtag (shared single tag) and the
  Pacheco hashtag-sequence (ordered multiset per post) variants.
- Mentions: casefold handle.

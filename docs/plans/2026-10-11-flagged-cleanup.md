# Flagged cleanup, 2026-10-11

Branches `fix/flagged-cleanup` (kenya) and `fix/site-proxy` (site), merged to master.

## 1. Template text flag

`kma.template_spam` runs inside `v2_daily.rank` (`with_template_flags`) over the members' own
non-repost posts of every kept community, read from the window's posts view. The post pairs the trace
discards are not needed: the detector works on a community's own posts, which is how its thresholds were
set. Columns added to `daily_communities`: `template_label` (null, `template`, `template_promo`),
`template_rate`, `template_promo_rate`, `template_link_rate`, `template_eligible`, `template_authors`.
Null for dropped communities. The exporter writes `CommunityRow.template` (null when the run has no
columns); the site shows "Repeated template text" with the agreed tooltip. Modal's text-trace job is
unchanged, so nothing was redeployed there. `schema_version` stays 1 (additive).

## 2. Transitive families

`kma/families.py` (hub-star, from the dossier) now backs `coordination.content_clusters`,
`stories.candidate_stories` and the adjudication investigation's `families()`. See
`docs/analysis/2026-10-11-transitive-families.md`.

## 3. Explorer loose ends

- `emergence` is nullable; the 16 infinite tags are exported and shown as "new".
- The validator no longer exempts `/tags/*`: only the tag name is data, fields inside a tag entry are
  checked. The exemption in `assert_explorer_safe` is gone.
- Profiles: `read_profiles` reads the window-dated `authors/` objects, then walks older ones newest first
  in batches of 100 for any member still without one, for at most 120 s. Measured on `cloud`: 10,560
  members, 0 without a profile, 0 older objects read.
- V1.5: topics (`explorer_topics`), coverage by stage, five parquet slices (`explorer_slices`), DuckDB-WASM
  and the parquet extension hosted in the bucket (`explorer_vendor`, pins in `explorer/vendor.json`), site
  `/topics`, `/topics/[tid]`, `/search`, `/query`, the `/time` coverage panel. Measurements are in
  `deploy/cloud/README.md`.

What the topics view is not: terms are c-TF-IDF over a sample of the window's posts per topic, so they
describe this window, not the fit. 68% of window posts are noise (-1) and are counted, not listed. Every
window post is kNN-placed because the fit predates the window; kNN placement is approximate (held-out topic
agreement 0.8017) and the page says so. A refit is the proper fix and is not done here.

## 4. Incitement duplicates

1,000 posts, as stated, from two causes; readers take the latest row per post, so nothing double counts.
`db.pending_posts` now raises on a real read failure instead of treating it as "nothing scored", and
`incitement.score_new` re-checks the scored prefix just before writing. Details and numbers:
`analysis/tests/test_incitement_duplicates.py` and the commit messages.

## 5. tf1 incitement pass

`ENRICH_INCITEMENT=0` turns it off. Install procedure for Tom: `docs/plans/2026-10-11-tf1-incitement-disable.md`.
Not executed on tf1.

## 6. Small items

- 6a `verify.sh`: no hang reproduced; the slow step was `preflight`'s `r2 read` (3 min 38 s), now 1.7 s.
- 6b pi0 wrappers: gauge keeps the last non-idle value and gains `last_run_assigned_posts`; both wrappers log the
  resolved git sha. Install procedure for Tom in `deploy/pi0/README.md`. Nothing written to pi0.
- 6c `middleware.ts` became `proxy.ts`. It needed `@opennextjs/cloudflare` 1.20.10; 1.19 refused a
  Node-runtime proxy (measured).
- 6d vendored files carry the version in the name and are served `immutable`.
- 6e CSP on pages: a nonce is not possible (static pages), so a static policy with `'unsafe-inline'` for scripts and
  styles, no remote origin, and `wasm-unsafe-eval` plus `worker-src blob:` on `/query` only.
- 6f `bun run lint` ignores `.open-next`.

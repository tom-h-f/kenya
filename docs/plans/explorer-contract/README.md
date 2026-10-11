# Explorer contract, schema_version 1

`schema.json` (JSON Schema 2020-12) is the source of truth. `types.ts` is generated
from it (`bun install && bun gen-types.mjs`). Producer (`kma.explorer_export`) and app build against
these two files. Any change bumps `schema_version` and both sides in one PR.

Where it lives: bucket `kenya-monitor-dashboard`, prefix `explorer/v1/`. Layout in
`$defs.Files`. `manifest.json` is the only mutable object. Everything under
`runs/{run_id}/` is written once.

## Rules the data must satisfy (tests in stream E enforce them)

1. Every file has `stamp` (run, window, computed_at, parameters). No file without it.
2. A missing source is listed in `manifest.unavailable` and its view renders
   "not available". No placeholder numbers, no zero fills.
3. No per-account hate, incitement, toxicity, deletion, suspension or "gone" field,
   anywhere, including parquet slices. Aggregate-by-topic flags only if Tom approves
   (`TopicDetail.flags`), with scored denominator and scope.
4. No prevalence rate. Counts are rows or posts in the collected corpus, with the
   denominator beside them. No raw toxicity series.
5. Counts of what the detector surfaced carry `stamp.params`; the app says "surfaced
   under these parameters".
6. Dropped communities (below the relevance floor) are counts only: no members,
   graph, evidence or account cards.
7. Labels are neutral: `Community 07`, never a characterising name.
8. No verdict words in any string the exporter writes: bot, troll, inauthentic,
   fake, sockpuppet, astroturf, "coordinated account". Linter in stream E.
9. Ids are strings. Handles may change: `id` is the key, `handle` is display.
10. An account on the denylist (`explorer/denylist.txt` in the repo, one id per line)
    is absent from every output including evidence posts and slices.

## Edge rule

A fused edge exists if the accounts are linked in any trace. The default drawn set is
`shared >= 2 or a non-co_retweet trace`; single-shared-retweet edges go to
`c{cid}.thin.json` and are opt-in. Measured on the 21-day run: 77% of top-5-per-node
backbone edges rest on exactly one shared retweet, so the default hides them.

## Fixtures

`fixtures/` is built by stream E from synthetic ids (clearly fake, e.g. 9000000000000000001+)
and synthetic handles. Never from live data. Both repos validate against `schema.json`.

## Tooling (stream E)

- `fixtures/` : full synthetic bundle in bucket layout (`fixtures/explorer/v1/...`), two runs,
  3 kept communities of 60, 30 and 12, one dropped, `time_coverage` unavailable. Regenerate
  with `cd analysis && uv run python tests/explorer/make_fixtures.py`. Output is byte-stable.
- `validate_bundle.py ROOT [--denylist FILE]` : schema plus cross-file checks (manifest hashes,
  shard placement, edge indices, dropped communities, unavailable views, forbidden field names
  including parquet columns). The producer's `assert_explorer_safe` should call it.
- `lint_language.py PATH... [--allowlist lint-allowlist.json] [--include-user-content]` : fails
  on verdict wording. Allowlist entries are exact phrases per file glob, each with a reason.
- `copy/site-copy.json` : caveat, edge phrases, methodology language rule.
- Tests: `cd analysis && uv run pytest tests/explorer` and `cd docs/plans/explorer-contract && bun test`
  (independent ajv validation of every fixture).
- Hosting: DuckDB-WASM and its parquet extension are served from the bucket at `explorer/v1/vendor/{file}` (`$defs.Files.vendor`), under versioned names, uploaded by `kma.explorer_vendor`.

## Changes inside schema_version 1

All additive or loosening, so `schema_version` stays 1 and older fixtures and runs still validate, with one removal noted below.

- 2026-10-11: `Manifest.corrections_url` removed (and the `corrections` block of `copy/site-copy.json`). No
  bump: the manifest is rewritten every run and consumers read named fields only (the site ignores unknown
  manifest keys), so an old manifest that still carries it is harmless. Validation against this schema is for
  fresh bundles only.
- 2026-10-11: `CommunityRow.template` (`TemplateText`, nullable): advisory repeated-template-text flag per
  kept community, from `kma.template_spam` over the members' own posts. Null when the run did not compute it.
  Dropped communities carry none.
- 2026-10-11: `HashtagIndex.emerging[].emergence` is nullable: null means the prior rate is zero (a tag
  not seen before), shown as "new". Before, such tags were left out.
- 2026-10-11: topics. `TopicIndex.topics[]` gains `name`, `fit_posts_in_window`, `knn_posts_in_window`;
  `fit` gains `knn_posts`, `window_posts`, `window_noise_posts`, `listed_topics`; `TopicDetail` gains `name`,
  `posts_in_window`, `fit_posts_in_window`, `knn_posts_in_window`, `daily`. `communities[].community_posts` is
  the posts by that community's members in the window that carry any topic row, so a topic share is a share of
  the topic-assigned subset.
- 2026-10-11: `TimeCoverage.basis` and the stage `topics`.
- 2026-10-11: the forbidden-field walk no longer exempts every key under `/tags/`. Only the tag name (the key
  directly under `tags` in a hashtag shard) is data; keys inside a tag entry are fields and are checked.
- 2026-10-11: `name` joins the user-content keys of the language lint (a topic name is its top terms).
- Vendor files: `explorer/vendor.json` pins them, `python -m kma.explorer_vendor` uploads them.

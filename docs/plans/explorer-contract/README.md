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
- `copy/site-copy.json` : caveat, edge phrases, methodology language rule, corrections config.
  `corrections.contact` is a marked placeholder; the site must refuse to deploy while
  `contact_is_placeholder` is true.
- Tests: `cd analysis && uv run pytest tests/explorer` and `cd docs/plans/explorer-contract && bun test`
  (independent ajv validation of every fixture).
- Hosting: DuckDB-WASM is served from the bucket at `explorer/v1/vendor/{file}` (`$defs.Files.vendor`).

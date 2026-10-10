# Explorer contract, schema_version 1

`schema.json` (JSON Schema 2020-12) is the source of truth. `types.ts` is generated
from it (`bun gen-types.mjs`). Producer (`kma.explorer_export`) and app build against
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

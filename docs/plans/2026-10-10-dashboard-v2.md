# Public dashboard on v2 data

Decided 2026-10-10. Supersedes `2026-09-14-dashboard.md` (v1 coordination
producer, Access-gated). Branch `feat/dashboard-v2`; the site is the private repo
`tom-h-f/kenya-tomfr-net` (Next 16, OpenNext, Worker `kenya-ui`).

## Rules

- **What may appear** is only what `docs/analysis/2026-09-13-publishable-statistics.md`
  permits. The producer can emit nothing else: cards are built from a fixed set
  of functions, and `assert_public` rejects account keys, `@handles`, profile
  URLs and id-sized numbers before any write.
- **Persisted output only.** The producer reads what the daily and the
  collector wrote. It rebuilds nothing, loads no model, contacts no X endpoint.
- **A missing source is "not available".** Fixed reason string, no number.
  Exception text is never published (it can carry object keys). If no live card
  can be read, the run refuses to publish rather than overwrite a good page.
- **Two blobs, never one.** `public/summary.json` here. Account-level views are
  a separate Worker, separate bucket, behind Access, later. Nothing account
  level is produced by this change.
- **The Worker never binds `kenya-monitor-2027`.** An R2 binding cannot be
  prefix-scoped.

## What the page shows

| Card | Kind | Source | Denominator / definition |
|---|---|---|---|
| Method reproduction | recorded | `recorded.yaml: benchmark_reproduction` | Macro-F1 on 6 published IO benchmark datasets, ours against the paper. Iran also shown against the reference code. Headline (datasets within 0.2) is computed from the rows |
| Kenya relevance filter | recorded | `relevance_error` | 88 human-labelled posts, drawn fresh, blind, corpus-weighted. Precision and recall for the keyword gate and the learned classifier |
| What the detector surfaced | recorded | `adjudication` | 17 communities per method on frozen snapshot `2026-09-05-promotion-off__emb20260912`, 0.5 relevance floor, read blind. Influence operations found: 0. Live run contributes only "listed anything: yes/no" |
| Where the method failed | recorded | `negative_results` | three items, each with figure, basis and source: self-amplification discriminator refuted, text-similarity 0.85 cut, composition drift |
| Daily detector run | live | `coord2/kind=daily_runs` | newest run: id, window, age, stale flag (36 h), parameters, stage seconds, run days over 14 d and missing days |
| Collection volume | live | `posts/` listing + parquet footers | rows written per UTC `dt` day, all types, by partition scope; not distinct posts, not Kenyan posts; gap days from the full prefix |
| Enrichment lag | live | `posts/ embeddings/ relevance/ hatespeech/ incitement/` | newest `dt` per stage, lag vs posts, rows in the last 7 closed days; throughput, not coverage |
| Not shown | static | producer constant | five withheld items with the reason for each |

## Removed, and why

| Removed from v1 | Reason |
|---|---|
| `kind=clusters`, `edges`, `scorecards`, `verdicts` reads | v1 stopped producing them 2026-10-10; the page would serve a frozen run |
| Cluster network, node/edge counts, cluster cards, names | account-level structure; per-account claims are not publishable |
| Corroborated vs candidate counts, share series, Kenya-share-by-tier | counts of coordinated accounts; corroboration copy is false now |
| Raw toxicity series, heatmap, coded series | composition drift: the raw series moved with the collector's target list |
| `--loop`, `compose` `dashboard` profile | replaced by a systemd oneshot on `cloud` |
| Site `/coordination`, `/toxicity` pages, Plot and d3 deps, `jose`, Access vars, checked-in `summary.json` fixture | nothing left to draw; no ops route ships; a fixture means sample numbers can render in production |

## Left out on purpose (one judgement call each)

- **The depth result** (0 of 390 deepened accounts kept a top-500 place against
  22.7% of the holdout). It is a negative result in kind, but the 2026-09-13
  document names three and does not name this one. Add it by appending an item
  to `negative_results` in `recorded.yaml`; `kma.figures.ranking_stability`
  already computes it live.
- **"17 communities, 328 accounts"** as a pair. The document lists it as
  publishable and also lists "counts of coordinated accounts" as not. Resolved
  toward the stricter reading: the 17 appears only as the sample size of a
  blind read, the 328 does not appear.
- **Self-amplification d = +0.78** has no group sizes in any project document.
  The card says so rather than inventing them.

## Refresh

- Producer: `python -m kma.dashboard`, a systemd oneshot on host `cloud`.
- **Trigger: `OnSuccess=` on the daily**, via a drop-in
  (`kma-v2-daily.service.d/dashboard.conf`). Justification: the daily's persisted
  tables are the input, so chaining on success removes a timing guess (a fixed
  04:30 would race a run that took 40 minutes on a 4 vCPU box). A drop-in rather
  than `ExecStartPost=` because `ExecStartPost` failure would fail the daily and
  stay in its restart policy, and the daily's gauge must not depend on a
  publish step.
- **Plus a 12:00 UTC safety run.** If the daily fails, nothing chains, and the
  page would silently age. The safety run republishes with the daily card
  showing the age and `stale: true`.
- Cost, measured on a Mac: 92 s, 344 MB. Footers over the `posts/` prefix
  dominate (8,142 objects). Not measured on `cloud`.
- Writes `public/summary.json` and `public/archive/dt=.../run=....json`.

## How the JSON reaches the Worker, and the credentials

Chosen: **dedicated bucket `kenya-monitor-dashboard`, bound to the Worker.**

| Option | Credential that has to exist | Verdict |
|---|---|---|
| Bundled static asset, redeployed by wrangler after each run | a Cloudflare API token with Workers Scripts edit on the account, on `cloud`, plus bun, OpenNext build and the site repo on the host | largest surface: a token that can replace the Worker |
| **Dedicated bucket + binding** | an R2 S3 token scoped to one bucket, on `cloud`; the Worker holds no secret (a binding) | smallest: the token can write one bucket of public blobs and nothing else |

Credentials, who creates them:

| # | Credential | Created by | Lives on | Scope |
|---|---|---|---|---|
| 1 | R2 API token, Object Read & Write, bucket `kenya-monitor-dashboard` only | Tom, Cloudflare dashboard (R2 > Manage API tokens; wrangler cannot mint S3 keys) | `cloud:/etc/kma/dashboard.env` (0640 root:tom) as `R2_DASHBOARD_ACCESS_KEY_ID` / `R2_DASHBOARD_SECRET_ACCESS_KEY` | one bucket |
| 2 | Existing data-bucket read key | exists | `cloud:/etc/kma/v2-daily.env` | unchanged |
| 3 | Cloudflare login for `wrangler deploy` | Tom | Tom's machine only | deploy time, not runtime |

The producer refuses to publish into `kenya-monitor-2027` and does not fall
back to the data key if the dashboard key is missing.

## Go-live

Worker `kenya-ui`, `workers_dev: false`, custom domain route
`kenya.tomfr.net` (creates the DNS record in zone `tomfr.net`). Steps are in the
site repo README. The bucket already exists. No Access app is needed for the
public page; Access is for the later ops Worker.

## Open

- Whether the 12:00 UTC safety run is wanted or the stale flag from the next
  daily is enough.
- `STALE_AFTER_HOURS = 36` is a choice, not a measurement.
- Hosted-box timing of the producer.
- Whether `kma.dossier` and the parked `--source v1` leads path still need
  `kind=scorecards`; unrelated to this page.

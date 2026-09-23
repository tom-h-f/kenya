# Leads: from a daily listing to a phone

Workstream B of `docs/plans/2026-09-22-detection-readiness.md`. This is the
contract between the three pieces that meet here - the daily v2 listing
(workstream A), the canary (workstream C), and the leads pass and poller built
on `feat/leads-alerts`.

```
v2 daily (Modal, A) --> coord2/kind=communities --+
trend discovery (pi0) --> trend_candidates/ ------+--> leads pass (Modal)
canary (Modal, C) --> injected listing rows ------+      |  diff, queue <= 10, dossiers, reader
                  --> leads/kind=canary_expected         v
                                                  leads/kind=verdicts
                                                         |
                                           kma-leads-notify (tf1) --> ntfy kenya-leads
```

## What A writes: the community listing

`coord2/platform=x/kind=communities/dt=<day>/run=<run>.parquet`, one row per
account in every community of four or more, the shape `03_communities.py`
already produces:

| column | type | required | meaning |
|---|---|---|---|
| `user_id` | string | yes | account id |
| `community` | int | yes | Leiden label, **valid within this run only** |
| `community_eigenvalue` | float | no | orders the queue when present |
| `kenya_share` | float | no | per account; community mean below 0.2 is not queued |
| `min_entities`, `text_threshold`, `text_overlap`, `window_days` | | no | the run's parameters; runs are diffed only against runs with the same values |

All members, not only the top 25 listed per community: the diff matches
communities by membership, and a truncated membership reads as churn.

## What the leads pass decides

`kma.leads` matches each current community to its highest-Jaccard predecessor.
From the measurement in
`analysis/investigations/2026-09-22-leads-thresholds/01_jitter.py` (numbers in the `kma.leads` docstring):

- **new**: best Jaccard < 0.2
- **changed**: Jaccard < 0.5, or at least 5 accounts joined and they are at
  least half the predecessor's size
- **same**: everything else; not judged again

A trend tag is new if it was not a candidate in the previous trend run.

At most 10 candidates are judged per pass, 3 slots held for trend tags;
everything not judged is written to `kind=queue` with `queued=false`. A run
with no comparable predecessor is a **baseline** and judges nothing; a run
where more than 60% of communities are new is a **reset** and judges no
communities.

`worth_a_look` is set for any `influence_operation`, and for a **new**
Kenya-relevant `political_campaign` or `unclear` call.

## What C writes: the canary

1. Canary accounts carry ids of the form `canary:<canary_id>:<n>`, and the
   injected community reaches the listing like any other. The leads pass
   recognises it by that prefix, gives the verdict row `canary_id`, and judges
   it outside the 10-candidate budget - it never displaces a real lead, and a
   baseline or reset day still judges it.
2. At injection, C writes one row to
   `leads/platform=x/kind=canary_expected/dt=<day>/canary=<canary_id>.parquet`:

   | column | type | meaning |
   |---|---|---|
   | `canary_id` | string | unique per injection, e.g. `2026-09-28` |
   | `injected_at` | timestamptz | |
   | `deadline` | timestamptz | when its absence becomes an alert |

   Set the deadline after the next leads pass could have judged it plus the
   poller's interval - with the daily pass at 06:30 UTC, injection before that
   pass plus 36 hours is safe.

3. The poller sends every verdict with a `canary_id` whatever the reader said,
   titled `[CANARY] <canary_id>: ...`, and sends `[CANARY] MISSING <canary_id>`
   at high priority once the deadline passes with no verdict for it. The
   `[CANARY]` prefix is never rewritten.

Canary accounts have no posts in R2 - the canary never touches the archive -
so their dossier is empty and the reader will usually say `unclear`. That is
expected: the canary proves the chain delivers, not that the reader is right.

## What lands in R2

Under `leads/platform=x/`, all private (dossiers carry handles and text):

- `kind=queue/dt=/run=<run>.parquet` - every candidate, queued or skipped
- `kind=dossiers/dt=/run=<run>/<candidate>.parquet` - packet JSON and prompt
- `kind=verdicts/dt=/run=<run>.parquet` - judged candidates; `mode` is
  `production` or `validation`, and the poller reads only `production`

## Going live

1. Modal: the `kenya-r2` and `anthropic` secrets already exist. Nothing new.
2. `modal deploy analysis/modal_leads.py` - schedules the daily pass at
   06:30 UTC. Move the cron if A's daily listing lands later.
3. tf1 ntfy: create a token that can write topic `kenya-leads`, and subscribe
   the phone to that topic.
4. tf1: add `NTFY_TOKEN=<token>` to `~/kenya-monitor-2027/.env`, then
   `docker compose --profile leads up -d --build leads-notify` in
   `~/kenya-monitor-2027/analysis`.
5. Before step 4, `docker compose --profile leads run --rm leads-notify --dry-run`
   shows what the first pass would send.

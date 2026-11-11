# Leads: from a daily listing to a verdict

Workstream B of `docs/plans/2026-09-22-detection-readiness.md`. This is the
contract between the pieces that meet here - the daily v2 listing (workstream
A), the canary (workstream C), and the leads pass built on `feat/leads-alerts`.

```
v2 daily (cloud, 03:30 UTC) --> coord2/kind=daily_members + daily_communities
        |  OnSuccess=
        v
kma-leads --rule (cloud) --> leads/kind=verdicts (flagged groups, private)
        |                --> kma_leads.prom --> Grafana Cloud rules (iot repo) --> IRM
kma-canary (cloud, Mon 06:00) --> leads/kind=canary_checks, kma_canary.prom
```

Production judges with a rule, not a model: `kma.leads_rule`, run by
`kma-leads --rule` on `cloud` after every successful daily pass. No Anthropic
key is used anywhere. The model reader (`--judge`, `modal_leads.py`) is kept
for hand use and is not scheduled. The rest of this document is the contract
the rule pass and the original model pass share; where they differ it says so.

## The rule

A diffed community is flagged when it passed the Kenya gate (`kept`) and
either carries a `template_label` or takes at least half of its internal edge
trace labels from the non-retweet traces (text similarity, hashtag sequence,
co-URL, fast retweet). A flag alerts when the group is new, or its matched
predecessor was not flagged, and no group alerted in the last 14 days shares
half its members. Every new or changed community is checked: the budget of
ten was the model's cost bound and does not apply.

Alerts split by size: 8 or more accounts push to a phone
(`CloudKenyaCampaignFlagged`), 4-7 land in IRM as info
(`CloudKenyaSmallGroupFlagged`). Over the 20 daily runs to 2026-11-11 the rule
gave 23 alerts, 6 of them 8+ accounts, on 5 of the 20 days. The 0.5 share and
the 8-account line were chosen on that backtest; whether flagged groups are
campaigns has not been labelled. Read an alert with
`uv run kma-leads alerts --days 2` on `cloud`: handles and sample posts for
each alerted group.

The weekly canary (`kma.canary_check`) plants a 20-account group shaped on a
real operation, in memory, reruns the daily's rank on the same window with
nothing persisted, and checks the planted accounts land together in one
community the rule flags. It does not use the daily's `--canary`, which would
put the planted group into the public counts.

## What A writes: the community membership

`coord2/platform=x/kind=daily_members/dt=<day>/run=<run>.parquet`, one row per
member of every community of four or more (`kma.v2_daily`; `leads_run.py`
reads this kind, not the `daily_accounts` listing, which carries only the
accounts that made the triage budget). The contract:

| column | type | required | meaning |
|---|---|---|---|
| `user_id` | string | yes | account id |
| `community` | int | yes | Leiden label, **valid within this run only** |
| `community_eigenvalue` | float | no | orders the queue when present |
| `kenya_share` | float | no | per account; community mean below 0.2 is not queued |
| `min_entities`, `text_threshold`, `text_overlap`, `window_days` | | no | the run's parameters; runs are diffed only against runs with the same values |

All members, not only the top 25 listed per community: the diff matches
communities by membership, and a truncated membership reads as churn.

`v2_daily` writes `user_id`, `community`, `within_score`, `centrality` and the
run tags (`run_id`, `window_id`, `window_end`, `window_days`). The optional
columns above are used when present and are absent from it today, so the Kenya
share floor does not act on a daily run and runs are compared on `window_days`
alone.

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
   consumer's interval - with the daily pass at 06:30 UTC, injection before
   that pass plus 36 hours is safe.

3. An alerting consumer should send every verdict with a `canary_id` whatever
   the reader said, titled `[CANARY] <canary_id>: ...` (`canary.alert_title`),
   and `[CANARY] MISSING <canary_id>` once the deadline passes with no verdict
   for it. The `[CANARY]` prefix is never rewritten.

Canary accounts have no posts in R2 - the canary never touches the archive -
so their dossier is empty and the reader will usually say `unclear`. That is
expected: the canary proves the chain delivers, not that the reader is right.

## What lands in R2

Under `leads/platform=x/`, all private (dossiers carry handles and text):

- `kind=queue/dt=/run=<run>.parquet` - every candidate, queued or skipped
- `kind=dossiers/dt=/run=<run>/<candidate>.parquet` - packet JSON and prompt
- `kind=verdicts/dt=/run=<run>.parquet` - judged candidates; `mode` is
  `production` or `validation`, and an alerting consumer reads only
  `production`

## Going live

The rule pass and canary are systemd units on `cloud`; install steps are in
`deploy/cloud/README.md` (Leads rule and canary). Do not deploy
`modal_leads.py`: its schedule calls the model reader every day.

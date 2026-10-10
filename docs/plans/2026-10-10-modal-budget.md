# Modal budget: about $30 a month, and a way to see it

Written 2026-10-10. Tom's rule: what is scheduled and fits stays on Modal under
about $30/month; anything big or one-off goes to mike-pc or the `cloud` host.
September came in at $44.07 because of one-off GPU jobs, not the schedule.

## Policy

| Work | Where | Why |
|---|---|---|
| Relevance scoring, daily (`kma-relevance.score_pending`) | Modal, A10G, `Period(days=1)` | Fits: measured $0.12-0.27/run, about $4-5/month. |
| Daily v2 pass (`modal_v2_daily.py`) | host `cloud`, CPU | Moving off Modal. The file still carries `Cron("30 3 * * *")`: do not deploy it. |
| One-off or large GPU (backfills, gate runs, text-sim over a big window, training) | mike-pc, RTX 3070 8 GB | Free. Size to a peak under 7 GiB; see `~/.claude/rules/mike-pc.md` (wake, `awake on <duration>`, tmux). |
| One-off CPU (replay sweeps, ad hoc coord2) | `cloud` (4 vCPU, 15 GB) | `modal_coord2.rank` and `rank_window` ask for 32 GB; whether a real window fits in 15 GB needs to be measured. |
| Anything that needs more than mike-pc or `cloud` can give | Modal, by hand | Tell Tom the estimate first. Never behind a schedule. |

A schedule is only acceptable on Modal if its per-run cost is measured and the
monthly product is a small fraction of the budget. Check `modal app list` and
the CLI below after deploying anything.

## Measured spend

Modal billing API, daily resolution, complete UTC days.

| Month | Total | By app |
|---|---|---|
| 2026-08 | $2.78 | hatespeech-backfill $2.73, bench-nli $0.06 |
| 2026-09 | $44.07 | hatespeech-backfill $20.89, iohunter-gate $15.53, relevance $3.59, replay $2.84, textsim $0.86, usv-gym-sweep $0.29, coord2 $0.06 |
| 2026-10 (01-09) | $1.46 | relevance $1.46 |

Two one-off jobs are $36.42 of September's $44.07: the A100 hate-speech
backfill (09-03 alone $12.42) and the A10G iohunter-gate (three runs of
$3.67-5.58 on 09-06 and 09-07). Neither was scheduled.

Relevance, per run (the only scheduled app):

- Daily runs 10-01 to 10-09: $0.119-0.271, mean $0.163; 27-39k new posts each.
  Execution 249-552 s for 10-03 to 10-09
  (`modal function calls kma-relevance/score_pending`).
- Drains: 2026-09-13, 1,312,692 posts for $2.19; 2026-09-30, 706,255 posts for
  $1.15. Both about $1.65 per million posts, all-in.
- At that rate the daily 27-39k posts are $0.04-0.06 of GPU; the rest of each
  run is start-up, model load and the unwindowed `pending_posts` scan. Which of
  those dominates needs to be measured before anyone optimises it.

## Checking spend

    cd analysis
    uv run --with modal python -m kma.modal_spend                    # this month, $30
    uv run --with modal python -m kma.modal_spend --month 2026-09 --budget 25

Prints month-to-date, trailing 7 days and projected month end per app, then
the verdict. Budget also from `KMA_MODAL_BUDGET`. Exit 0 within budget, 1
projected over, 2 no data (an empty report is said, not read as $0).

- The API reports complete UTC days only. Today's spend, including a run in
  progress, is invisible until the day ends.
- Projection is month-to-date plus the trailing 7-day rate over the remaining
  days, today included. A one-off run in the window inflates it for a week.
- It sums every app in the workspace, not only `kma-*` (`usv-gym-sweep` shows).

Reading it as of 2026-10-10: $1.46 to date, $0.153/day, projected $4.84 (16%).

## Schedule audit

Live on Modal (`modal app info`): one.

| App | Schedule | Per run | Per month |
|---|---|---|---|
| `kma-relevance.score_pending` | every 1 day, fires 14:45 UTC | $0.12-0.27 | $3.6-8.1 at 30 runs, about $4.9 at the mean |

`kma-replay`, `kma-coord2`, `kma-textsim`, `kma-hatespeech-backfill` are
deployed with no schedule: they cost only when someone calls them.

In the repo with a schedule and not deployed (absent from `modal app list`, no
billing rows since 2026-08-01):

| App | Schedule | Note |
|---|---|---|
| `kma-v2-daily.run_daily` | `Cron("30 3 * * *")`, 20 h timeout | Calls relevance, an A10G text trace and an 8 CPU / 32 GB rank. Never ran on Modal, so no per-run cost exists. Largest billed rows for the pieces: textsim $0.04-0.15, coord2 rank $0.07, on snapshots rather than a 90-day window. The cost of a real daily run needs to be measured. |
| `kma-leads.daily` | `Cron("30 6 * * *")` | Modal side is CPU only; the cost is the Anthropic key, not Modal. |
| `kma-canary.weekly` | `Cron("0 5 * * 1")` | 2 CPU, 8 GB, 30 min cap. Small. |

What could silently exceed the budget:

- Deploying `modal_v2_daily.py` to Modal. It has no per-run cost on record and
  its stages scale with the window. Keep it on `cloud`.
- A schedule can go quiet without an error: `kma-relevance` billed nothing from
  09-15 to 09-29, which `modal_v2_daily.py` attributes to the workspace spend
  limit (not verified here). A flat spend line for a week is a fault, not
  savings.
- A relevance backlog makes one run long, not unbounded: `pending_posts` is
  unwindowed and `limit=0`, so a run takes everything pending, newest first, in
  100k chunks, until the 4 h timeout. The 09-30 drain took about 1 h of A10G
  for 706k posts ($1.15).

## Relevance gap

`score_pending` scores every post with text that has no score under the current
model, not the last day: `db.pending_posts` anti-joins the full `posts/` view
against `relevance/model=<MODEL>/`, with no `dt` window by design (its
docstring: a window would strand history). The schedule stopping 09-15 left a
backlog, and the next run drains all of it.

Measured 2026-10-10 14:30 UTC from R2, reads only: posts with `dt` from
2026-09-14 to 2026-10-10 with non-empty text, anti-joined on `platform_post_id`
to the 2,453,427 ids scored under `kenya-relevance-afroxlmr-2026-09-13`.

| `dt` | Unscored |
|---|---|
| 2026-09-14 to 2026-10-08 | 0 |
| 2026-10-09 | 7,658 (collected after that day's 14:45 run) |
| 2026-10-10 | 20,330 (collected since) |
| Distinct ids | 26,282 |

So there is no standing gap. The 09-30 run (706,255 posts, $1.15) closed the
09-14 to 09-30 backlog, and every run since has been current. The 26,282 are
the ordinary one-day lag and clear on the 14:45 UTC run. The 09-10 to 09-13
partitions are also at 0. Posts older than 09-10 were not re-measured here; the
09-13 full pass covered 1,312,692 of 1,312,692.

Consequence, from the `modal_v2_daily.py` docstring (measured 2026-09-22):
reports written in the 09-15 to 09-30 window saw only 1,523 of 9,989 listed
posts scored and came out `relevance_scored=False`. The scores exist now;
whether those outputs are re-run is Tom's call, not a cost question.

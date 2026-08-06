# Verifying the conversation-arm banding

Deployed 2026-08-06. This document is the check to run in a few days, written
before the result is known so the bar cannot move afterwards.

Background and the measurements that motivated the change:
[census-tuning.md §11](census-tuning.md).

## What changed

The snowball conversation arm was selected by `ORDER BY max(reply_count) DESC
LIMIT 60`, unbanded - deliberately the biggest threads, which is exactly the
population `HUB_CAP_MAX` deletes on the analysis side. It is now banded to
`band_min..band_max` (3..100), TTL-aware at selection time via
`census.threaded_expr`, and widened to 250 to match the retweeted arm.

## The claim being tested

co_reply was starved of *pairable* supply, not of volume. Accounts must share
2+ reply parents to be testable, and the arm was spending its budget on threads
whose replies were thrown away.

If that is right, in order:

1. co_reply `pairable` rises (more accounts sharing 2+ in-band parents)
2. tested pairs per pairable account rises from 0.57 toward co_retweet's ~24
3. `edges_bonferroni` on co_reply rises from ~30-70
4. bridge accounts rise from 1-2 into the tens
5. only then does `n_corroborated_clusters` become a real measurement

**Steps 1-3 are the test of this change. Steps 4-5 are the test of the
underlying theory and may fail independently** - see "If it does not work".

## Baseline, 2026-08-05 12:30 pass

| quantity | co_reply | co_retweet |
|---|---|---|
| pairable accounts | 3,949 | 27,758 |
| edges tested | 2,251 | 667,620 |
| tested pairs per pairable account | **0.57** | 24 |
| `edges_bonferroni` | 56 | 18,756 |
| accounts in validated layer | 72 | 4,094 |

Bridge accounts across both layers: **1**. Account pairs validated in both
channels: **0**, and that has been 0 in all 23 persisted passes.

Corroboration history, for the flicker: 0,1,1,1,0,0 over the six passes ending
2026-08-05 12:30, tracking bridge count 2,2,2,2,1,1.

Collector side at deploy: `conversations_in_band` 2,423, of which 2,412
unthreaded (the backlog), 250 selected per pass at degree 58-100.

## First pass after deploy, 2026-08-06 13:53 (`fce6d5c`)

Recorded immediately, so the collector-side half of the claim is settled before
the analysis-side half has had time to move.

| | old (`17f4338`) | new (`fce6d5c`) |
|---|---|---|
| `top_conversations` | 60 | 250 |
| `selected_conversations` | 60 | 250 |
| `conv_deg_min` / `conv_deg_max` | not recorded | 60 / 100 |
| `conversations_in_band` | not recorded | 2,407 |
| `conversations_unthreaded` | not recorded | 2,397 |
| `reply_rows` | 2,026 | **5,843** |

Reply supply up 2.9x, every selected conversation inside the band. Steps 1-3 of
the claim are now the open question; the arm itself is doing what it was
changed to do.

Two things the first pass taught that the design did not anticipate:

**The API caps delivered replies at ~36 per thread** regardless of the root's
stated `reply_count` (observed over 38 fetches: min 20, p50 36, max 36, none
above 100). `SNOWBALL_REPLIES_LIMIT=150` never binds. So per-conversation yield
is fixed and the ONLY lever on reply volume is the number of conversations -
widening `top_conversations` was the right choice and going deeper would have
bought nothing. It also means the 82 parents above the hub cap in §11 accumulate
across repeated fetches and other collection paths, not within one fetch.

**`due_conversations` was 177, not 250.** `threaded_expr` (R2 evidence) and
`_due` (local JSON state) disagree on 73 threads, and the disagreement is
correct in both directions: a thread that returned zero replies is marked in
local state but leaves no rows in R2, so the R2 check reads it as unworked
while the local check rightly skips it. The two guards are complementary, not
redundant. Do not "simplify" by deleting either.

## The check

Run this after 3+ days of collection (2026-08-09 or later).

The new `census_runs` columns do not exist in R2 until the first post-deploy
pass writes one. `union_by_name=true` unions the columns that are present in at
least one file, so before that pass the column is not merely NULL, it is
unbindable and the query errors. The guard below turns that into a clear
message.

```bash
cd analysis && uv run python - <<'PY'
import pandas as pd
from kma.db import connect, BUCKET, census_runs_source
con = connect()
pd.set_option("display.width", 260)

# 1. Collector: is the arm working its band, and is the backlog draining?
src = census_runs_source('x')
cols = set(con.execute(f"SELECT * FROM {src} LIMIT 0").df().columns)
if "conversations_in_band" not in cols:
    print("NO post-deploy census pass has written yet - new columns absent.\n"
          "Check the container is on the new image before reading anything else.")
else:
    print(con.execute(f"""
     SELECT date_trunc('day', collected_at) AS day,
            count(*) AS passes,
            avg(conversations_in_band)::INT AS in_band,
            avg(conversations_unthreaded)::INT AS backlog,
            avg(selected_conversations)::INT AS selected,
            avg(conv_deg_min)::INT AS deg_min, avg(conv_deg_max)::INT AS deg_max,
            avg(reply_rows)::INT AS reply_rows
     FROM {src}
     WHERE pass_kind='baseline' AND conversations_in_band IS NOT NULL
     GROUP BY 1 ORDER BY 1
    """).df().to_string())

# 2. Analysis: did co_reply's pairable supply and yield move?
g = f"r2://{BUCKET}/coordination/platform=x/kind=run_metrics/dt=*/run=*.parquet"
print(con.execute(f"""
 SELECT computed_at, channel, pairable, edges_tested, edges_bonferroni,
        round(edges_tested::DOUBLE / nullif(pairable,0), 2) AS tested_per_pairable,
        n_corroborated_clusters AS corr
 FROM read_parquet('{g}', union_by_name=true, hive_partitioning=true)
 WHERE computed_at > now() - INTERVAL 7 DAY
 ORDER BY computed_at, channel
""").df().to_string())
PY
```

Bridge accounts are not persisted (that was option 3, not done), so compute
them from the persisted layers:

```bash
cd analysis && uv run python - <<'PY'
import pandas as pd
from kma.db import connect, BUCKET
con = connect()

def load(ch):
    g = (f"r2://{BUCKET}/coordination/platform=x/kind=edges/channel={ch}"
         f"/method=svn_bonf/dt=*/run=*.parquet")
    d = con.execute(f"SELECT computed_at, src, dst FROM read_parquet('{g}', union_by_name=true)").df()
    d["pass"] = pd.to_datetime(d["computed_at"], utc=True).dt.floor("10min")
    return d

rt, rp = load("co_retweet"), load("co_reply")
rows = []
for p in sorted(set(rt["pass"]) | set(rp["pass"]))[-12:]:
    a, b = rt[rt["pass"] == p], rp[rp["pass"] == p]
    A, B = set(a["src"]) | set(a["dst"]), set(b["src"]) | set(b["dst"])
    pa = set(map(frozenset, zip(a["src"], a["dst"])))
    pb = set(map(frozenset, zip(b["src"], b["dst"])))
    rows.append({"pass": str(p)[:16], "rt_accts": len(A), "rp_accts": len(B),
                 "bridge_accts": len(A & B), "shared_pairs": len(pa & pb)})
print(pd.DataFrame(rows).to_string())
PY
```

## How to read it

**Working.** `backlog` falls from ~2,400 while `selected` holds near 250 and
`deg_min`/`deg_max` stay inside 3-100. co_reply `pairable` and
`tested_per_pairable` both rise. This is the change doing what it was built to
do, whatever happens to corroboration.

**Deployed but inert.** `conversations_in_band` is NULL, or `selected` is 0
while `backlog` is large. That is a plumbing fault, not a finding: check the
container is on the new image (`docker inspect collector-monitor-1`) and that
`replies_view` reached `hot_objects` from the scheduler.

**Band exhausted.** `backlog` approaches 0 and `deg_min` sits at 3. The arm has
worked its whole supply and `SNOWBALL_REFRESH_HOURS`, not
`SNOWBALL_TOP_CONVERSATIONS`, now governs. Do not raise the budget further; it
would only re-fetch inside the TTL.

**Contradicted.** `pairable` flat after 3+ days with a draining backlog. The
banding is not the constraint and §11's diagnosis is wrong.

## If it does not work

The specific way this can fail while still being correct: **`pairable` rises
but bridge accounts stay at 1-2.** That means the collector change worked and
the theory behind it did not - repliers and retweeters in this corpus are
genuinely different people. That is a finding about Kenyan X discourse, not a
bug, and it makes further census tuning pointless.

In that case the route is a third channel rather than more collection:
`co_url`, `co_hashtag`, `co_mention` are post-derived and genuinely different
behaviours. `coordination.coverage()` reports what share of posts carry each
field, which is the cheap first check. Splitting `co_retweet` into census and
post arms remains rejected for the reason in §10 - it would make corroboration
easier and less meaningful.

## Standing caveat

`n_corroborated_clusters` is not yet safe to read alone. Until bridge accounts
are in the tens it means "cannot be evaluated", and a 1 is a coin flip on a
single account. Persisting the bridge count next to it is the outstanding
follow-up.

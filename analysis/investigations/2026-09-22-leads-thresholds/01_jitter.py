"""How much does a Leiden community's membership move between runs when
nothing real has changed?

    cd analysis && uv run python investigations/2026-09-22-leads-thresholds/01_jitter.py

`kma.leads` matches communities across runs by member Jaccard and needs two
cuts: below which a community has no predecessor (new), and below which it has
changed enough to judge again. Both have to sit outside the run-to-run noise of
the partition itself, or every run re-alerts on the same groups.

There is no daily v2 series yet - that is what workstream A is building - so
the noise is measured on v1's persisted runs, which are the same algorithm
family (Leiden, modularity) refreshed ~3 times a day over the same growing
corpus. v1 runs a different graph at a different resolution, so the numbers
set a starting point for v2 and must be re-measured once two weeks of daily v2
listings exist.

Also measures, for the trend channel, how often a candidate tag recurs from one
trend run to the next.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from kma import leads
from kma.db import BUCKET, connect

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
MIN_SIZE = leads.MIN_COMMUNITY_SIZE


def load_runs(con, n: int) -> list[tuple[str, pd.DataFrame]]:
    files = [r[0] for r in con.sql(
        f"SELECT file FROM glob('r2://{BUCKET}/coordination/platform=x/kind=clusters/**/*.parquet') "
        "ORDER BY file"
    ).fetchall()][-n:]
    out = []
    for f in files:
        d = con.sql(f"SELECT author_id, cluster_id FROM read_parquet('{f}')").df()
        d = d.rename(columns={"author_id": "user_id", "cluster_id": "community"})
        out.append((f.split("run=")[1].removesuffix(".parquet"), d))
    return out


def pair(prev: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    sized = lambda f: f[f.groupby("community")["user_id"].transform("size") >= MIN_SIZE]  # noqa: E731
    return leads.best_matches(leads.groups(sized(prev), "community", "user_id"),
                              leads.groups(sized(cur), "community", "user_id"))


def summarise(m: pd.DataFrame) -> dict:
    j = m["jaccard"]
    return {
        "communities": len(m),
        "identical": round(float((j == 1.0).mean()), 3),
        "j>=0.8": round(float((j >= 0.8).mean()), 3),
        "j>=0.5": round(float((j >= 0.5).mean()), 3),
        "0.2<=j<0.5": round(float(((j >= 0.2) & (j < 0.5)).mean()), 3),
        "j<0.2": round(float((j < 0.2).mean()), 3),
        "j==0": round(float((j == 0).mean()), 3),
        "grew>=5&50%": int(((m["joined"] >= leads.GROWTH_MIN)
                            & (m["joined"] >= leads.GROWTH_SHARE * m["previous_size"])).sum()),
    }


def main() -> None:
    con = connect()
    runs = load_runs(con, RUNS)
    print(f"{len(runs)} v1 runs, {runs[0][0]} .. {runs[-1][0]}; communities of {MIN_SIZE}+ only\n")

    for lag, label in ((1, "consecutive (~8h)"), (3, "~1 day"), (9, "~3 days"), (21, "~7 days")):
        rows = [summarise(pair(runs[i - lag][1], runs[i][1])) for i in range(lag, len(runs))]
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        med = frame.median(numeric_only=True)
        print(f"lag {lag:>2} runs, {label}: {len(rows)} pairs, median over pairs")
        print("   " + "  ".join(f"{k}={med[k]:g}" for k in frame.columns))

    # The distribution itself, consecutive pairs pooled: is it bimodal, and
    # where is the valley?
    pooled = pd.concat([pair(runs[i - 1][1], runs[i][1]) for i in range(1, len(runs))])
    hist, edges = np.histogram(pooled["jaccard"], bins=np.linspace(0, 1, 11))
    print("\nconsecutive-pair best-match Jaccard, pooled:")
    for h, lo, hi in zip(hist, edges[:-1], edges[1:], strict=True):
        print(f"   [{lo:.1f},{hi:.1f}{']' if hi == 1 else ')'} {h:6d} {'#' * int(60 * h / hist.max())}")

    # What each candidate cut would flag per consecutive run, which is the
    # re-alert rate on unchanged data if the cut were used.
    print("\nflags per consecutive run at candidate cuts (median, max):")
    per_run = [pair(runs[i - 1][1], runs[i][1]) for i in range(1, len(runs))]
    for new_below, changed_below in ((0.1, 0.3), (0.2, 0.5), (0.3, 0.6), (0.2, 0.7)):
        t = leads.Thresholds(new_below=new_below, changed_below=changed_below)
        n_new = [int((leads.classify(m, t) == leads.STATUS_NEW).sum()) for m in per_run]
        n_chg = [int((leads.classify(m, t) == leads.STATUS_CHANGED).sum()) for m in per_run]
        size = [len(m) for m in per_run]
        print(f"   new<{new_below} changed<{changed_below}: new {np.median(n_new):g} (max {max(n_new)}), "
              f"changed {np.median(n_chg):g} (max {max(n_chg)}) of median {np.median(size):g} communities")

    # Trend channel.
    t = con.sql(
        f"SELECT run_id, tag, authors, emergence FROM read_parquet("
        f"'r2://{BUCKET}/trend_candidates/platform=x/*/*.parquet', union_by_name=true, hive_partitioning=true)"
    ).df()
    runs_t = sorted(t["run_id"].unique())
    print(f"\ntrend runs: {len(runs_t)}; candidates per run: "
          f"{t.groupby('run_id').size().describe()[['mean', 'min', 'max']].round(2).to_dict()}")
    fresh, recurring = 0, 0
    for a, b in zip(runs_t[:-1], runs_t[1:], strict=True):
        prev_tags = set(t.loc[t["run_id"] == a, "tag"])
        cur = t.loc[t["run_id"] == b]
        d = leads.diff_trends(t.loc[t["run_id"] == a], cur)
        fresh += int((d["status"] == leads.STATUS_NEW).sum())
        recurring += int(cur["tag"].isin(prev_tags).sum())
    print(f"across {len(runs_t) - 1} consecutive trend-run pairs: {fresh} new tags, {recurring} recurring")
    grown = t.sort_values("run_id").groupby("tag")["authors"].agg(["first", "max", "count"])
    print("tags seen in 2+ runs, authors first -> max:")
    print(grown[grown["count"] > 1].sort_values("count", ascending=False).head(12).to_string())


if __name__ == "__main__":
    main()

"""Weekly model-versus-gate disagreement: the standing drift alarm.

    cd analysis && uv run python investigations/2026-09-12-relevance-classifier/11_drift.py [--weeks 12]

Neither number is truth, so this does not say which is right. What it says is
whether the RELATIONSHIP between them is moving. The two disagree at a stable
rate while the corpus keeps its shape; a jump means one of three things, all
worth knowing: a new campaign vocabulary the lexicon has not got, a change in
what the collector is bringing in, or the model drifting away from the corpus it
was trained on in September 2026.

Scored posts only - unscored ones are just the scorer's backlog.
"""

import argparse

import pandas as pd

from kma import measure, relevance
from kma.db import connect, posts_source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weeks", type=int, default=12)
    ap.add_argument("--sample", type=int, default=40_000, help="posts per week to read")
    args = ap.parse_args()

    con = connect()
    posts = con.sql(f"""
        WITH s AS ({relevance.latest_scores_cte()}),
        p AS (SELECT platform_post_id, text, created_at FROM {posts_source('x')}
              WHERE text IS NOT NULL AND created_at > now() - INTERVAL {int(args.weeks)} WEEK
              QUALIFY row_number() OVER (
                  PARTITION BY platform_post_id ORDER BY collected_at DESC) = 1)
        SELECT date_trunc('week', p.created_at) AS week, p.text, s.p_kenya
        FROM p JOIN s ON s.platform_post_id = p.platform_post_id
        USING SAMPLE reservoir({int(args.sample) * int(args.weeks)} ROWS) REPEATABLE (0)
    """).df()
    if posts.empty:
        raise SystemExit("no scored posts in the window")

    posts["model"] = posts["p_kenya"] >= relevance.THRESHOLD
    posts["gate"] = posts["text"].map(measure.domain_bucket) == "kenya"
    weekly = posts.groupby(posts["week"].dt.date).agg(
        posts=("model", "size"),
        model_kenya=("model", "mean"),
        gate_kenya=("gate", "mean"),
        model_not_gate=("model", lambda s: float((s & ~posts.loc[s.index, "gate"]).mean())),
        gate_not_model=("gate", lambda s: float((posts.loc[s.index, "gate"] & ~s).mean())),
    ).round(3)
    weekly["disagree"] = (weekly["model_not_gate"] + weekly["gate_not_model"]).round(3)
    print(weekly.to_string())
    recent, earlier = weekly["disagree"].iloc[-4:].mean(), weekly["disagree"].iloc[:-4].mean()
    print(f"\nlast 4 weeks {recent:.3f} against {earlier:.3f} before them"
          f" ({'stable' if abs(recent - earlier) < 0.05 else 'MOVED - look at what changed'})")


if __name__ == "__main__":
    main()

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
    ap.add_argument("--weeks", type=int, default=8)
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
    posts["bucket"] = posts["text"].map(measure.domain_bucket)
    posts["gate"] = posts["bucket"] == "kenya"
    # Both directions need both columns, so this is a frame-wise apply: an
    # `agg` lambda sees one column, and `gate & ~gate` is silently always zero.
    def rates(group: pd.DataFrame) -> pd.Series:
        out = {
            "posts": len(group),
            "model_kenya": group["model"].mean(),
            "gate_kenya": group["gate"].mean(),
            "model_not_gate": (group["model"] & ~group["gate"]).mean(),
            "gate_not_model": (group["gate"] & ~group["model"]).mean(),
        }
        # Per bucket as well as overall: the corpus mix itself moves (the
        # collector's composition drift is documented in OBJECTIVES C1), and a
        # rate that moves with the mix is an alarm that cries wolf every time
        # targeting changes. Within a bucket, like is compared with like.
        for bucket in ("ambiguous", "kenya", "offdomain"):
            rows = group[group["bucket"] == bucket]
            out[f"disagree:{bucket}"] = (
                (rows["model"] != rows["gate"]).mean() if len(rows) else float("nan"))
        return pd.Series(out)

    weekly = posts.groupby(posts["week"].dt.date)[["model", "gate", "bucket"]].apply(rates).round(3)
    weekly["posts"] = weekly["posts"].astype(int)
    weekly["disagree"] = (weekly["model_not_gate"] + weekly["gate_not_model"]).round(3)
    print(weekly.to_string())
    # The alarm watches the `ambiguous` bucket: 76% of the corpus, where the
    # model does all of its work and the lexicon abstains by construction.
    series = weekly["disagree:ambiguous"].dropna()
    recent, earlier = series.iloc[-4:].mean(), series.iloc[-8:-4].mean()
    print(f"\ndisagreement inside the ambiguous bucket: last 4 weeks {recent:.3f} "
          f"against the 4 before them {earlier:.3f}"
          f" ({'stable' if abs(recent - earlier) < 0.05 else 'MOVED - look at what changed'})")
    print("Compare adjacent windows only. Reaching further back crosses collection-regime\n"
          "changes: in early July the ambiguous bucket disagreed at 0.55 because it was mostly\n"
          "Kenyan content, and by August mostly not. That is the collector's composition drift\n"
          "(OBJECTIVES C1), not the model moving.")


if __name__ == "__main__":
    main()

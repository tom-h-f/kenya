"""Temporal fingerprints: activity-rhythm forensics per account.

- circadian profile: hour-of-day histogram (Africa/Nairobi) per account with
  >=MIN_POSTS captured posts; distinctiveness = JSD vs the corpus profile
- shift cohorts: among distinctive accounts, connect pairs whose profiles are
  near-identical (JSD < PAIR_EPS, calibrated against random-pair quantiles),
  report connected components of >=3 accounts
- interval regularity: coefficient of variation of inter-post gaps
  (scheduler-like posting has low CV)
- awakened accounts: in-window posting rate far above lifetime rate
  (profile tweet_count / account age) - dormant-then-activated pattern

Baseline is the corpus hourly distribution, not uniform: capture cadence
shapes every profile, so only deviation from the shared rhythm counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _lib
from kma import db

MIN_POSTS = 20
DISTINCT_Q = 0.90
PAIR_EPS_Q = 0.001
MIN_COHORT = 3


def jsd(p: np.ndarray, q: np.ndarray) -> float | np.ndarray:
    m = (p + q) / 2

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = a * np.log2(a / b)
        return np.where(a > 0, t, 0.0).sum(axis=-1)

    return (kl(p, m) + kl(q, m)) / 2


def main() -> None:
    args = _lib.parse_args("Temporal fingerprints", default_sample=500)
    con = db.connect()

    posts = con.sql(
        f"""
        SELECT author_id, author_handle,
               hour(created_at AT TIME ZONE 'Africa/Nairobi') AS hr,
               epoch(created_at) AS ts
        FROM {db.posts_source('x')}
        WHERE created_at IS NOT NULL
        QUALIFY row_number() OVER (
            PARTITION BY platform_post_id ORDER BY collected_at DESC
        ) = 1
        """
    ).df()
    counts = posts.groupby("author_id").size()
    eligible = counts[counts >= MIN_POSTS].index
    if args.sample:
        eligible = eligible[: args.sample]
    posts = posts[posts["author_id"].isin(eligible)]
    handles = posts.groupby("author_id")["author_handle"].first()
    print(f"{len(eligible)} accounts with >={MIN_POSTS} posts")

    hist = (
        posts.groupby(["author_id", "hr"]).size().unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )
    profiles = (hist + 0.5).div((hist + 0.5).sum(axis=1), axis=0)
    corpus = hist.sum(axis=0)
    corpus = ((corpus + 0.5) / (corpus + 0.5).sum()).to_numpy()

    P = profiles.to_numpy()
    ids = profiles.index.to_numpy()
    base_jsd = jsd(P, corpus[None, :])
    distinct_cut = np.quantile(base_jsd, DISTINCT_Q)
    sel = base_jsd >= distinct_cut
    D, dids = P[sel], ids[sel]
    print(f"{len(dids)} distinctive accounts (JSD-to-corpus >= {distinct_cut:.4f})")

    n = len(D)
    pair_j = np.full((n, n), 1.0)
    for i in range(n):
        pair_j[i, i + 1:] = jsd(D[i][None, :], D[i + 1:])
    rng = np.random.default_rng(7)
    ri = rng.integers(0, len(P), 4000)
    rj = rng.integers(0, len(P), 4000)
    keep = ri != rj
    rand_j = jsd(P[ri[keep]], P[rj[keep]])
    eps = np.quantile(rand_j, PAIR_EPS_Q)
    print(f"pair threshold JSD < {eps:.5f} (q{PAIR_EPS_Q} of random pairs)")

    adj = pair_j < eps
    comp = -np.ones(n, dtype=int)
    c = 0
    for i in range(n):
        if comp[i] >= 0:
            continue
        stack, comp[i] = [i], c
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adj[u] | adj[:, u]):
                if comp[v] < 0:
                    comp[v] = c
                    stack.append(v)
        c += 1

    clusters_cache = _lib.coordination_clusters()
    in_cluster = set(clusters_cache.get("author_id", pd.Series(dtype=str)))
    rows = []
    for cid in range(c):
        idx = np.flatnonzero(comp == cid)
        if len(idx) < MIN_COHORT:
            continue
        members = dids[idx]
        sub = pair_j[np.ix_(idx, idx)]
        vals = sub[np.triu_indices(len(idx), 1)]
        peak = profiles.loc[members].mean(axis=0).nlargest(3).index.tolist()
        rows.append(
            {
                "cohort": cid,
                "accounts": len(idx),
                "mean_pair_jsd": round(float(vals.mean()), 5),
                "peak_hours_eat": peak,
                "also_in_coord_cluster": sum(m in in_cluster for m in members),
                "handles": ", ".join(handles.loc[members].head(8)),
            }
        )
    cohorts = pd.DataFrame(
        rows, columns=["cohort", "accounts", "mean_pair_jsd", "peak_hours_eat",
                       "also_in_coord_cluster", "handles"],
    ).sort_values("accounts", ascending=False)
    _lib.show(cohorts, "shift cohorts (near-identical circadian profiles)")
    _lib.save(cohorts, "01_shift_cohorts.csv")

    g = posts.sort_values("ts").groupby("author_id")["ts"]
    gaps = g.apply(lambda s: s.diff().dropna())
    reg = gaps.groupby(level=0).agg(["count", "mean", "std"])
    reg = reg[reg["count"] >= MIN_POSTS - 1]
    reg["cv"] = reg["std"] / reg["mean"]
    reg["median_gap_min"] = (
        gaps.groupby(level=0).median().reindex(reg.index) / 60
    ).round(1)
    reg = reg.join(handles).sort_values("cv").reset_index()
    reg["in_coord_cluster"] = reg["author_id"].isin(in_cluster)
    _lib.show(
        reg[["author_handle", "count", "cv", "median_gap_min", "in_coord_cluster"]],
        "most regular posting intervals (low CV = scheduler-like)",
    )
    _lib.save(
        reg[["author_id", "author_handle", "count", "cv", "median_gap_min",
             "in_coord_cluster"]],
        "01_regular_intervals.csv",
    )

    authors = db.latest_authors(con).df()[
        ["platform_user_id", "handle", "created_at", "tweet_count", "followers_count"]
    ].dropna(subset=["created_at"])
    window = posts.groupby("author_id").agg(
        n_window=("ts", "size"), t0=("ts", "min"), t1=("ts", "max")
    )
    aw = window.merge(
        authors, left_index=True, right_on="platform_user_id"
    )
    aw["age_days"] = (
        pd.Timestamp.now(tz="UTC") - pd.to_datetime(aw["created_at"], utc=True)
    ).dt.days.clip(lower=1)
    aw["window_days"] = ((aw["t1"] - aw["t0"]) / 86_400).clip(lower=1.0)
    aw["window_rate"] = aw["n_window"] / aw["window_days"]
    aw["lifetime_rate"] = aw["tweet_count"] / aw["age_days"]
    aw["awaken_ratio"] = aw["window_rate"] / aw["lifetime_rate"].clip(lower=0.01)
    awakened = aw[(aw["age_days"] > 365) & (aw["awaken_ratio"] > 20)]
    awakened = awakened.sort_values("awaken_ratio", ascending=False)
    awakened["in_coord_cluster"] = awakened["platform_user_id"].isin(in_cluster)
    cols = ["handle", "age_days", "tweet_count", "n_window", "window_rate",
            "lifetime_rate", "awaken_ratio", "followers_count", "in_coord_cluster"]
    _lib.show(awakened[cols], "awakened accounts (old, quiet lifetime, loud now)")
    _lib.save(
        awakened[["platform_user_id"] + cols], "01_awakened.csv"
    )

    _lib.print_caveats()


if __name__ == "__main__":
    main()

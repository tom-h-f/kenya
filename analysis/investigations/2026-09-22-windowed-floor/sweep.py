"""The `MIN_ENTITIES` sweep for windowed v2: what each (width, floor) keeps,
what it reports on noise, and whether it finds a planted campaign.

Three arms per (width, floor), all over the same timed traces:

- REAL: `windowed.run_rows` as production would run it.
- NULL: every window's incidence curveball-shuffled per trace before the
  detector sees it. Both degree sequences survive, so what the detector still
  reports is explained by activity and popularity, not co-action. The chance
  share is null over real, in account-windows shown to a reader.
- PLANT: a synthetic hashtag-for-hire campaign inserted into one day, with
  organic background borrowed from real accounts on other days. Scored in its
  own window only - windows are independent, so the rest is not recomputed.

Run through `modal_windowed_sweep.py`; the functions here are pure so the
tests and the Modal app share them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kma import windowed


@dataclass(frozen=True)
class Plant:
    """One campaign shape. Distinct entities per account per trace are what the
    floor counts, so they are the parameters, not totals."""

    name: str
    k: int
    tweets: int
    tweets_each: int
    tag_variants: int
    tags_each: int
    authors: int
    authors_each: int
    urls: int
    urls_each: int
    burst_hours: int = 6


# Light: below floor 5 on every trace by construction - it is visible there
# only if organic background lifts an account over the floor.
LIGHT = Plant("light", k=12, tweets=6, tweets_each=4, tag_variants=3, tags_each=2,
              authors=3, authors_each=2, urls=2, urls_each=1)
# Heavy: clears floor 10 on co_retweet alone.
HEAVY = Plant("heavy", k=20, tweets=15, tweets_each=10, tag_variants=6, tags_each=5,
              authors=4, authors_each=3, urls=3, urls_each=2)
PLANTS = {p.name: p for p in (LIGHT, HEAVY)}


def load(cache) -> dict[str, pd.DataFrame]:
    """Timed traces cached by `01_sweep.py extract`."""
    from pathlib import Path

    return {n: pd.read_parquet(Path(cache) / f"{n}.parquet") for n in windowed.TIMED_TRACES}


def labelled(rows: dict[str, pd.DataFrame], width: str) -> dict[str, pd.DataFrame]:
    return {
        name: trace.assign(window=windowed.window_label(trace["created_at"], width))
        for name, trace in rows.items()
    }


def complete_windows(rows: dict[str, pd.DataFrame], width: str) -> list[str]:
    """Windows wholly inside the extracted range. The first and last are
    partial - a day cut at the extraction boundary would read as a quiet day."""
    stamps = pd.concat([t["created_at"] for t in rows.values()])
    start, end = stamps.min(), stamps.max()
    span = pd.Timedelta(days=1 if width == "day" else 7)
    out = []
    for label in sorted(set().union(*(set(t["window"]) for t in labelled(rows, width).values()))):
        begin = pd.Timestamp(label, tz="UTC")
        if begin >= start and begin + span <= end:
            out.append(label)
    return out


def summary(per_window: pd.DataFrame) -> dict:
    shown = per_window[per_window["community_size"] >= windowed.MIN_COMMUNITY]
    per = per_window.groupby("window")["user_id"].size() if len(per_window) else pd.Series(dtype=int)
    comms = shown.groupby("window")["community"].nunique() if len(shown) else pd.Series(dtype=int)
    return {
        "windows_scored": int(per_window["window"].nunique()) if len(per_window) else 0,
        "accounts_distinct": int(per_window["user_id"].nunique()) if len(per_window) else 0,
        "account_windows": int(len(per_window)),
        "accounts_median": float(per.median()) if len(per) else 0.0,
        "communities": int(comms.sum()) if len(comms) else 0,
        "communities_median": float(comms.median()) if len(comms) else 0.0,
        "shown_account_windows": int(len(shown)),
        "largest_community": int(per_window["community_size"].max()) if len(per_window) else 0,
    }


def run_real(rows, *, width, floor, windows):
    lab = labelled(rows, width)
    frames = []
    for window in windows:
        in_window = {n: t[t["window"] == window] for n, t in lab.items()}
        networks, _ = windowed.window_networks(in_window, min_entities=floor)
        if networks:
            frames.append(windowed.score_window(networks).assign(window=window))
    return pd.concat(frames, ignore_index=True) if frames else _empty()


def run_null(rows, *, width, floor, windows, seed=0, burn_in=100):
    lab = labelled(rows, width)
    frames = []
    for i, window in enumerate(windows):
        shuffled = {
            n: windowed.curveball(t[t["window"] == window], burn_in=burn_in, seed=seed * 10_000 + i)
            for n, t in lab.items()
        }
        networks, _ = windowed.window_networks(shuffled, min_entities=floor)
        if networks:
            frames.append(windowed.score_window(networks).assign(window=window))
    return pd.concat(frames, ignore_index=True) if frames else _empty()


def _empty():
    return pd.DataFrame(columns=["user_id", "centrality", "community", "community_size",
                                 "community_eigenvalue", "community_centrality", "window"])


def plant(
    rows: dict[str, pd.DataFrame], shape: Plant, day: str, *, seed: int
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Insert one campaign into `day`. Returns the augmented traces (a copy;
    the input is never modified) and the synthetic account ids.

    Background: each synthetic account takes one real account's actions from a
    different day, re-timed onto the plant day at the same time of day. So a
    planted account carries organic entities as well as the campaign's, as a
    rented account would - a clean campaign-only node is easier to find than
    anything real."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp(day, tz="UTC")
    start = base + pd.Timedelta(hours=int(rng.integers(6, 13)))
    ids = [f"synthetic_{shape.name}_{seed}_{i:03d}" for i in range(shape.k)]

    def when(n):
        return start + pd.to_timedelta(rng.uniform(0, shape.burst_hours * 3600, n), unit="s")

    def pick(pool, each):
        return rng.choice(pool, size=min(each, len(pool)), replace=False)

    tag = f"camp{seed}"
    campaign = {
        "co_retweet": [f"{tag}_tweet{i}" for i in range(shape.tweets)],
        "hashtag_sequence": [f"{tag}|kenya|linda{i}" for i in range(shape.tag_variants)],
        "fast_retweet": [f"{tag}_author{i}" for i in range(shape.authors)],
        "co_url": [f"https://example.test/{tag}/{i}" for i in range(shape.urls)],
    }
    each = {"co_retweet": shape.tweets_each, "hashtag_sequence": shape.tags_each,
            "fast_retweet": shape.authors_each, "co_url": shape.urls_each}

    # Donors: real accounts active on some other day, one per synthetic account.
    days = {name: t["created_at"].dt.strftime("%Y-%m-%d") for name, t in rows.items()}
    by_day = pd.concat(
        [pd.DataFrame({"user_id": t["user_id"].to_numpy(), "day": days[n].to_numpy()})
         for n, t in rows.items()],
        ignore_index=True,
    ).drop_duplicates()
    donors = by_day[by_day["day"] != day]
    donor_rows = donors.sample(shape.k, random_state=int(rng.integers(2**31)))
    wanted = set(zip(donor_rows["user_id"], donor_rows["day"]))
    donor_traces = {
        name: t[[(u, d) in wanted for u, d in zip(t["user_id"], days[name])]].assign(
            day=days[name][[(u, d) in wanted for u, d in zip(t["user_id"], days[name])]]
        )
        for name, t in rows.items()
    }

    added: dict[str, list[pd.DataFrame]] = {n: [] for n in rows}
    for sid, (_, donor) in zip(ids, donor_rows.iterrows(), strict=True):
        for name, trace in rows.items():
            dt = donor_traces[name]
            mine = dt[(dt["user_id"] == donor["user_id"]) & (dt["day"] == donor["day"])]
            if len(mine):
                offset = mine["created_at"] - mine["created_at"].dt.floor("D")
                added[name].append(pd.DataFrame(
                    {"user_id": sid, "entity": mine["entity"].to_numpy(),
                     "created_at": base + offset.to_numpy()}
                ))
            if name in campaign:
                chosen = pick(campaign[name], each[name])
                added[name].append(pd.DataFrame(
                    {"user_id": sid, "entity": chosen, "created_at": when(len(chosen))}
                ))

    out = {}
    for name, trace in rows.items():
        extra = [f for f in added[name] if len(f)]
        out[name] = pd.concat([trace[["user_id", "entity", "created_at"]], *extra],
                              ignore_index=True) if extra else trace
    return out, ids


def plant_metrics(per_window: pd.DataFrame, ids: list[str], window: str) -> dict:
    """Would a reader see the campaign in its window's community report?

    `recall` and `precision` are for the community holding most planted
    accounts; `community_rank` is its place among the window's communities of
    `MIN_COMMUNITY`+ ordered by leading eigenvalue, the report's order."""
    frame = per_window[per_window["window"] == window]
    planted = frame[frame["user_id"].isin(ids)]
    out = {"present": len(planted) / len(ids), "recall": 0.0, "precision": 0.0,
           "fragments": 0, "community_rank": None, "communities_in_window": 0}
    shown = frame[frame["community_size"] >= windowed.MIN_COMMUNITY]
    ranked = (shown.drop_duplicates("community")
              .sort_values("community_eigenvalue", ascending=False)["community"].tolist())
    out["communities_in_window"] = len(ranked)
    if planted.empty:
        return out
    hits = planted.groupby("community").size()
    best = hits.idxmax()
    size = int(frame.loc[frame["community"] == best, "community_size"].iloc[0])
    out.update(
        recall=float(hits.max() / len(ids)),
        precision=float(hits.max() / size),
        fragments=int(len(hits)),
        community_rank=(ranked.index(best) + 1) if best in ranked else None,
    )
    return out

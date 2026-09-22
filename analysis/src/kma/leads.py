"""What changed since the last run, and which of it a reader should see.

The daily v2 pass writes a fresh community listing and the collector writes
trend candidates twice a day. Almost all of each is the same as last time, and
re-adjudicating it would spend the reader's budget on groups already judged.
This module is the diff: it decides which communities are new or have changed
enough to judge again, which trend tags are new, and ranks them into a bounded
queue for `modal_leads.py`.

## Communities cannot be joined on their id

A community id is Leiden's positional label for one partition, reissued every
run - community 3 today and community 3 tomorrow are unrelated. Neither does
`coordination.stable_cluster_id` help: it hashes the exact member set, so one
account joining makes it a different id. Communities are matched across runs
by member overlap instead: each current community's best predecessor is the
previous community it shares the highest Jaccard with.

## What "materially changed" means

Three outcomes per current community, from its best match:

- **new** - best Jaccard below `NEW_BELOW`: no predecessor shares enough of it.
- **changed** - a predecessor exists but either the Jaccard is below
  `CHANGED_BELOW`, or the community gained at least `GROWTH_MIN` members and
  `GROWTH_SHARE` of its predecessor's size. Growth is its own test because a
  community that absorbs a wave of new accounts keeps every old member - its
  Jaccard falls, but containment of the old group stays at 1, and "the same
  campaign, now twice as big" is exactly what a reader should see again.
- **same** - everything else, and not re-judged.

The thresholds are set from measured run-to-run jitter on unchanged data; see
`analysis/investigations/2026-09-22-leads-thresholds/findings.md`.

## Trend tags

A tag is **new** if it was not a candidate in the previous trend run. A tag
already seen is **changed** only if its author count grew by `GROWTH_MIN` and
doubled - a tag being re-selected by the emergence rule is not news.

TRIAGE, NEVER A VERDICT. Everything here ranks groups for a reader. Nothing
here says anything about any person.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from kma.coordination import stable_cluster_id

log = logging.getLogger("kma")

# Set from the jitter measurement in the 2026-09-22 leads-thresholds
# investigation. Change them only with a new measurement beside the old one.
NEW_BELOW = 0.2
CHANGED_BELOW = 0.5
GROWTH_MIN = 5
GROWTH_SHARE = 0.5
MIN_COMMUNITY_SIZE = 4

# The reader's daily budget, a cost bound rather than a statement about how
# much deserves attention. `TREND_RESERVE` slots are held back for trend tags
# so a busy day of community churn cannot starve the second channel.
MAX_CANDIDATES = 10
TREND_RESERVE = 3

STATUS_NEW = "new"
STATUS_CHANGED = "changed"
STATUS_SAME = "same"

SOURCE_COMMUNITY = "community"
SOURCE_TREND = "trend"


@dataclass(frozen=True)
class Thresholds:
    new_below: float = NEW_BELOW
    changed_below: float = CHANGED_BELOW
    growth_min: int = GROWTH_MIN
    growth_share: float = GROWTH_SHARE
    min_size: int = MIN_COMMUNITY_SIZE


def groups(frame: pd.DataFrame, key: str, member: str) -> dict:
    """{group id: frozenset of member ids} from a long member table."""
    if frame.empty:
        return {}
    ids = frame[[key, member]].dropna().astype({member: str})
    return {k: frozenset(g[member]) for k, g in ids.groupby(key)}


def best_matches(previous: dict, current: dict) -> pd.DataFrame:
    """Each current group's highest-Jaccard predecessor, one row per group.

    Only predecessors sharing at least one member are scored, through an
    inverted index, so the cost is the overlap and not the product of the two
    partitions. Ties on Jaccard go to the larger shared count, then to the
    smaller id so the result does not depend on dict order."""
    owner: dict[str, list] = defaultdict(list)
    for pid, members in previous.items():
        for m in members:
            owner[m].append(pid)

    rows = []
    for cid, members in current.items():
        shared: dict = defaultdict(int)
        for m in members:
            for pid in owner.get(m, ()):
                shared[pid] += 1
        best = None
        for pid, n in shared.items():
            union = len(members) + len(previous[pid]) - n
            key = (n / union, n, _neg(pid))
            if best is None or key > best[0]:
                best = (key, pid, n)
        if best is None:
            rows.append({"group": cid, "size": len(members), "previous": None,
                         "previous_size": 0, "shared": 0, "jaccard": 0.0,
                         "containment": 0.0, "joined": len(members)})
            continue
        (_, pid, n) = best
        rows.append({
            "group": cid,
            "size": len(members),
            "previous": pid,
            "previous_size": len(previous[pid]),
            "shared": n,
            "jaccard": best[0][0],
            # How much of the predecessor survives. 1.0 with a low Jaccard is
            # growth, not replacement.
            "containment": n / len(previous[pid]),
            "joined": len(members) - n,
        })
    return pd.DataFrame(rows, columns=["group", "size", "previous", "previous_size",
                                       "shared", "jaccard", "containment", "joined"])


def _neg(value):
    """Sort key that prefers the smaller id when used inside a max()."""
    try:
        return -float(value)
    except (TypeError, ValueError):
        return tuple(-ord(c) for c in str(value))


def classify(matches: pd.DataFrame, t: Thresholds = Thresholds()) -> pd.Series:
    """new / changed / same for each row of `best_matches`."""
    grew = (matches["joined"] >= t.growth_min) & (
        matches["joined"] >= t.growth_share * matches["previous_size"]
    )
    status = pd.Series(STATUS_SAME, index=matches.index)
    status[(matches["jaccard"] < t.changed_below) | grew] = STATUS_CHANGED
    status[matches["jaccard"] < t.new_below] = STATUS_NEW
    return status


def diff_communities(
    previous: pd.DataFrame | None,
    current: pd.DataFrame,
    t: Thresholds = Thresholds(),
) -> pd.DataFrame:
    """Community-level diff of two v2 listings.

    Both frames are one row per account with `user_id` and `community`, the
    columns `03_communities.py` writes and the daily pipeline persists.
    `community_eigenvalue` orders the result when present - it is what the
    detector's own score rewards. Communities under `min_size` are dropped on
    both sides before matching, as the listing drops them.

    With no previous run every community is new: the first daily run has
    nothing to compare against, and saying so is better than inventing a
    baseline."""
    need = {"user_id", "community"}
    missing = need - set(current.columns)
    if missing:
        raise ValueError(
            f"scores lack {sorted(missing)}; the diff needs the per-run community "
            "listing (see docs/analysis/leads.md for the contract)"
        )
    cur = _sized(current, t.min_size)
    prev = _sized(previous, t.min_size) if previous is not None else cur.iloc[0:0]

    matches = best_matches(groups(prev, "community", "user_id"),
                           groups(cur, "community", "user_id"))
    if matches.empty:
        return matches.assign(status=pd.Series(dtype=str))
    matches["status"] = classify(matches, t)
    matches["candidate_id"] = [
        stable_cluster_id(sorted(m)) for m in
        (groups(cur, "community", "user_id")[g] for g in matches["group"])
    ]
    if "community_eigenvalue" in cur.columns:
        eig = cur.groupby("community")["community_eigenvalue"].first()
        matches["eigenvalue"] = matches["group"].map(eig)
    if "kenya_share" in cur.columns:
        matches["kenya_share"] = matches["group"].map(cur.groupby("community")["kenya_share"].mean())
    order = ["eigenvalue", "size"] if "eigenvalue" in matches.columns else ["size"]
    return matches.sort_values(order, ascending=False).reset_index(drop=True)


def _sized(frame: pd.DataFrame, min_size: int) -> pd.DataFrame:
    counts = frame.groupby("community")["user_id"].transform("size")
    return frame[counts >= min_size]


def diff_trends(
    previous: pd.DataFrame | None,
    current: pd.DataFrame,
    t: Thresholds = Thresholds(),
) -> pd.DataFrame:
    """Tag-level diff of two `trend_candidates/` runs, highest emergence first."""
    cur = current.copy()
    if cur.empty:
        return cur.assign(status=pd.Series(dtype=str))
    before = (
        previous.drop_duplicates("tag").set_index("tag")["authors"]
        if previous is not None and not previous.empty else pd.Series(dtype=float)
    )
    prior = cur["tag"].map(before)
    grew = (cur["authors"] - prior.fillna(0) >= t.growth_min) & (cur["authors"] >= 2 * prior)
    cur["previous_authors"] = prior
    cur["status"] = STATUS_SAME
    cur.loc[grew.fillna(False), "status"] = STATUS_CHANGED
    cur.loc[prior.isna(), "status"] = STATUS_NEW
    cur["candidate_id"] = "trend:" + cur["tag"].astype(str)
    return cur.sort_values("emergence", ascending=False).reset_index(drop=True)


def queue(
    communities: pd.DataFrame,
    trends: pd.DataFrame,
    limit: int = MAX_CANDIDATES,
    trend_reserve: int = TREND_RESERVE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The bounded adjudication queue, and everything it skipped.

    New before changed within each channel, then the channel's own order.
    Trends get up to `trend_reserve` slots; slots a channel cannot use pass to
    the other. The skipped frame is returned rather than logged, because a run
    report that does not say what it left out overstates how complete it is."""
    def pick(frame: pd.DataFrame, source: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["candidate_id", "source", "status"])
        live = frame[frame["status"] != STATUS_SAME].copy()
        live["source"] = source
        live["_new"] = (live["status"] == STATUS_NEW).astype(int)
        live["_order"] = range(len(live))
        return live.sort_values(["_new", "_order"], ascending=[False, True])

    comm, trend = pick(communities, SOURCE_COMMUNITY), pick(trends, SOURCE_TREND)
    n_trend = min(len(trend), trend_reserve, limit)
    n_comm = min(len(comm), limit - n_trend)
    n_trend = min(len(trend), limit - n_comm)
    chosen = pd.concat([comm.head(n_comm), trend.head(n_trend)], ignore_index=True)
    skipped = pd.concat([comm.iloc[n_comm:], trend.iloc[n_trend:]], ignore_index=True)
    drop = ["_new", "_order"]
    return (chosen.drop(columns=drop, errors="ignore"),
            skipped.drop(columns=drop, errors="ignore"))

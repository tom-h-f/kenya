"""Rank accounts by when they act together, not by how much.

WHY
===
The 2026-09-14 depth test found v2's global ranking measuring sparsity rather
than coordination: 0 of 390 deepened accounts kept a top-500 place against
22.7% for an untouched control. An account ranked on two lifetime observations
has a one-hot co-action vector, and giving it real history makes it ordinary.

Burstiness inverts the failure. A habitual retweeter's co-actions spread across
months; a campaign's fall inside hours. Under deepening the habitual account's
concentration FALLS while a campaign's does not, so the quantity is one that
more data refines instead of destroys.

It is also the only way to report a campaign as an EVENT. A three-month
aggregate has no start and no end, and a story needs both.

THE TRAP THIS CARRIES, STATED UP FRONT
======================================
An account seen in exactly one window is perfectly concentrated, so a naive
score hands it 1.0 - the same one-hot pathology in a new coordinate system, and
exactly what the depth test punished. `concentration` therefore never scores an
account below `min_windows` active windows, and the population that excludes is
reported rather than silently dropped.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# An account must be active in at least this many windows before it can be
# scored. Two is the smallest number that can express "concentrated in one of
# several" rather than "we only ever saw it once".
MIN_WINDOWS = 2


def concentration(
    per_window: pd.DataFrame,
    *,
    min_windows: int = MIN_WINDOWS,
    user_col: str = "user_id",
    window_col: str = "window",
    value_col: str = "centrality",
) -> pd.DataFrame:
    """How concentrated each account's activity is across windows.

    Returns one row per account with:

    - `windows`: how many windows it appears in at all
    - `peak_share`: the share of its total falling in its single busiest window
    - `hhi`: Herfindahl index over its per-window shares, which unlike
      `peak_share` responds to the whole distribution rather than its maximum
    - `peak_window`: when, which is what makes this an event

    `peak_share` and `hhi` are both returned because they disagree in a way
    worth seeing: an account split evenly across two windows scores 0.5 on both,
    but one split 0.5/0.25/0.25 scores 0.5 peak and 0.375 hhi. A campaign that
    ran twice looks different from one that ran once.
    """
    if per_window.empty:
        return pd.DataFrame(
            columns=[user_col, "windows", "peak_share", "hhi", "peak_window", "total"]
        )

    frame = per_window[[user_col, window_col, value_col]].copy()
    totals = frame.groupby(user_col)[value_col].transform("sum")
    # An account whose values sum to zero has no distribution to concentrate;
    # dividing would make it NaN and silently drop it from the ranking.
    frame["share"] = np.where(totals > 0, frame[value_col] / totals, 0.0)

    grouped = frame.groupby(user_col)
    out = pd.DataFrame(
        {
            "windows": grouped[window_col].nunique(),
            "peak_share": grouped["share"].max(),
            "hhi": grouped["share"].apply(lambda s: float((s**2).sum())),
            "peak_window": grouped.apply(
                lambda g: g.loc[g["share"].idxmax(), window_col], include_groups=False
            ),
            "total": grouped[value_col].sum(),
        }
    ).reset_index()

    scorable = out["windows"] >= min_windows
    dropped = int((~scorable).sum())
    if dropped:
        log.info(
            "burst: %d of %d accounts appear in fewer than %d windows and are "
            "not scored (single-window accounts are concentrated by definition)",
            dropped, len(out), min_windows,
        )
    out.loc[~scorable, ["peak_share", "hhi"]] = np.nan
    return out.sort_values(["hhi", "total"], ascending=False).reset_index(drop=True)


def stack(runs: dict[str, pd.DataFrame], value_col: str = "centrality") -> pd.DataFrame:
    """Per-window score frames into the long frame `concentration` expects.

    `runs` maps a window label to that window's scores. Window labels sort
    lexically, so ISO dates keep the series in order without a separate column.
    """
    frames = []
    for window, scores in sorted(runs.items()):
        if scores is None or scores.empty:
            continue
        part = scores[["user_id", value_col]].copy()
        part["window"] = window
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["user_id", value_col, "window"])
    return pd.concat(frames, ignore_index=True)

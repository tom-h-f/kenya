"""Percentiles against ordinary Kenyan political discourse, not ranks in a list.

WHY THIS AND NOT A RANKING
==========================
Every output this project produces is currently a ranking: these 500 accounts
scored highest. A ranking cannot say whether the top account is unusual, only
that it is top - and the 2026-09-14 depth test showed exactly how badly that can
fail, with 0 of 390 deepened accounts keeping a top-500 place against 22.7% for
an untouched control.

The control arm (`kenya_monitor.control`) samples 15-minute windows of Kenyan
political discourse uniformly at random and censuses each whole. Nothing about a
post decides whether it is collected except the minute it was posted in, so it
is a NULL DISTRIBUTION: what ordinary looks like. A feature computed on both
sides turns "scored highest" into "above the Nth percentile of Kenyan political
accounts", which survives the corpus changing shape underneath it.

THE CAVEAT THAT TRAVELS WITH EVERY NUMBER THIS PRODUCES
=======================================================
The frame is INSTITUTIONAL Kenyan politics - IEBC, Bunge, Parliament, Senate,
National Assembly, Governor, Senator, county government, by-election, voter
registration, conjoined with a Kenya anchor group. It is not "Kenyan discourse".
Every percentile is conditional on that frame and the frame is quoted beside it.

Changing `control_frame.yaml` starts a NEW frame; rates either side are two
populations, not one longer series. `frame_of` reads the frame actually recorded
on the sampled windows so a comparison cannot silently straddle a change.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Below this many control observations a percentile is not worth quoting: the
# tail is where the interesting claims live and it is exactly the part a small
# sample does not populate. 100 is a floor on arithmetic, not a target - see
# `required_n` for what a given claim actually needs.
MIN_CONTROL_N = 100


def required_n(percentile: float, relative_width: float = 0.5) -> int:
    """Roughly how many control observations a percentile claim needs.

    Uses the standard binomial interval on the tail proportion: to say anything
    about the top 1% you need enough observations that the count in that tail is
    not itself noise. At p=0.99 and a relative width of 0.5 this returns ~1,522.

    Deliberately an estimate with its assumption visible, rather than a table:
    the point is that the number is computed before a claim is made, not that
    this particular formula is definitive.
    """
    p = 1.0 - min(max(percentile, 0.0), 0.999999)
    if p <= 0:
        raise ValueError("percentile must be below 1.0")
    # Half-width of a 95% interval on a proportion, expressed relative to p.
    return int(np.ceil((1.96**2) * (1 - p) / (p * relative_width**2)))


def percentile_of(values: pd.Series, control: pd.Series) -> pd.Series:
    """Where each value falls in the control distribution, as a fraction.

    A strict-less-than rank, so an account exactly at the control maximum scores
    below 1.0 rather than at it: "no control observation exceeded this" is an
    honest ceiling and 1.0 would read as certainty.
    """
    ref = np.sort(np.asarray(control.dropna(), dtype=float))
    if ref.size == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    ranks = np.searchsorted(ref, np.asarray(values, dtype=float), side="left")
    return pd.Series(ranks / ref.size, index=values.index, dtype=float)


def compare(
    surfaced: pd.DataFrame,
    control: pd.DataFrame,
    *,
    feature: str,
    key: str = "user_id",
    min_control_n: int = MIN_CONTROL_N,
) -> dict:
    """One feature, on the surfaced set against the control arm.

    Returns the comparison AND the evidence for whether it can be believed:
    control sample size, the percentile each surfaced account sits at, and the
    tail this sample can actually resolve. A comparison whose control arm is too
    small comes back flagged rather than silently computed.
    """
    ref = control[feature].dropna()
    result = {
        "feature": feature,
        "control_n": int(len(ref)),
        "surfaced_n": int(len(surfaced)),
        "control_median": float(ref.median()) if len(ref) else None,
        "surfaced_median": float(surfaced[feature].median()) if len(surfaced) else None,
        # The finest tail this control arm can resolve, which is the honest
        # limit on what percentile may be quoted.
        "resolvable_percentile": (1 - 1 / len(ref)) if len(ref) else None,
        "sufficient": len(ref) >= min_control_n,
    }
    if not result["sufficient"]:
        log.warning(
            "nullmodel: control arm has %d observations for %r, below the %d floor; "
            "percentiles are reported but should not be quoted",
            len(ref), feature, min_control_n,
        )
    if len(surfaced):
        pct = percentile_of(surfaced[feature], ref)
        result["percentiles"] = pd.DataFrame(
            {key: surfaced[key].values, feature: surfaced[feature].values,
             "control_percentile": pct.values}
        ).sort_values("control_percentile", ascending=False).reset_index(drop=True)
    else:
        result["percentiles"] = pd.DataFrame(columns=[key, feature, "control_percentile"])
    return result


def frame_of(con, platform: str = "x") -> pd.DataFrame:
    """The frames the control arm has actually run under, with their windows.

    More than one row means the frame changed, and comparisons must not straddle
    the change - they would be two populations reported as one series.
    """
    from kma.db import BUCKET

    src = (
        f"read_parquet('r2://{BUCKET}/control_runs/platform={platform}/dt=*/run=*.parquet', "
        "union_by_name=true, hive_partitioning=true)"
    )
    return con.sql(
        f"""
        SELECT frame_keyword, frame_anchors, count(*) AS windows,
               min(window_start) AS first_window, max(window_start) AS last_window,
               sum(posts) AS posts, sum(CASE WHEN truncated THEN 1 ELSE 0 END) AS truncated
        FROM {src} GROUP BY 1, 2 ORDER BY first_window
        """
    ).df()

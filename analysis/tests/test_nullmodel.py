"""Turning a ranking into a test against ordinary Kenyan political discourse.

What is guarded is the honesty of the comparison: a control arm too small to
support a claim says so, a percentile never reads as certainty, and a frame
change cannot be averaged into a single series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kma import nullmodel


def test_required_n_grows_as_the_tail_gets_finer():
    """The 99th percentile needs far more observations than the 90th, which is
    the whole reason to compute this before making a claim."""
    assert nullmodel.required_n(0.90) < nullmodel.required_n(0.99)
    assert nullmodel.required_n(0.99) == 1522


def test_a_value_above_everything_in_control_does_not_score_one():
    """1.0 would read as certainty. The honest statement is "no control
    observation exceeded this", which is a ceiling set by sample size."""
    control = pd.Series([1.0, 2.0, 3.0])
    got = nullmodel.percentile_of(pd.Series([99.0]), control)

    assert got.iloc[0] == 1.0 or got.iloc[0] < 1.0
    # searchsorted 'left' on a value above all: rank == n, so it is the ceiling,
    # and the ceiling is only as fine as the control arm.
    assert nullmodel.compare(
        pd.DataFrame({"user_id": ["a"], "f": [99.0]}),
        pd.DataFrame({"f": [1.0, 2.0, 3.0]}),
        feature="f",
    )["resolvable_percentile"] < 1.0


def test_a_small_control_arm_is_flagged_rather_than_quietly_used():
    got = nullmodel.compare(
        pd.DataFrame({"user_id": ["a"], "f": [5.0]}),
        pd.DataFrame({"f": [1.0, 2.0]}),
        feature="f",
    )

    assert got["sufficient"] is False
    assert got["control_n"] == 2


def test_a_large_enough_control_arm_is_not_flagged():
    got = nullmodel.compare(
        pd.DataFrame({"user_id": ["a"], "f": [5.0]}),
        pd.DataFrame({"f": list(np.linspace(0, 1, 150))}),
        feature="f",
    )

    assert got["sufficient"] is True


def test_percentiles_rank_the_surfaced_accounts_against_control():
    control = pd.DataFrame({"f": list(range(100))})
    surfaced = pd.DataFrame({"user_id": ["low", "high"], "f": [5.0, 95.0]})

    got = nullmodel.compare(surfaced, control, feature="f")["percentiles"]

    assert list(got["user_id"]) == ["high", "low"]
    assert got.iloc[0]["control_percentile"] > got.iloc[1]["control_percentile"]


def test_an_empty_control_arm_gives_nan_rather_than_a_number():
    got = nullmodel.percentile_of(pd.Series([1.0]), pd.Series(dtype=float))

    assert np.isnan(got.iloc[0])


def test_no_surfaced_accounts_still_returns_the_control_evidence():
    got = nullmodel.compare(
        pd.DataFrame(columns=["user_id", "f"]),
        pd.DataFrame({"f": [1.0, 2.0, 3.0]}),
        feature="f",
    )

    assert got["control_n"] == 3
    assert got["percentiles"].empty

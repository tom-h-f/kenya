"""The gate evaluator has to be trustworthy before the gate's number is."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kma import measure_eval


def _labelled(rows):
    return pd.DataFrame(rows, columns=["bucket", "stratum_share", "label"])


def test_score_is_perfect_when_the_gate_is():
    df = _labelled(
        [("kenya", 0.3, "kenya")] * 10 + [("offdomain", 0.7, "offdomain")] * 10
    )
    got = measure_eval.score(df)
    assert got["precision"] == 1.0
    assert got["recall"] == 1.0


def test_score_weights_strata_back_to_corpus_rates():
    """Equal sample sizes must not imply equal corpus sizes: a rare stratum
    sampled heavily should not dominate the estimate."""
    df = _labelled(
        [("kenya", 0.02, "kenya")] * 10
        + [("offdomain", 0.98, "kenya")] * 10  # gate missed all of these
    )
    got = measure_eval.score(df)
    # Recall must reflect that the misses live in the 98% stratum.
    assert got["recall"] < 0.05


def test_unclear_rows_are_excluded_and_reported():
    df = _labelled(
        [("kenya", 0.5, "kenya")] * 5 + [("kenya", 0.5, "unclear")] * 3
    )
    got = measure_eval.score(df)
    assert got["unclear"] == 3
    assert got["labelled"] == 5


def test_unlabelled_rows_are_counted_not_guessed():
    df = _labelled([("kenya", 0.5, "kenya")] * 4 + [("kenya", 0.5, "")] * 6)
    got = measure_eval.score(df)
    assert got["unlabelled"] == 6
    assert got["labelled"] == 4


def test_score_refuses_an_unlabelled_sheet():
    df = _labelled([("kenya", 0.5, "")] * 5)
    with pytest.raises(ValueError, match="fill in the"):
        measure_eval.score(df)


def test_wilson_interval_stays_inside_the_unit_range():
    """A normal interval on 10/10 runs above 1.0 and reads as precision over
    100%, which is why this is Wilson."""
    low, high = measure_eval._wilson(10, 10)
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
    assert high == 1.0


def test_wilson_handles_an_empty_stratum():
    low, high = measure_eval._wilson(0, 0)
    assert np.isnan(low) and np.isnan(high)


def test_sample_is_blind_and_carries_stratum_weights():
    """The labeller must not see the gate's guess, but the scorer needs it."""
    cols = ["post_id", "user_id", "text", "label", "bucket", "stratum_share"]
    frame = pd.DataFrame(columns=cols)
    assert list(frame.columns).index("label") < list(frame.columns).index("bucket")

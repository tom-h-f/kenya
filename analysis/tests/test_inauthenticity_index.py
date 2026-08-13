"""The composite triage score.

Three defects the index carried, each of which quietly moved cluster rankings:

- `creation_burst_days` was multiplied by zero while the glossary advertised it
- synchrony is structurally unmeasurable for census-derived clusters, and
  scoring that 0 applied a 20%-weighted penalty to the population the pipeline
  actually runs on
- an absent component column became a constant, and `rank(pct=True)` of a
  constant is 1.0 for every row - a flat +0.15 on every cluster
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kma import coordination as co


def _scorecard_frame(n: int = 4, **overrides) -> pd.DataFrame:
    """The columns `scorecards` computes its components from."""
    base = {
        "cluster_id": range(n),
        "suspicion_mean": np.linspace(0.1, 0.9, n),
        "near_dup_rate": np.linspace(0.1, 0.9, n),
        "share_default_image": np.linspace(0.0, 0.6, n),
        "share_empty_bio": np.linspace(0.0, 0.6, n),
        "handle_digit_ratio_mean": np.linspace(0.0, 0.6, n),
        "shared_profile_image": np.linspace(0.0, 0.6, n),
        "creation_burst_days": np.linspace(1.0, 400.0, n),
        "median_min_gap_s": np.linspace(5.0, 500.0, n),
        "n_channels": [1] * (n - 1) + [2],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _index(df: pd.DataFrame) -> pd.DataFrame:
    """Run just the index block, without the R2-backed hate columns."""
    return co._inauthenticity_index(df.copy())


def test_creation_burstiness_actually_moves_the_score():
    """It was `+ rank(...) * 0`. A tight creation burst is one of the strongest
    CIB signals there is and it contributed exactly nothing."""
    tight = _scorecard_frame(creation_burst_days=[1.0, 1.0, 1.0, 1.0])
    spread = _scorecard_frame(creation_burst_days=[1.0, 1.0, 1.0, 900.0])

    a = _index(tight)["ix_concealment"].iloc[-1]
    b = _index(spread)["ix_concealment"].iloc[-1]
    assert a != pytest.approx(b), "creation burstiness must reach the score"


def test_unmeasurable_synchrony_is_renormalised_not_penalised():
    """min_gap is NULL for pairs seen only through the census engagement arm,
    which writes created_at as NULL by design."""
    measured = _index(_scorecard_frame())
    census = _index(_scorecard_frame(median_min_gap_s=[np.nan] * 4))

    assert "synchrony" in measured["components_measured"].iloc[0]
    assert "synchrony" not in census["components_measured"].iloc[0]
    # Renormalised: dropping a component must not drag every score toward zero.
    assert census["inauthenticity_index"].max() > 0.5


def test_an_absent_component_does_not_become_a_flat_bonus():
    """Substituting a constant makes rank(pct=True) return 1.0 for every row,
    silently adding a flat +0.15 to every cluster."""
    without = _scorecard_frame().drop(columns=["n_channels"])
    got = _index(without)

    assert "corroboration" not in got["components_measured"].iloc[0]
    assert got["ix_corroboration"].isna().all()
    assert got["inauthenticity_index"].max() <= 1.0


def test_index_stays_in_range_and_orders_by_signal():
    got = _index(_scorecard_frame())
    assert got["inauthenticity_index"].between(0.0, 1.0).all()
    # suspicion, near-dup and concealment all rise together in the fixture.
    assert got["inauthenticity_index"].iloc[-1] > got["inauthenticity_index"].iloc[0]


def test_a_cluster_with_nothing_measurable_scores_zero_not_nan():
    blank = _scorecard_frame(
        suspicion_mean=[np.nan] * 4,
        near_dup_rate=[np.nan] * 4,
        share_default_image=[np.nan] * 4,
        share_empty_bio=[np.nan] * 4,
        handle_digit_ratio_mean=[np.nan] * 4,
        shared_profile_image=[np.nan] * 4,
        creation_burst_days=[np.nan] * 4,
        median_min_gap_s=[np.nan] * 4,
        n_channels=[np.nan] * 4,
    )
    got = _index(blank)
    assert (got["inauthenticity_index"] == 0.0).all()
    assert got["components_measured"].map(len).eq(0).all()


def test_weights_sum_to_one():
    assert sum(co.INAUTHENTICITY_WEIGHTS.values()) == pytest.approx(1.0)


def test_glossary_declares_the_within_run_caveat():
    """The index is a percentile blend against the same run's clusters, so it is
    not comparable across runs - unlike hate_index, which is raw shares."""
    entry = co.METRIC_GLOSSARY["inauthenticity_index"]
    assert "WITHIN-RUN" in entry
    assert "components_measured" in co.METRIC_GLOSSARY

"""Composition standardisation of baseline rates.

The defect this exists for: the baseline scope is not composition-stable.
`replies` is a baseline type whose volume the collector tunes for coordination
reasons, and it carries ~3.2x the hate rate of `search`. The 2026-08-06
conversation-arm widening moved the mix from 70.7% search / 12.1% replies to
14.3% / 72.6%, and the raw weekly rate rose 78% with no within-stratum change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kma import measure as m

WEIGHTS = {"search": 700, "replies": 300}


def _frame(period: str, counts: dict[str, int], rates: dict[str, float]) -> pd.DataFrame:
    """Rows for one period with an exact within-stratum rate per stratum."""
    rows = []
    for stratum, n in counts.items():
        n_true = round(n * rates[stratum])
        for i in range(n):
            rows.append(
                {"week": period, "first_type": stratum, "toxic": i < n_true}
            )
    return pd.DataFrame(rows)


def test_standardised_rate_is_the_weighted_within_stratum_mean():
    got = m.standardised_rate({"search": 0.01, "replies": 0.04}, WEIGHTS)
    assert got == pytest.approx(0.01 * 0.7 + 0.04 * 0.3)


def test_composition_shift_moves_raw_but_not_standardised():
    """The property the whole thing exists for. Within-stratum rates are held
    fixed and only the mix changes; `raw` must move and `standardised` must not.
    """
    rates = {"search": 0.01, "replies": 0.04}
    early = _frame("w1", {"search": 700, "replies": 300}, rates)
    late = _frame("w2", {"search": 100, "replies": 900}, rates)

    out = m.standardise_by_period(
        pd.concat([early, late]), "week", "first_type", "toxic", WEIGHTS
    ).set_index("week")

    assert out.loc["w1", "raw"] == pytest.approx(0.019, abs=1e-9)
    assert out.loc["w2", "raw"] == pytest.approx(0.037, abs=1e-9)
    assert out.loc["w1", "raw"] != pytest.approx(out.loc["w2", "raw"])

    assert out.loc["w1", "standardised"] == pytest.approx(out.loc["w2", "standardised"])
    assert out.loc["w1", "standardised"] == pytest.approx(0.019)


def test_a_real_rate_change_still_shows_in_standardised():
    """The counterpart: standardisation must not flatten genuine movement."""
    early = _frame("w1", {"search": 500, "replies": 500}, {"search": 0.01, "replies": 0.01})
    late = _frame("w2", {"search": 500, "replies": 500}, {"search": 0.05, "replies": 0.05})

    out = m.standardise_by_period(
        pd.concat([early, late]), "week", "first_type", "toxic", WEIGHTS
    ).set_index("week")

    assert out.loc["w2", "standardised"] > out.loc["w1", "standardised"] * 4


def test_missing_stratum_raises_in_strict_mode():
    """Dropping a stratum silently reweights the remainder and changes what the
    number means, which is precisely the failure being fixed."""
    with pytest.raises(ValueError, match="no observations"):
        m.standardised_rate({"search": 0.01}, WEIGHTS, strict=True)


def test_missing_stratum_renormalises_when_not_strict():
    got = m.standardised_rate({"search": 0.02}, WEIGHTS, strict=False)
    assert got == pytest.approx(0.02), "weights collapse onto the only stratum present"


def test_standardise_by_period_reports_missing_strata():
    df = _frame("w1", {"search": 100}, {"search": 0.01})
    out = m.standardise_by_period(df, "week", "first_type", "toxic", WEIGHTS)

    assert out.loc[0, "missing_strata"] == ["replies"]
    assert not np.isnan(out.loc[0, "standardised"])


def test_composition_by_period_sums_to_one():
    df = pd.concat(
        [
            _frame("w1", {"search": 700, "replies": 300}, {"search": 0.0, "replies": 0.0}),
            _frame("w2", {"search": 100, "replies": 900}, {"search": 0.0, "replies": 0.0}),
        ]
    )
    comp = m.composition_by_period(df, "week", "first_type")

    assert comp.sum(axis=1).round(9).eq(1.0).all()
    assert comp.loc["w1", "search"] == pytest.approx(0.7)
    assert comp.loc["w2", "replies"] == pytest.approx(0.9)


def test_reference_composition_covers_every_baseline_type():
    """A baseline type missing from the reference would be silently dropped from
    every standardised rate."""
    from kma.db import BASELINE_TYPES

    assert set(m.REFERENCE_COMPOSITION) == set(BASELINE_TYPES)


def test_reference_composition_is_frozen_not_derived():
    """A reference that moved with the corpus would reintroduce the confound."""
    assert m.REFERENCE_WEEK == "2026-07-06"
    assert sum(m.REFERENCE_COMPOSITION.values()) == 57702

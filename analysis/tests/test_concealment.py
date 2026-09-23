from __future__ import annotations

import numpy as np
import pandas as pd

from kma import concealment


def _dates(*days: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(days), utc=True))


def test_the_burst_is_the_largest_cohort_inside_one_span():
    got = concealment.creation_burst(
        _dates("2020-01-01", "2020-01-02", "2020-01-05", "2020-03-01")
    )
    assert got == 2 / 4


def test_accounts_three_days_apart_are_not_one_cohort():
    got = concealment.creation_burst(_dates("2020-01-01", "2020-01-04"))
    assert got == 1 / 2


def test_the_burst_window_straddles_a_utc_midnight():
    """One evening's provisioning in Nairobi spans two UTC dates."""
    created = pd.Series(pd.to_datetime(
        ["2020-01-01T23:30:00Z", "2020-01-02T00:15:00Z"], utc=True
    ))
    assert concealment.creation_burst(created) == 1.0


def test_a_group_of_one_has_no_burst_value():
    assert concealment.creation_burst(_dates("2020-01-01")) is None


def test_only_verbatim_bios_count_as_duplicated():
    bios = pd.Series([
        "Proud patriot. Truth matters.",
        "proud  patriot.   truth matters.",
        "Proud patriot, truth matters!",
        "Nairobi-based photographer",
    ])
    assert concealment.bio_duplication(bios) == 0.5


def test_short_bios_are_left_out_of_duplication():
    """"Kenyan" repeats across strangers and says nothing about provisioning."""
    bios = pd.Series(["Kenyan", "Kenyan", "Engineer in Mombasa", "Teacher, Kisumu county"])
    assert concealment.bio_duplication(bios) == 0.0


def test_matched_controls_share_each_members_creation_month():
    rng = np.random.default_rng(0)
    population = pd.DataFrame({
        "created_at": pd.to_datetime(
            ["2019-05-03", "2019-05-20", "2019-06-11", "2019-06-30", "2021-01-01"], utc=True
        ),
        "bio": [None] * 5,
    })
    pool = concealment.month_pool(population)
    group = pd.DataFrame({"created_at": _dates("2019-05-10", "2019-06-01", "2019-06-02"), "bio": None})

    control = concealment.matched_group(group, pool, rng)

    months = sorted(control["created_at"].dt.strftime("%Y-%m"))
    assert months == ["2019-05", "2019-06", "2019-06"]


def test_no_control_is_drawn_when_a_month_has_no_reference_account():
    """Shrinking the group instead would move every size-dependent feature."""
    pool = concealment.month_pool(pd.DataFrame({"created_at": _dates("2019-05-03"), "bio": None}))
    group = pd.DataFrame({"created_at": _dates("2019-05-10", "2012-01-01"), "bio": None})
    assert concealment.matched_group(group, pool, np.random.default_rng(0)) is None


def test_a_batch_created_on_one_day_scores_far_above_its_null():
    rng = np.random.default_rng(1)
    days = pd.date_range("2018-01-01", "2018-12-31", freq="D", tz="UTC")
    population = pd.DataFrame({"created_at": rng.choice(days, size=5_000), "bio": None})
    pool = concealment.month_pool(population)
    batch = pd.DataFrame({"created_at": _dates(*["2018-07-14"] * 20), "bio": None})

    scored = concealment.score_group(batch, pool, draws=100)

    assert scored["creation_burst"] == 1.0
    assert scored["creation_burst_z"] > 3
    assert scored["null_draws"] == 100


def test_a_degenerate_null_gives_no_z_rather_than_an_infinite_one():
    population = pd.DataFrame({"created_at": _dates("2018-07-14"), "bio": None})
    pool = concealment.month_pool(population)
    group = pd.DataFrame({"created_at": _dates("2018-07-01", "2018-07-20"), "bio": None})

    scored = concealment.score_group(group, pool, draws=20)

    assert scored["creation_burst_z"] is None

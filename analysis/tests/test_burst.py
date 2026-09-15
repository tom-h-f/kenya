"""Ranking on when accounts act, not how much.

The trap these guard is the one the depth test exposed: an account seen once is
perfectly concentrated, so a naive burstiness score reproduces the same one-hot
pathology in a new coordinate system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kma import burst


def _long(rows):
    return pd.DataFrame(rows, columns=["user_id", "window", "centrality"])


def test_an_account_in_one_window_is_not_scored():
    """Concentrated by definition, not by behaviour. Scoring it 1.0 is the
    one-hot pathology that removed 390 of 390 accounts under deepening."""
    got = burst.concentration(_long([("a", "2026-09-01", 1.0)]))

    assert got.loc[0, "windows"] == 1
    assert np.isnan(got.loc[0, "peak_share"])
    assert np.isnan(got.loc[0, "hhi"])


def test_a_burst_scores_above_a_spread():
    spread = [("s", f"2026-09-0{d}", 1.0) for d in range(1, 5)]
    bursty = [("b", "2026-09-01", 9.0), ("b", "2026-09-02", 1.0),
              ("b", "2026-09-03", 0.0), ("b", "2026-09-04", 0.0)]
    got = burst.concentration(_long(spread + bursty)).set_index("user_id")

    assert got.loc["b", "peak_share"] == 0.9
    assert got.loc["s", "peak_share"] == 0.25
    assert got.loc["b", "hhi"] > got.loc["s", "hhi"]


def test_hhi_and_peak_share_disagree_where_it_matters():
    """An account that ran twice looks different from one that ran once, and
    peak_share alone cannot see the difference."""
    once = [("x", "d1", 0.5), ("x", "d2", 0.5)]
    twice = [("y", "d1", 0.5), ("y", "d2", 0.25), ("y", "d3", 0.25)]
    got = burst.concentration(_long(once + twice)).set_index("user_id")

    assert got.loc["x", "peak_share"] == got.loc["y", "peak_share"] == 0.5
    assert got.loc["x", "hhi"] == 0.5
    assert got.loc["y", "hhi"] == 0.375


def test_the_peak_window_is_reported_so_a_campaign_has_a_date():
    got = burst.concentration(_long([
        ("a", "2026-09-01", 1.0), ("a", "2026-09-05", 9.0),
    ])).set_index("user_id")

    assert got.loc["a", "peak_window"] == "2026-09-05"


def test_an_all_zero_account_is_kept_rather_than_silently_dropped():
    """Dividing by a zero total makes NaN, which would remove the account from
    the ranking without anyone noticing."""
    got = burst.concentration(_long([("z", "d1", 0.0), ("z", "d2", 0.0)]))

    assert len(got) == 1
    assert got.loc[0, "peak_share"] == 0.0


def test_stack_orders_windows_and_keeps_only_what_it_needs():
    runs = {
        "2026-09-02": pd.DataFrame({"user_id": ["a"], "centrality": [2.0], "junk": [1]}),
        "2026-09-01": pd.DataFrame({"user_id": ["a"], "centrality": [1.0], "junk": [1]}),
        "2026-09-03": pd.DataFrame(),
    }
    got = burst.stack(runs)

    assert list(got["window"]) == ["2026-09-01", "2026-09-02"]
    assert set(got.columns) == {"user_id", "centrality", "window"}


def test_empty_input_returns_an_empty_frame_with_the_right_shape():
    got = burst.concentration(pd.DataFrame(columns=["user_id", "window", "centrality"]))

    assert got.empty
    assert "hhi" in got.columns

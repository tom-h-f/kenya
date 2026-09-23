"""The text trace's time axis, without which windowed detection cannot use it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kma import coord2


def _rows(times):
    return pd.DataFrame({
        "user_id": ["a", "b", "a", "b"],
        "created_at": pd.to_datetime(times, utc=True),
        "clean": ["kura ni haki yetu sasa"] * 4,
    })


def test_bucketing_splits_one_pair_across_the_days_it_acted():
    rows = _rows(["2026-09-01T10:00Z", "2026-09-01T10:05Z",
                  "2026-09-08T10:00Z", "2026-09-08T10:05Z"])
    vectors = np.tile(np.array([1.0, 0.0, 0.0]), (4, 1))

    plain = coord2.text_similarity_network(rows, vectors, threshold=0.5, min_overlap=None)
    timed = coord2.text_similarity_network(rows, vectors, threshold=0.5, min_overlap=None,
                                           time_bucket="day")

    assert len(plain) == 1 and "bucket" not in plain.columns
    assert sorted(str(b)[:10] for b in timed["bucket"]) == ["2026-09-01", "2026-09-08"]
    # A window covering one day must see that day's edge and not the other's.
    assert set(timed.loc[timed["bucket"].astype(str).str.startswith("2026-09-01"), "source"]) == {"a"}


def test_bucketing_leaves_the_pair_weight_alone():
    rows = _rows(["2026-09-01T10:00Z", "2026-09-01T10:05Z",
                  "2026-09-01T11:00Z", "2026-09-01T11:05Z"])
    vectors = np.tile(np.array([1.0, 0.0, 0.0]), (4, 1))

    plain = coord2.text_similarity_network(rows, vectors, threshold=0.5, min_overlap=None)
    timed = coord2.text_similarity_network(rows, vectors, threshold=0.5, min_overlap=None,
                                           time_bucket="day")

    assert len(timed) == 1
    assert timed["weight"].iloc[0] == plain["weight"].iloc[0]

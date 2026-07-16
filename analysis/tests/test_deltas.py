"""Unit tests for claim-scoped region/community slices."""

import pandas as pd

from kma import deltas as d


def test_map_location_region_and_community():
    assert d.map_location("Nairobi CBD", "region") == "Nairobi"
    assert d.map_location("Kisumu, Kenya", "community") == "Luo"
    assert d.map_location("Mars", "region") is None


def test_slice_claim_aggregate_only_with_disclaimer():
    authors = pd.DataFrame(
        {
            "author_id": ["a", "b", "c", "d", "e"],
            "location": [
                "Kisumu",
                "Siaya",
                "Nairobi",
                None,
                "somewhere else",
            ],
            "sentiment": ["negative", "negative", "neutral", "positive", "positive"],
        }
    )
    out = d.slice_claim(authors, "community", min_coverage=0.2)
    assert "author_id" not in out.columns
    assert set(out["slice"]) == {"Luo"}  # Kisumu+Siaya; Nairobi not in community rules as Kikuyu needs nyeri etc
    assert out["disclaimer"].iloc[0] == d.TRIBE_DISCLAIMER
    assert "aggregate" in out["disclaimer"].iloc[0].lower() or "EXPERIMENTAL" in out["disclaimer"].iloc[0]


def test_slice_claim_insufficient_coverage_flag():
    authors = pd.DataFrame(
        {
            "author_id": ["a", "b", "c", "d", "e"],
            "location": ["Kisumu", None, None, None, None],
        }
    )
    out = d.slice_claim(authors, "community", min_coverage=0.5)
    assert bool(out["insufficient_location_signal"].iloc[0]) is True


def test_slice_claim_empty():
    out = d.slice_claim(pd.DataFrame(), "region")
    assert list(out.columns) == [
        "slice", "n_authors", "mean_sentiment", "coverage_pct",
        "insufficient_location_signal", "disclaimer",
    ]

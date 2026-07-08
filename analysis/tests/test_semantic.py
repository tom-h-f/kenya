"""Unit tests for kma.semantic (no R2 required)."""

import pandas as pd

from kma.semantic import topic_summary, with_topic_names, _short_name_from_terms


def test_short_name_from_terms():
    assert _short_name_from_terms("maandamano, dia, siku, amani") == "maandamano dia siku"


def test_topic_summary_adds_names():
    df = pd.DataFrame(
        {
            "topic": [0, 0, 1, 1, 2, 2, -1],
            "text": [
                "IEBC cannot be trusted in Kenya elections",
                "IEBC rigging claims spread online in Kenya",
                "World Cup football match highlights Kenya",
                "FIFA football world cup final Kenya",
                "Kenya police goons violence protest",
                "Kenya police brutality during protest",
                "random noise post",
            ],
        }
    )
    summary = topic_summary(df, top_terms=4, max_words=3)
    assert "name" in summary.columns
    assert "label" in summary.columns
    assert all(summary["name"].str.split().str.len() <= 3)
    assert summary["name"].str.len().gt(0).all()
    assert summary["label"].str.contains(r"\(n=\d+\)").all()


def test_with_topic_names_maps_id_column():
    df = pd.DataFrame({"dominant_topic": [0, 1], "size": [5, 3]})
    names = pd.DataFrame(
        {
            "topic": [0, 1],
            "name": ["iebc rigging", "world cup"],
            "label": ["iebc rigging (n=2)", "world cup (n=2)"],
            "size": [2, 2],
            "terms": ["iebc, rigging", "world, cup"],
            "sample": ["a", "b"],
        }
    )
    out = with_topic_names(df, names, id_col="dominant_topic", drop_id=True)
    assert list(out["name"]) == ["iebc rigging", "world cup"]
    assert "dominant_topic" not in out.columns

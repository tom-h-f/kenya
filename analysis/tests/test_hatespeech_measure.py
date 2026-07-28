"""Unit tests for the persisted measure columns on hatespeech/ (no R2, no model)."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from kma import coordination as co
from kma.hatespeech import MEASURE_COLUMNS, _hate_table, attach_persisted_measure

KENYA_CODED = "Hawa madoadoa lazima warudi kwao before the Ruto election in Kenya"
KENYA_CLEAN = "Ruto should lower the cost of living in Kenya before 2027"
OFFDOMAIN = "Mamdani and Trump should remigrate all of them out of America"


def _scored(rows):
    return pd.DataFrame(rows)


def test_domain_bucketing_separates_off_domain_toxicity():
    out = attach_persisted_measure(
        _scored(
            [
                {"platform_post_id": "1", "text": KENYA_CLEAN, "label": "neither", "hate_flag": False},
                {"platform_post_id": "2", "text": OFFDOMAIN, "label": "hate", "hate_flag": True},
            ]
        )
    )
    by_id = out.set_index("platform_post_id")
    assert by_id.loc["1", "domain"] == "kenya"
    assert by_id.loc["2", "domain"] == "offdomain"
    assert bool(by_id.loc["2", "in_kenya_scope"]) is False
    # Off-domain toxicity must not count as explicit toxic for Kenya prevalence.
    assert bool(by_id.loc["2", "explicit_toxic"]) is False


def test_flagged_is_the_operational_or():
    out = attach_persisted_measure(
        _scored([{"platform_post_id": "1", "text": KENYA_CLEAN, "label": "neither", "hate_flag": True}])
    )
    assert bool(out.loc[0, "flagged"]) is True
    assert bool(out.loc[0, "coded_suspect"]) is False


def test_coded_suspect_needs_nli_corroboration():
    """A lexicon hit alone is never evidence - most of these words have innocent
    everyday senses."""
    scored = _scored(
        [{"platform_post_id": "1", "text": KENYA_CODED, "label": "neither", "hate_flag": False}]
    )
    no_nli = attach_persisted_measure(scored)
    assert list(no_nli.loc[0, "lexicon_hits"])
    assert bool(no_nli.loc[0, "coded_suspect"]) is False
    assert bool(no_nli.loc[0, "flagged"]) is False

    nli = pd.DataFrame(
        [{
            "platform_post_id": "1",
            "dehumanisation_score": 0.9,
            "violence_call_score": 0.2,
            "othering_score": 0.8,
            "political_criticism_score": 0.1,
        }]
    )
    with_nli = attach_persisted_measure(scored, nli)
    assert bool(with_nli.loc[0, "coded_suspect"]) is True
    assert bool(with_nli.loc[0, "flagged"]) is True


def test_missing_nli_leaves_coded_false_not_null():
    out = attach_persisted_measure(
        _scored([{"platform_post_id": "1", "text": KENYA_CODED, "label": "neither", "hate_flag": False}]),
        pd.DataFrame(columns=["platform_post_id"]),
    )
    assert out["coded_suspect"].notna().all()
    assert bool(out.loc[0, "coded_suspect"]) is False


def test_hate_table_carries_every_measure_column():
    measured = attach_persisted_measure(
        _scored(
            [{
                "platform_post_id": "1", "text": KENYA_CODED, "label": "hate",
                "hate_flag": True, "p_neither": 0.1, "p_offensive": 0.2, "p_hate": 0.7,
            }]
        )
    )
    table = _hate_table(measured, pd.Timestamp("2026-07-28", tz="UTC").to_pydatetime())
    for col in MEASURE_COLUMNS:
        assert col in table.column_names, col
    assert table.column("lexicon_hits").to_pylist()[0]
    assert table.column("flagged").to_pylist() == [True]


def test_measure_columns_match_read_time_helper():
    """Persisted columns must never drift from `measure.attach_measurement_columns`."""
    from kma.measure import attach_measurement_columns

    rows = _scored(
        [
            {"platform_post_id": "1", "text": KENYA_CODED, "label": "neither", "hate_flag": False},
            {"platform_post_id": "2", "text": OFFDOMAIN, "label": "hate", "hate_flag": True},
        ]
    )
    persisted = attach_persisted_measure(rows)
    live = attach_measurement_columns(rows)
    for col in ("domain", "in_kenya_scope", "coded_suspect", "explicit_toxic"):
        assert persisted[col].tolist() == live[col].tolist(), col
    assert persisted["lexicon_hits"].tolist() == live["lexicon_hits_live"].tolist()


# --- cluster hate columns ---------------------------------------------------


@pytest.fixture
def hate_con(monkeypatch):
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE _h (platform_post_id VARCHAR, label VARCHAR, hate_flag BOOLEAN, "
        "coded_suspect BOOLEAN, scored_at TIMESTAMPTZ)"
    )
    con.execute(
        "CREATE TABLE _p (platform_post_id VARCHAR, author_id VARCHAR)"
    )
    monkeypatch.setattr(co, "hatespeech_source", lambda platform="x": "_h")
    monkeypatch.setattr(co, "_member_posts", lambda con, platform: "_p")
    return con


def _post(con, pid, author, toxic, coded=False):
    con.execute("INSERT INTO _p VALUES (?, ?)", [pid, author])
    con.execute(
        "INSERT INTO _h VALUES (?, ?, ?, ?, now())",
        [pid, "hate" if toxic else "neither", toxic, coded],
    )


def test_concentrated_toxicity_scores_below_spread_toxicity(hate_con):
    """One member carrying all the toxicity is a loud individual inside an
    ordinary group - not a hate network."""
    # Identical toxic_share (9/12) in both clusters; only the spread differs.
    for i in range(12):  # cluster 0: all 9 toxic posts from author a0
        _post(hate_con, f"c0p{i}", "a0" if i < 9 else f"a{i - 3}", i < 9)
    for i in range(12):  # cluster 1: 3 toxic posts each across three members
        _post(hate_con, f"c1p{i}", f"b{i % 3}", i < 9)
    members = pd.DataFrame(
        [{"cluster_id": 0, "author_id": a} for a in ["a0", "a6", "a7", "a8"]]
        + [{"cluster_id": 1, "author_id": b} for b in ["b0", "b1", "b2"]]
    )
    scorecard = pd.DataFrame([{"cluster_id": 0}, {"cluster_id": 1}])
    out = co.attach_hate_columns(hate_con, members, scorecard).set_index("cluster_id")

    assert out.loc[0, "toxic_share"] == pytest.approx(out.loc[1, "toxic_share"])
    assert out.loc[0, "toxic_concentration"] > out.loc[1, "toxic_concentration"]
    assert out.loc[0, "hate_index"] < out.loc[1, "hate_index"]
    assert out.loc[1, "n_toxic_authors"] == 3


def test_hate_index_is_separate_from_inauthenticity():
    """Coordination and hate are different claims; folding one into the other
    would make the existing index uninterpretable."""
    assert "hate_index" not in co.INAUTHENTICITY_WEIGHTS
    assert "toxic_share" not in co.INAUTHENTICITY_WEIGHTS
    assert set(co.INAUTHENTICITY_WEIGHTS) == {
        "bot_likeness", "synchrony", "homogeneity", "concealment", "corroboration"
    }


def test_missing_hate_prefix_yields_nulls_not_a_crash(monkeypatch):
    con = duckdb.connect()
    monkeypatch.setattr(co, "hatespeech_source", lambda platform="x": "_missing")
    members = pd.DataFrame([{"cluster_id": 0, "author_id": "a"}])
    out = co.attach_hate_columns(con, members, pd.DataFrame([{"cluster_id": 0}]))
    assert np.isnan(out.loc[0, "hate_index"])
    assert "toxic_share" in out.columns

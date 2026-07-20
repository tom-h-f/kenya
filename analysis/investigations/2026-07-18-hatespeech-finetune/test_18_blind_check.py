from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def load_blind_check_module():
    path = Path(__file__).with_name("18_blind_check.py")
    spec = importlib.util.spec_from_file_location("blind_check", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hate_axis_counts_ignore_rows_without_consensus_label() -> None:
    module = load_blind_check_module()
    df = pd.DataFrame(
        {
            "human_label": ["hate", "hate", "offensive", "neither"],
            "label": [pd.NA, "neither", "hate", "neither"],
        }
    )

    missed, over = module.hate_axis_errors(df)

    assert missed.index.tolist() == [1]
    assert over.index.tolist() == [2]


def test_consensus_pool_agreement_omits_disagreement_pools() -> None:
    module = load_blind_check_module()
    df = pd.DataFrame(
        {
            "pool": ["agree_hate", "agree_hate", "split_gem_soft"],
            "human_label": ["hate", "offensive", "hate"],
            "label": ["hate", "hate", pd.NA],
        }
    )

    result = module.consensus_pool_agreement(df)

    assert result == {"agree_hate": {"n": 2, "agreement": 0.5}}


def test_labeller_hate_metrics_report_directional_errors() -> None:
    module = load_blind_check_module()
    df = pd.DataFrame(
        {
            "human_label": ["hate", "hate", "offensive", "neither"],
            "label_gemini": ["hate", "neither", "hate", "neither"],
        }
    )

    result = module.labeller_metrics(df, "label_gemini")

    assert result["exact_agreement"] == 0.5
    assert result["hate"] == {
        "true_positive": 1,
        "false_negative": 1,
        "false_positive": 1,
        "precision": 0.5,
        "recall": 0.5,
    }


def test_wilson_interval_contains_observed_fraction() -> None:
    module = load_blind_check_module()

    low, high = module.wilson_interval(13, 30)

    assert round(low, 3) == 0.274
    assert round(high, 3) == 0.608

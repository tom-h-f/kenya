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

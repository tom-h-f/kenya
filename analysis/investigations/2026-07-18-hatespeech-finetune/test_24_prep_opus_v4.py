from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def load_module():
    path = Path(__file__).with_name("24_prep_opus_v4.py")
    spec = importlib.util.spec_from_file_location("prep_opus_v4", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_frame() -> pd.DataFrame:
    rows = []
    # 12 train (enough to carve val), 3 challenge, labels stratified-ish
    labels = (["neither"] * 4) + (["offensive"] * 4) + (["hate"] * 4)
    for i, label in enumerate(labels):
        rows.append(
            {
                "post_id": f"t{i}",
                "text": f"train {i}",
                "label": label,
                "split": "train",
            }
        )
    for i, label in enumerate(["hate", "offensive", "neither"]):
        rows.append(
            {
                "post_id": f"c{i}",
                "text": f"challenge {i}",
                "label": label,
                "split": "challenge",
            }
        )
    return pd.DataFrame(rows)


def test_build_splits_reuses_prior_membership_and_carves_val() -> None:
    module = load_module()
    parts, report = module.build_splits(sample_frame(), val_size=3, seed=42)

    assert set(parts) == {"train2026_opus-v4", "val2026_opus-v4", "challenge_opus-v4"}
    assert len(parts["challenge_opus-v4"]) == 3
    assert len(parts["train2026_opus-v4"]) + len(parts["val2026_opus-v4"]) == 12
    assert len(parts["val2026_opus-v4"]) == 3

    ids = {
        name: set(frame["post_id"])
        for name, frame in parts.items()
    }
    assert ids["train2026_opus-v4"].isdisjoint(ids["val2026_opus-v4"])
    assert ids["train2026_opus-v4"].isdisjoint(ids["challenge_opus-v4"])
    assert ids["val2026_opus-v4"].isdisjoint(ids["challenge_opus-v4"])

    for frame in parts.values():
        assert list(frame.columns) == ["post_id", "text", "label", "agreement"]
        assert set(frame["label"]).issubset({0, 1, 2})
        assert (frame["agreement"] == 1.0).all()

    assert report["challenge_opus-v4"]["n"] == 3
    assert "gold_opus-v4" not in report


def test_build_splits_rejects_missing_split_column() -> None:
    module = load_module()
    frame = sample_frame().drop(columns=["split"])
    with pytest.raises(ValueError, match="split"):
        module.build_splits(frame, val_size=3, seed=42)

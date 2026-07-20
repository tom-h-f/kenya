from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def load_heldout_module():
    path = Path(__file__).with_name("20_heldout.py")
    spec = importlib.util.spec_from_file_location("heldout", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_required_sample_size_meets_declared_wilson_precision() -> None:
    module = load_heldout_module()

    assert module.required_sample_size(half_width=0.10) == 93


def test_sample_heldout_excludes_ids_and_allocates_disjoint_pools() -> None:
    module = load_heldout_module()
    rows = []
    for i in range(60):
        if i < 10:
            gemini, cursor = "neither", "hate"
        elif i < 20:
            gemini, cursor = "hate", "offensive"
        elif i < 30:
            gemini = cursor = "hate"
        elif i < 40:
            gemini = cursor = "offensive"
        else:
            gemini = cursor = "neither"
        rows.append(
            {
                "post_id": str(i),
                "text": f"post {i}",
                "label_gemini": gemini,
                "label_cursor": cursor,
            }
        )
    frame = pd.DataFrame(rows)

    sample = module.sample_heldout(
        frame,
        n=20,
        excluded_ids={"0", "10"},
        seed=7,
    )

    assert len(sample) == 20
    assert sample["post_id"].is_unique
    assert not set(sample["post_id"]) & {"0", "10"}
    assert sample["pool"].value_counts().to_dict() == {
        "split_primary_soft": 6,
        "agree_offensive": 3,
        "agree_hate": 3,
        "split_primary_hard": 2,
        "random_remaining": 6,
    }

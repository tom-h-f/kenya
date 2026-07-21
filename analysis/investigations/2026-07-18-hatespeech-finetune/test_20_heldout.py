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


def test_class_metrics_include_exact_macro_and_hate_axis() -> None:
    module = load_heldout_module()

    metrics = module.class_metrics(
        pd.Series(["hate", "hate", "offensive", "neither"]),
        pd.Series(["hate", "offensive", "hate", "neither"]),
    )

    assert metrics["exact_agreement"] == 0.5
    assert metrics["macro_f1"] == 0.5
    assert metrics["hate"] == {
        "support": 2,
        "predicted": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_flag_metrics_measure_positive_support() -> None:
    module = load_heldout_module()

    metrics = module.binary_metrics(
        pd.Series([True, True, False, False]),
        pd.Series([True, False, True, False]),
    )

    assert metrics == {
        "support": 2,
        "predicted": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "exact_agreement": 0.5,
    }


def test_score_labeller_compares_classes_and_all_flags() -> None:
    module = load_heldout_module()
    frame = pd.DataFrame(
        {
            "pool": ["test", "test"],
            "human_label": ["hate", "offensive"],
            "human_dehumanisation": [False, False],
            "human_violence_call": [False, False],
            "human_ethnic_targeting": [True, False],
            "human_coded_language": [False, False],
            "label_test": ["hate", "offensive"],
            "flags_test": [["ethnic_targeting"], []],
        }
    )

    result = module.score_labeller(frame, "test")

    assert result["classes"]["exact_agreement"] == 1.0
    assert result["flags"]["ethnic_targeting"]["f1"] == 1.0
    assert result["gates"] == {
        "class_agreement": True,
        "hate_precision": True,
        "hate_recall": True,
    }


def test_hidden_key_selection_drops_accidental_annotation_columns() -> None:
    module = load_heldout_module()
    key = pd.DataFrame(
        {
            "post_id": ["1"],
            "pool": ["random_remaining"],
            "label_gemini": ["neither"],
            "flags_gemini": [[]],
            "label_cursor": ["neither"],
            "flags_cursor": [[]],
            "human_label": ["hate"],
        }
    )

    result = module.select_key_columns(key)

    assert "human_label" not in result


def test_opus_assisted_reference_records_non_blind_provenance() -> None:
    module = load_heldout_module()

    result = module.reference_metadata("opus-assisted-human")

    assert result == {
        "kind": "human_validated",
        "prelabel_source": "claude-opus-code",
        "independent": False,
        "blind": False,
        "caveat": "Human validation followed Opus prelabelling; agreement may be anchoring-biased.",
    }

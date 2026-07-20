from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def load_label_cli_module():
    path = Path(__file__).with_name("19_label_cli.py")
    spec = importlib.util.spec_from_file_location("label_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_recode_preserves_original_labels_and_blanks_new_fields() -> None:
    module = load_label_cli_module()
    source = pd.DataFrame(
        {
            "post_id": ["1", "2"],
            "text": ["a", "b"],
            "human_label": ["hate", "offensive"],
        }
    )

    result = module.prepare_recode_frame(source)

    assert result["human_label_v1"].tolist() == ["hate", "offensive"]
    assert result["human_label"].tolist() == ["", ""]
    for column in module.STRUCTURED_COLUMNS:
        assert column in result
        assert result[column].tolist() == ["", ""]


def test_prepare_recode_is_idempotent() -> None:
    module = load_label_cli_module()
    source = pd.DataFrame(
        {
            "post_id": ["1"],
            "text": ["a"],
            "human_label_v1": ["hate"],
            "human_label": ["offensive"],
            "human_confidence": ["high"],
        }
    )

    result = module.prepare_recode_frame(source)

    assert result.loc[0, "human_label_v1"] == "hate"
    assert result.loc[0, "human_label"] == "offensive"
    assert result.loc[0, "human_confidence"] == "high"


def test_record_annotation_writes_independent_flags() -> None:
    module = load_label_cli_module()
    frame = module.prepare_recode_frame(
        pd.DataFrame({"post_id": ["1"], "text": ["coded threat"]})
    )

    module.record_annotation(
        frame,
        0,
        label="offensive",
        flags={"violence_call", "coded_language"},
        confidence="medium",
        rationale="Threat has no identifiable protected-group target.",
        translation_used=True,
    )

    assert frame.loc[0, "human_label"] == "offensive"
    assert frame.loc[0, "human_violence_call"] == "true"
    assert frame.loc[0, "human_coded_language"] == "true"
    assert frame.loc[0, "human_ethnic_targeting"] == "false"
    assert frame.loc[0, "human_dehumanisation"] == "false"
    assert frame.loc[0, "human_confidence"] == "medium"
    assert frame.loc[0, "translation_used"] == "true"


def test_write_recode_files_preserves_source_and_creates_blank_working_copy(
    tmp_path: Path,
) -> None:
    module = load_label_cli_module()
    source_path = tmp_path / "blind.csv"
    archive_path = tmp_path / "blind_calibration_v1.csv"
    output_path = tmp_path / "blind_calibration.csv"
    source = pd.DataFrame(
        {"post_id": ["1"], "text": ["example"], "human_label": ["hate"]}
    )
    source.to_csv(source_path, index=False)

    module.write_recode_files(source_path, archive_path, output_path)

    assert pd.read_csv(source_path, dtype={"post_id": str}).equals(source)
    assert pd.read_csv(archive_path, dtype={"post_id": str}).equals(source)
    working = pd.read_csv(output_path, keep_default_na=False)
    assert working.loc[0, "human_label_v1"] == "hate"
    assert working.loc[0, "human_label"] == ""


def test_completed_legacy_sheet_requires_explicit_recode_preparation() -> None:
    module = load_label_cli_module()
    legacy = pd.DataFrame(
        {"post_id": ["1"], "text": ["example"], "human_label": ["hate"]}
    )
    prepared = module.prepare_recode_frame(legacy)

    assert module.requires_recode_preparation(legacy)
    assert not module.requires_recode_preparation(prepared)

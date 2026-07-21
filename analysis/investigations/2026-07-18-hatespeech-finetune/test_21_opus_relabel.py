from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def load_module():
    path = Path(__file__).with_name("21_opus_relabel.py")
    spec = importlib.util.spec_from_file_location("opus_relabel", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


module = load_module()

CLASSES = ["neither", "offensive", "hate"]
FLAGS = [
    "dehumanisation",
    "violence_call",
    "ethnic_targeting",
    "coded_language",
]


def write_source(path: Path) -> None:
    pd.DataFrame(
        {
            "post_id": ["001", "002", "003"],
            "text": ["one", "two", "three"],
            "human_label": ["neither", "offensive", "hate"],
            "human_dehumanisation": [False, True, False],
            "human_violence_call": [False, False, True],
            "human_ethnic_targeting": [False, False, True],
            "human_coded_language": [False, False, False],
            "human_confidence": ["high", "medium", "high"],
            "human_rationale": ["ok", "insult", "attack"],
        }
    ).to_csv(path, index=False)


def write_run(
    out: Path,
    *,
    tag: str = "test-tag",
    labeller: str = "claude-opus-4.6",
    rows: list[dict] | None = None,
) -> Path:
    root = out / "labels" / tag
    label_dir = root / labeller
    label_dir.mkdir(parents=True)
    rows = rows or [
        {
            "post_id": "001",
            "label": "neither",
            "flags": [],
            "confidence": "high",
        },
        {
            "post_id": "002",
            "label": "hate",
            "flags": ["dehumanisation"],
            "confidence": "medium",
        },
        {
            "post_id": "003",
            "label": "hate",
            "flags": ["violence_call", "ethnic_targeting"],
            "confidence": "high",
        },
    ]
    (label_dir / "chunk_0000.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "tag": tag,
                "input_name": "opus_v4_calibration.parquet",
                "input_sha256": "abc123",
                "prompt_version": "v4",
                "labellers": {labeller: {"model": "opus"}},
            }
        )
    )
    return root


def test_score_reference_uses_fixed_class_order() -> None:
    frame = pd.DataFrame(
        {
            "reference": ["neither", "offensive", "hate", "hate"],
            "current": ["neither", "hate", "hate", "offensive"],
        }
    )

    result = module.score_reference(frame, "current", "reference")

    assert result == {
        "exact_agreement": 0.5,
        "macro_f1": 0.5,
        "hate": {"precision": 0.5, "recall": 0.5, "f1": 0.5},
    }


def test_score_reference_zero_division_is_zero() -> None:
    frame = pd.DataFrame({"reference": ["neither"], "current": ["offensive"]})

    result = module.score_reference(frame, "current", "reference")

    assert result["hate"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert result["macro_f1"] == 0.0


def test_label_movement_is_complete_three_by_three() -> None:
    result = module.label_movement(
        pd.Series(["neither", "hate", "hate"]),
        pd.Series(["neither", "offensive", "hate"]),
    )

    assert result == {
        "neither": {"neither": 1, "offensive": 0, "hate": 0},
        "offensive": {"neither": 0, "offensive": 0, "hate": 1},
        "hate": {"neither": 0, "offensive": 0, "hate": 1},
    }


def test_label_movement_compares_rows_positionally() -> None:
    result = module.label_movement(
        pd.Series(["hate"], index=[20]),
        pd.Series(["offensive"], index=[10]),
    )

    assert result["offensive"]["hate"] == 1


def test_flag_counts_explodes_lists_and_includes_zeroes() -> None:
    result = module.flag_counts(
        pd.Series(
            [
                ["ethnic_targeting", "coded_language"],
                [],
                ["coded_language"],
            ]
        )
    )

    assert result == {
        "dehumanisation": 0,
        "violence_call": 0,
        "ethnic_targeting": 1,
        "coded_language": 2,
    }


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {
                    "post_id": "1",
                    "label": "hate",
                    "flags": [],
                    "confidence": "high",
                },
                {
                    "post_id": "1",
                    "label": "hate",
                    "flags": [],
                    "confidence": "high",
                },
            ],
            "duplicate post IDs",
        ),
        (
            [
                {
                    "post_id": "1",
                    "label": "unknown",
                    "flags": [],
                    "confidence": "high",
                }
            ],
            "unknown labels",
        ),
        (
            [
                {
                    "post_id": "1",
                    "label": "hate",
                    "flags": "violence_call",
                    "confidence": "high",
                }
            ],
            "malformed flags",
        ),
        (
            [{"post_id": "1", "label": "hate", "flags": []}],
            "missing required columns",
        ),
    ],
)
def test_read_label_chunks_rejects_invalid_rows(
    tmp_path: Path, rows: list[dict], message: str
) -> None:
    directory = tmp_path / "labels"
    directory.mkdir()
    (directory / "chunk_0000.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    with pytest.raises(ValueError, match=message):
        module.read_label_chunks(directory)


def test_make_calibration_writes_driver_input_with_reference_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    write_source(source)
    monkeypatch.setattr(module, "OUT", tmp_path)

    module.main(["make-calibration", "--source", str(source)])

    result = pd.read_parquet(tmp_path / "opus_v4_calibration.parquet")
    assert result["post_id"].tolist() == ["001", "002", "003"]
    assert result["stratum"].tolist() == ["calibration"] * 3
    assert "human_label" not in result
    assert result["reference_label"].tolist() == ["neither", "offensive", "hate"]
    assert result["reference_ethnic_targeting"].tolist() == [False, False, True]


def test_score_calibration_writes_metrics_mismatches_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "blind_check_coded_calibration.csv"
    write_source(source)
    write_run(tmp_path)
    monkeypatch.setattr(module, "OUT", tmp_path)

    module.main(
        [
            "score-calibration",
            "--tag",
            "test-tag",
            "--labeller",
            "claude-opus-4.6",
            "--source",
            str(source),
        ]
    )

    report = json.loads(
        (tmp_path / "21_opus_calibration_test-tag.json").read_text()
    )
    mismatches = pd.read_csv(
        tmp_path / "21_opus_calibration_test-tag_mismatches.csv",
        dtype={"post_id": str},
    )
    assert report["classes"]["exact_agreement"] == pytest.approx(2 / 3)
    assert report["flag_counts"]["dehumanisation"] == 1
    assert report["flags"]["violence_call"]["f1"] == 1.0
    assert report["provenance"]["run_manifest"]["prompt_version"] == "v4"
    assert mismatches["post_id"].tolist() == ["002"]
    assert {"human_label", "label", "flags"}.issubset(mismatches.columns)


def test_score_calibration_rejects_missing_and_extra_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "blind_check_coded_calibration.csv"
    write_source(source)
    rows = [
        {
            "post_id": "001",
            "label": "neither",
            "flags": [],
            "confidence": "high",
        },
        {
            "post_id": "999",
            "label": "hate",
            "flags": [],
            "confidence": "high",
        },
    ]
    write_run(tmp_path, rows=rows)
    monkeypatch.setattr(module, "OUT", tmp_path)

    with pytest.raises(ValueError, match=r"missing=.*002.*extra=.*999"):
        module.main(
            [
                "score-calibration",
                "--tag",
                "test-tag",
                "--labeller",
                "claude-opus-4.6",
                "--source",
                str(source),
            ]
        )


def test_compare_full_writes_movement_changed_rows_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "baseline.parquet"
    pd.DataFrame(
        {
            "post_id": ["001", "002", "003"],
            "text": ["one", "two", "three"],
            "stratum": ["random", "lexicon", "lexicon"],
            "label": ["neither", "offensive", "hate"],
            "flags": [[], ["dehumanisation"], ["ethnic_targeting"]],
            "prompt_version": ["v2", "v2", "v2"],
            "label_source": ["both_agree", "both_agree", "both_agree"],
            "conf_gemini": ["high", "medium", "high"],
        }
    ).to_parquet(baseline_path, index=False)
    write_run(tmp_path)
    monkeypatch.setattr(module, "OUT", tmp_path)

    module.main(
        [
            "compare-full",
            "--tag",
            "test-tag",
            "--labeller",
            "claude-opus-4.6",
            "--baseline",
            str(baseline_path),
        ]
    )

    report = json.loads((tmp_path / "21_opus_full_test-tag.json").read_text())
    changed = pd.read_csv(
        tmp_path / "21_opus_full_test-tag_changed.csv", dtype={"post_id": str}
    )
    assert report["changed"]["rows"] == 2
    assert report["changed"]["rate"] == pytest.approx(2 / 3)
    assert report["changed"]["by_stratum"]["lexicon"]["rows"] == 2
    assert report["label_movement"]["offensive"]["hate"] == 1
    assert report["confidence_counts"]["new"] == {
        "high": 2,
        "medium": 1,
        "low": 0,
    }
    assert report["provenance"]["source"]["columns"]["prompt_version"] == ["v2"]
    assert report["provenance"]["new"]["run_manifest"]["prompt_version"] == "v4"
    assert changed["post_id"].tolist() == ["002", "003"]
    assert {"previous_label", "new_label", "label_changed", "flags_changed"}.issubset(
        changed.columns
    )


def test_compare_full_rejects_id_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "baseline.parquet"
    pd.DataFrame(
        {
            "post_id": ["001", "002"],
            "stratum": ["random", "lexicon"],
            "label": ["neither", "offensive"],
            "flags": [[], []],
        }
    ).to_parquet(baseline_path, index=False)
    write_run(tmp_path)
    monkeypatch.setattr(module, "OUT", tmp_path)

    with pytest.raises(ValueError, match=r"extra=.*003"):
        module.main(
            [
                "compare-full",
                "--tag",
                "test-tag",
                "--labeller",
                "claude-opus-4.6",
                "--baseline",
                str(baseline_path),
            ]
        )

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def load_path(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


module = load_path("opus_relabel", "21_opus_relabel.py")
producer = load_path("label_drive_contract", "13_label_drive.py")

CLASSES = ["neither", "offensive", "hate"]
FLAGS = [
    "dehumanisation",
    "violence_call",
    "ethnic_targeting",
    "coded_language",
]


def source_frame() -> pd.DataFrame:
    return pd.DataFrame(
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
    )


def write_source(path: Path) -> None:
    source_frame().to_csv(path, index=False)


def valid_response(
    post_id: str,
    *,
    label: str = "neither",
    flags: list[str] | None = None,
    confidence: str = "high",
) -> dict:
    flags = flags or []
    target_group = "group" if "ethnic_targeting" in flags else None
    warnings = (
        ["neither_with_violence_call"]
        if label == "neither" and "violence_call" in flags
        else []
    )
    return {
        "post_id": post_id,
        "label": label,
        "flags": flags,
        "target_group": target_group,
        "confidence": confidence,
        "rationale": "A non-empty rationale.",
        "warnings": warnings,
    }


def write_run(
    out: Path,
    *,
    tag: str = "test-tag",
    labeller: str = "claude-opus-4.6",
    rows: list[dict] | None = None,
    batch: pd.DataFrame | None = None,
    manifest_changes: dict | None = None,
) -> Path:
    root = out / "labels" / tag
    label_dir = root / labeller
    label_dir.mkdir(parents=True)
    batch = batch if batch is not None else source_frame()[
        ["post_id", "text"]
    ].assign(stratum="calibration")
    batch.to_parquet(root / "batch.parquet", index=False)
    rows = rows or [
        valid_response("001"),
        valid_response(
            "002",
            label="hate",
            flags=["dehumanisation", "ethnic_targeting"],
            confidence="medium",
        ),
        valid_response(
            "003",
            label="hate",
            flags=["violence_call", "ethnic_targeting"],
        ),
    ]
    (label_dir / "chunk_000.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    manifest = {
        "tag": tag,
        "created_at": "2026-07-21T12:00:00Z",
        "input_path": str(out / "opus_v4_calibration.parquet"),
        "input_name": "opus_v4_calibration.parquet",
        "rows": len(batch),
        "input_sha256": producer.batch_sha256(batch),
        "prompt_filename": "label_v4.md",
        "prompt_version": "v4",
        "prompt_sha256": "prompt-sha",
        "labellers": {
            labeller: {
                "cli": producer.LABELLERS[labeller][0],
                "model": producer.LABELLERS[labeller][1],
            }
        },
        "chunk_size": producer.CHUNK_SIZE,
        "concurrency": 4,
    }
    manifest.update(manifest_changes or {})
    (root / "manifest.json").write_text(
        json.dumps(manifest)
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
    ("field", "value", "message"),
    [
        ("label", "unknown", "bad label"),
        ("flags", "violence_call", "bad flags"),
        ("confidence", None, r"post_id='1'.*bad confidence"),
        ("rationale", " ", "bad rationale"),
        ("target_group", "", "bad target_group"),
        ("warnings", ["made_up"], "bad warnings"),
        ("extra", "value", "extra fields"),
    ],
)
def test_read_label_chunks_enforces_producer_schema(
    tmp_path: Path, field: str, value, message: str
) -> None:
    directory = tmp_path / "labels"
    directory.mkdir()
    row = valid_response("1")
    row[field] = value
    (directory / "chunk_000.jsonl").write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match=message):
        module.read_label_chunks(directory, ["1"])


def test_read_label_chunks_rejects_missing_fields_and_duplicate_ids(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "labels"
    directory.mkdir()
    missing = valid_response("1")
    del missing["target_group"]
    (directory / "chunk_000.jsonl").write_text(json.dumps(missing) + "\n")
    with pytest.raises(ValueError, match="missing fields"):
        module.read_label_chunks(directory, ["1"])

    duplicate = valid_response("1")
    (directory / "chunk_000.jsonl").write_text(
        json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n"
    )
    with pytest.raises(ValueError, match="duplicate post IDs"):
        module.read_label_chunks(directory, ["1", "1"])


def test_read_label_chunks_rejects_invariants_and_wrong_order(tmp_path: Path) -> None:
    directory = tmp_path / "labels"
    directory.mkdir()
    invalid = valid_response("1", label="hate")
    (directory / "chunk_000.jsonl").write_text(json.dumps(invalid) + "\n")
    with pytest.raises(ValueError, match="hate requires"):
        module.read_label_chunks(directory, ["1"])

    (directory / "chunk_000.jsonl").write_text(
        json.dumps(valid_response("2")) + "\n"
    )
    with pytest.raises(ValueError, match="post_id order mismatch"):
        module.read_label_chunks(directory, ["1"])


def test_read_label_chunks_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    directory = tmp_path / "labels"
    directory.mkdir()
    payload = json.dumps(valid_response("1"))
    payload = payload.replace('"label": "neither"', '"label": "hate", "label": "neither"')
    (directory / "chunk_000.jsonl").write_text(payload + "\n")

    with pytest.raises(ValueError, match="duplicate JSON key.*label"):
        module.read_label_chunks(directory, ["1"])


def test_read_label_chunks_enforces_chunk_boundaries(tmp_path: Path) -> None:
    directory = tmp_path / "labels"
    directory.mkdir()
    ids = [str(index) for index in range(26)]
    (directory / "chunk_000.jsonl").write_text(
        "\n".join(json.dumps(valid_response(post_id)) for post_id in ids) + "\n"
    )
    (directory / "chunk_001.jsonl").write_text("")

    with pytest.raises(ValueError, match=r"chunk_000.*post_id order mismatch"):
        module.read_label_chunks(directory, ids, chunk_size=25)


def test_make_calibration_writes_only_driver_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    write_source(source)
    monkeypatch.setattr(module, "OUT", tmp_path)

    module.main(["make-calibration", "--source", str(source)])

    result = pd.read_parquet(tmp_path / "opus_v4_calibration.parquet")
    assert list(result.columns) == ["post_id", "text", "stratum"]
    assert result["post_id"].tolist() == ["001", "002", "003"]
    assert result["text"].tolist() == ["one", "two", "three"]
    assert result["stratum"].tolist() == ["calibration"] * 3
    assert not any(column.startswith("human_") for column in result)
    assert not any(column.startswith("reference_") for column in result)


def test_make_calibration_refuses_overwrite_unless_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    write_source(source)
    target = tmp_path / "opus_v4_calibration.parquet"
    target.write_bytes(b"keep-me")
    monkeypatch.setattr(module, "OUT", tmp_path)

    with pytest.raises(FileExistsError, match=r"already exist.*--force"):
        module.main(["make-calibration", "--source", str(source)])
    assert target.read_bytes() == b"keep-me"

    module.main(["make-calibration", "--source", str(source), "--force"])
    assert pd.read_parquet(target)["post_id"].tolist() == ["001", "002", "003"]


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
    assert len(report["provenance"]["reference_source"]["sha256"]) == 64
    assert mismatches["post_id"].tolist() == ["002"]
    assert {"human_label", "label", "flags"}.issubset(mismatches.columns)
    assert json.loads(mismatches.loc[0, "flags"]) == [
        "dehumanisation",
        "ethnic_targeting",
    ]


def test_score_calibration_rejects_missing_and_extra_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "blind_check_coded_calibration.csv"
    write_source(source)
    rows = [valid_response("001"), valid_response("999")]
    batch = pd.DataFrame(
        {
            "post_id": ["001", "999"],
            "text": ["one", "extra"],
            "stratum": ["calibration", "calibration"],
        }
    )
    write_run(tmp_path, rows=rows, batch=batch)
    monkeypatch.setattr(module, "OUT", tmp_path)

    with pytest.raises(ValueError, match=r"missing=.*999.*extra=.*002.*003"):
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


def test_score_calibration_checks_output_set_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    write_source(source)
    write_run(tmp_path)
    companion = tmp_path / "21_opus_calibration_test-tag_mismatches.csv"
    companion.write_text("keep-me")
    monkeypatch.setattr(module, "OUT", tmp_path)

    command = [
        "score-calibration",
        "--tag",
        "test-tag",
        "--source",
        str(source),
    ]
    with pytest.raises(FileExistsError, match="mismatches.csv"):
        module.main(command)
    assert not (tmp_path / "21_opus_calibration_test-tag.json").exists()
    assert companion.read_text() == "keep-me"

    module.main([*command, "--force"])
    assert json.loads(
        (tmp_path / "21_opus_calibration_test-tag.json").read_text()
    )["rows"] == 3


@pytest.mark.parametrize(
    ("manifest_changes", "message"),
    [
        ({"tag": "stale-tag"}, "manifest tag mismatch"),
        ({"rows": 99}, "manifest rows mismatch"),
        ({"input_sha256": "stale"}, "manifest batch hash mismatch"),
        (
            {
                "labellers": {
                    "claude-sonnet-4.6": {
                        "cli": "agy",
                        "model": "Claude Sonnet 4.6 (Thinking)",
                    }
                }
            },
            "requested labeller missing",
        ),
        (
            {
                "labellers": {
                    "claude-opus-4.6": {"cli": "agy", "model": "wrong-model"}
                }
            },
            "labeller mapping mismatch",
        ),
    ],
)
def test_score_calibration_rejects_stale_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_changes: dict,
    message: str,
) -> None:
    source = tmp_path / "source.csv"
    write_source(source)
    write_run(tmp_path, manifest_changes=manifest_changes)
    monkeypatch.setattr(module, "OUT", tmp_path)

    with pytest.raises(ValueError, match=message):
        module.main(
            [
                "score-calibration",
                "--tag",
                "test-tag",
                "--source",
                str(source),
            ]
        )


def test_score_calibration_rejects_changed_source_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    changed = source_frame()
    changed.loc[changed["post_id"] == "002", "text"] = "changed after run"
    changed.to_csv(source, index=False)
    write_run(tmp_path)
    monkeypatch.setattr(module, "OUT", tmp_path)

    with pytest.raises(ValueError, match=r"source text mismatch.*002"):
        module.main(
            [
                "score-calibration",
                "--tag",
                "test-tag",
                "--source",
                str(source),
            ]
        )


def test_score_calibration_rejects_batch_changed_after_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    write_source(source)
    root = write_run(tmp_path)
    batch = pd.read_parquet(root / "batch.parquet")
    batch.loc[0, "text"] = "tampered"
    batch.to_parquet(root / "batch.parquet", index=False)
    monkeypatch.setattr(module, "OUT", tmp_path)

    with pytest.raises(ValueError, match="manifest batch hash mismatch"):
        module.main(
            [
                "score-calibration",
                "--tag",
                "test-tag",
                "--source",
                str(source),
            ]
        )


def test_score_calibration_rejects_incomplete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    write_source(source)
    root = write_run(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["prompt_sha256"]
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(module, "OUT", tmp_path)

    with pytest.raises(ValueError, match=r"missing required fields.*prompt_sha256"):
        module.main(
            [
                "score-calibration",
                "--tag",
                "test-tag",
                "--source",
                str(source),
            ]
        )


def matching_batch(
    strata: list[str | None] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "post_id": ["001", "002", "003"],
            "text": ["one", "two", "three"],
            "stratum": strata or ["random", "lexicon", "lexicon"],
        }
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
            "confidence": ["low", "medium", "high"],
            "conf_gemini": ["high", "medium", "high"],
        }
    ).to_parquet(baseline_path, index=False)
    write_run(tmp_path, batch=matching_batch())
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
    assert report["confidence_counts"]["source"]["confidence"] == {
        "high": 1,
        "low": 1,
        "medium": 1,
    }
    assert report["provenance"]["source"]["columns"]["prompt_version"] == ["v2"]
    assert len(report["provenance"]["source"]["sha256"]) == 64
    assert report["provenance"]["new"]["run_manifest"]["prompt_version"] == "v4"
    assert changed["post_id"].tolist() == ["002", "003"]
    assert {"previous_label", "new_label", "label_changed", "flags_changed"}.issubset(
        changed.columns
    )
    assert json.loads(changed.loc[0, "previous_flags"]) == ["dehumanisation"]
    assert json.loads(changed.loc[0, "new_flags"]) == [
        "dehumanisation",
        "ethnic_targeting",
    ]


def test_compare_full_rejects_id_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "baseline.parquet"
    pd.DataFrame(
        {
            "post_id": ["001", "002"],
            "text": ["one", "two"],
            "stratum": ["random", "lexicon"],
            "label": ["neither", "offensive"],
            "flags": [[], []],
        }
    ).to_parquet(baseline_path, index=False)
    write_run(tmp_path, batch=matching_batch())
    monkeypatch.setattr(module, "OUT", tmp_path)

    with pytest.raises(ValueError, match=r"missing=.*003"):
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


def test_compare_full_rejects_changed_baseline_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "baseline.parquet"
    matching_batch().assign(
        text=["one", "CHANGED", "three"],
        label=["neither", "offensive", "hate"],
        flags=[[], [], ["ethnic_targeting"]],
    ).to_parquet(baseline_path, index=False)
    write_run(tmp_path, batch=matching_batch())
    monkeypatch.setattr(module, "OUT", tmp_path)

    with pytest.raises(ValueError, match=r"baseline text mismatch.*002"):
        module.main(
            [
                "compare-full",
                "--tag",
                "test-tag",
                "--baseline",
                str(baseline_path),
            ]
        )


def test_compare_full_rejects_changed_baseline_stratum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "baseline.parquet"
    matching_batch().assign(
        stratum=["random", "CHANGED", "lexicon"],
        label=["neither", "offensive", "hate"],
        flags=[[], [], ["ethnic_targeting"]],
    ).to_parquet(baseline_path, index=False)
    write_run(tmp_path, batch=matching_batch())
    monkeypatch.setattr(module, "OUT", tmp_path)

    with pytest.raises(ValueError, match=r"baseline stratum mismatch.*002"):
        module.main(
            [
                "compare-full",
                "--tag",
                "test-tag",
                "--baseline",
                str(baseline_path),
            ]
        )


def test_compare_full_checks_output_set_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "baseline.parquet"
    matching_batch().assign(
        label=["neither", "offensive", "hate"],
        flags=[[], ["dehumanisation"], ["ethnic_targeting"]],
    ).to_parquet(baseline_path, index=False)
    write_run(tmp_path, batch=matching_batch())
    report_path = tmp_path / "21_opus_full_test-tag.json"
    report_path.write_text("keep-me")
    monkeypatch.setattr(module, "OUT", tmp_path)

    command = [
        "compare-full",
        "--tag",
        "test-tag",
        "--baseline",
        str(baseline_path),
    ]
    with pytest.raises(FileExistsError, match="21_opus_full_test-tag.json"):
        module.main(command)
    assert report_path.read_text() == "keep-me"
    assert not (tmp_path / "21_opus_full_test-tag_changed.csv").exists()

    module.main([*command, "--force"])
    assert json.loads(report_path.read_text())["rows"] == 3


def test_compare_full_preserves_null_and_string_nan_strata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "baseline.parquet"
    strata = [None, "nan", "lexicon"]
    matching_batch(strata).assign(
        label=["neither", "offensive", "hate"],
        flags=[[], ["dehumanisation"], ["ethnic_targeting"]],
    ).to_parquet(baseline_path, index=False)
    write_run(tmp_path, batch=matching_batch(strata))
    monkeypatch.setattr(module, "OUT", tmp_path)

    module.main(
        [
            "compare-full",
            "--tag",
            "test-tag",
            "--baseline",
            str(baseline_path),
        ]
    )

    strata = json.loads(
        (tmp_path / "21_opus_full_test-tag.json").read_text()
    )["changed"]["by_stratum"]
    assert set(strata) == {"<null>", "nan", "lexicon"}
    assert strata["<null>"]["n"] == 1
    assert strata["nan"]["n"] == 1


@pytest.mark.parametrize("value", ["../escape", "/tmp/escape", ".", "..", "a/b", r"a\b"])
@pytest.mark.parametrize("option", ["--tag", "--labeller"])
def test_cli_rejects_unsafe_path_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    value: str,
) -> None:
    monkeypatch.setattr(module, "OUT", tmp_path)
    arguments = [
        "score-calibration",
        "--tag",
        "safe-tag",
        "--labeller",
        "claude-opus-4.6",
    ]
    arguments[arguments.index(option) + 1] = value

    with pytest.raises(ValueError, match="unsafe (tag|labeller)"):
        module.main(arguments)

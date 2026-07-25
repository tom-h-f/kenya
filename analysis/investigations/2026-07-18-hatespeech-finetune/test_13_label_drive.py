from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def load_label_drive_module():
    path = Path(__file__).with_name("13_label_drive.py")
    spec = importlib.util.spec_from_file_location("label_drive", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cursor_labeller_runs_in_read_only_ask_mode() -> None:
    module = load_label_drive_module()

    command = module.build_cmd("cursor", "sonnet-4.5", "10m")

    assert "--mode" in command
    assert command[command.index("--mode") + 1] == "ask"
    assert "label these" not in command


def test_claude_labeller_uses_print_mode() -> None:
    module = load_label_drive_module()

    command = module.build_cmd("claude", "opus", "10m")

    assert command == ["claude", "-p", "--model", "opus"]


def test_opus_labeller_uses_current_agy_identifier() -> None:
    module = load_label_drive_module()

    assert module.LABELLERS["claude-opus-4.6"] == (
        "agy",
        "claude-opus-4-6-thinking",
    )


def valid_response(**overrides: object) -> dict:
    response = {
        "post_id": "post-1",
        "label": "hate",
        "flags": ["ethnic_targeting"],
        "target_group": "Luo",
        "confidence": "high",
        "rationale": "The phrase attacks Luo people.",
    }
    response.update(overrides)
    return response


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ({k: v for k, v in valid_response().items() if k != "confidence"}, "confidence"),
        ({k: v for k, v in valid_response().items() if k != "rationale"}, "rationale"),
        ({k: v for k, v in valid_response().items() if k != "target_group"}, "target_group"),
        (valid_response(confidence="certain"), "confidence"),
        (valid_response(flags=[]), "ethnic_targeting"),
        (
            valid_response(label="offensive", flags=["ethnic_targeting"]),
            "ethnic_targeting",
        ),
        (
            valid_response(
                label="offensive",
                flags=[],
                target_group="Luo",
            ),
            "target_group",
        ),
    ],
    ids=[
        "missing-confidence",
        "missing-rationale",
        "missing-target-group",
        "invalid-confidence",
        "hate-without-ethnic-targeting",
        "ethnic-targeting-on-non-hate",
        "target-group-without-ethnic-targeting",
    ],
)
def test_parse_response_rejects_invalid_output(response: dict, error: str) -> None:
    module = load_label_drive_module()

    with pytest.raises(ValueError, match=error):
        module.parse_response(json.dumps(response), ["post-1"])


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (valid_response(flags=None), "flags"),
        (valid_response(flags=""), "flags"),
        (
            valid_response(flags=["ethnic_targeting", "ethnic_targeting"]),
            "duplicate",
        ),
        (valid_response(extra="field"), "extra fields"),
        ({k: v for k, v in valid_response().items() if k != "flags"}, "missing fields"),
        (valid_response(post_id=1), "post_id"),
        (valid_response(label=["hate"]), "label"),
        (valid_response(confidence=["high"]), "confidence"),
        (valid_response(rationale=1), "rationale"),
        (valid_response(rationale=""), "rationale"),
        (valid_response(target_group=1), "target_group"),
        (valid_response(target_group=""), "target_group"),
    ],
)
def test_parse_response_rejects_malformed_schema(response: dict, error: str) -> None:
    module = load_label_drive_module()

    with pytest.raises(ValueError, match=error):
        module.parse_response(json.dumps(response), ["post-1"])


def test_parse_response_rejects_reordered_ids() -> None:
    module = load_label_drive_module()
    first = valid_response(post_id="post-1")
    second = valid_response(post_id="post-2")

    with pytest.raises(ValueError, match="order"):
        module.parse_response(
            "\n".join([json.dumps(second), json.dumps(first)]),
            ["post-1", "post-2"],
        )


@pytest.mark.parametrize(
    "raw",
    [
        "explanation\n" + json.dumps(valid_response()),
        json.dumps(valid_response()) + "\ntrailing prose",
        "[]",
        "{not json}",
        json.dumps(valid_response()) + " " + json.dumps(valid_response()),
        "```unexpected\n" + json.dumps(valid_response()) + "\n```",
    ],
    ids=[
        "leading-prose",
        "trailing-prose",
        "array",
        "malformed-json",
        "multiple-objects-one-line",
        "unknown-fence",
    ],
)
def test_parse_response_rejects_non_jsonl_content(raw: str) -> None:
    module = load_label_drive_module()

    with pytest.raises(ValueError, match="JSONL"):
        module.parse_response(raw, ["post-1"])


def test_parse_response_allows_blank_lines_and_json_fences() -> None:
    module = load_label_drive_module()
    raw = f"\n```json\n{json.dumps(valid_response())}\n```\n"

    rows = module.parse_response(raw, ["post-1"])

    assert [row["post_id"] for row in rows] == ["post-1"]


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (
            valid_response(
                label="neither",
                flags=["dehumanisation"],
                target_group=None,
            ),
            "neither.*dehumanisation",
        ),
        (
            valid_response(
                label="neither",
                flags=["violence_call"],
                target_group=None,
            ),
            "neither.*violence_call",
        ),
        (
            valid_response(
                label="neither",
                flags=["ethnic_targeting"],
                target_group="Luo",
            ),
            "neither.*ethnic_targeting",
        ),
        (
            valid_response(target_group=None),
            "hate.*target_group",
        ),
    ],
)
def test_parse_response_enforces_complete_prompt_invariants(
    response: dict, error: str
) -> None:
    module = load_label_drive_module()

    with pytest.raises(ValueError, match=error):
        module.parse_response(json.dumps(response), ["post-1"])


def test_label_chunk_sends_sensitive_content_only_through_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_label_drive_module()
    prompt_path = tmp_path / "label_v4.md"
    prompt_path.write_text("PROMPT_SECRET_MARKER")
    chunk = tmp_path / "chunk_000.jsonl"
    chunk.write_text(
        json.dumps({"post_id": "post-1", "text": "POST_SECRET_MARKER"})
    )
    invocation = {}

    def run(command, **kwargs):
        invocation["command"] = command
        invocation["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(valid_response()),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", run)

    module.label_chunk(
        "claude-opus-4.6",
        "agy",
        "claude-opus-4-6-thinking",
        chunk,
        tmp_path / "labels",
        tmp_path / "failed",
        "10m",
        prompt_path,
    )

    command_text = " ".join(invocation["command"])
    assert "PROMPT_SECRET_MARKER" not in command_text
    assert "POST_SECRET_MARKER" not in command_text
    assert "PROMPT_SECRET_MARKER" in invocation["kwargs"]["input"]
    assert "POST_SECRET_MARKER" in invocation["kwargs"]["input"]


def test_existing_output_is_validated_before_skip(tmp_path: Path) -> None:
    module = load_label_drive_module()
    chunk = tmp_path / "chunk_000.jsonl"
    chunk.write_text(
        "\n".join(
            [
                json.dumps({"post_id": "post-1", "text": "one"}),
                json.dumps({"post_id": "post-2", "text": "two"}),
            ]
        )
    )
    out_dir = tmp_path / "labels"
    out_dir.mkdir()
    (out_dir / "chunk_000.jsonl").write_text(
        json.dumps(valid_response(post_id="post-1"))
    )

    with pytest.raises(ValueError, match="invalid existing output"):
        module.pending_chunks([chunk], out_dir)


def test_valid_existing_output_is_skipped(tmp_path: Path) -> None:
    module = load_label_drive_module()
    chunk = tmp_path / "chunk_000.jsonl"
    chunk.write_text(json.dumps({"post_id": "post-1", "text": "one"}))
    out_dir = tmp_path / "labels"
    out_dir.mkdir()
    existing = module.parse_response(
        json.dumps(valid_response(post_id="post-1")),
        ["post-1"],
    )
    (out_dir / "chunk_000.jsonl").write_text(
        "\n".join(json.dumps(row) for row in existing)
    )

    assert module.pending_chunks([chunk], out_dir) == []


def test_tag_lock_rejects_concurrent_holder_and_releases(tmp_path: Path) -> None:
    module = load_label_drive_module()
    lock_path = tmp_path / "test-run.lock"

    with module.tag_lock(lock_path):
        with pytest.raises(RuntimeError, match="already locked"):
            with module.tag_lock(lock_path):
                pass

    with module.tag_lock(lock_path):
        pass


def test_tag_lock_rejects_holder_in_another_process(tmp_path: Path) -> None:
    module = load_label_drive_module()
    module_path = Path(__file__).with_name("13_label_drive.py")
    lock_path = tmp_path / "cross-process.lock"
    code = """
import importlib.util
import sys
import time
from pathlib import Path

module_path = Path(sys.argv[1])
sys.path.insert(0, str(module_path.parent))
spec = importlib.util.spec_from_file_location("label_drive_child", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with module.tag_lock(Path(sys.argv[2])):
    print("locked", flush=True)
    time.sleep(30)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(module_path), str(lock_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="already locked"):
            with module.tag_lock(lock_path):
                pass
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_new_run_root_must_be_absent_or_empty(tmp_path: Path) -> None:
    module = load_label_drive_module()
    absent = tmp_path / "absent"
    empty = tmp_path / "empty"
    empty.mkdir()

    module.validate_new_run_root(absent)
    module.validate_new_run_root(empty)


def test_new_run_root_rejects_artifacts_without_manifest(tmp_path: Path) -> None:
    module = load_label_drive_module()
    root = tmp_path / "test-run"
    root.mkdir()
    (root / "batch.parquet").write_bytes(b"partial")

    with pytest.raises(ValueError, match="unverifiable provenance"):
        module.validate_new_run_root(root)

    assert not (root / "manifest.json").exists()


def test_atomic_write_preserves_target_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_label_drive_module()
    path = tmp_path / "state.json"
    path.write_text("old")

    def fail_replace(source, target):
        raise OSError("replace interrupted")

    monkeypatch.setattr(module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace interrupted"):
        module.atomic_write_text(path, "new")

    assert path.read_text() == "old"
    assert list(tmp_path.glob("*.tmp")) == []


def test_run_manifest_captures_reproducible_provenance(tmp_path: Path) -> None:
    module = load_label_drive_module()
    prompt_path = tmp_path / "label_v4.md"
    prompt_bytes = b"# Prompt v4\n\xcf\x80 exact bytes\n"
    prompt_path.write_bytes(prompt_bytes)
    batch = pd.DataFrame(
        [
            {
                "post_id": "post-1",
                "text": "text 1",
                "stratum": "sample",
                "metadata": "m1",
            },
            {
                "post_id": "post-2",
                "text": "text 2",
                "stratum": "sample",
                "metadata": "m2",
            },
        ]
    )
    created_at = datetime(2026, 7, 21, 12, 34, 56, tzinfo=timezone.utc)
    manifest = module.run_manifest(
        tag="opus-v4",
        prompt_path=prompt_path,
        labellers=["claude-opus-4.6"],
        input_name="/source/label_batch_001.parquet",
        batch=batch,
        created_at=created_at,
        chunk_size=25,
        concurrency=4,
    )

    assert manifest == {
        "tag": "opus-v4",
        "created_at": "2026-07-21T12:34:56Z",
        "input_path": "/source/label_batch_001.parquet",
        "input_name": "label_batch_001.parquet",
        "rows": 2,
        "input_sha256": module.batch_sha256(batch),
        "prompt_filename": "label_v4.md",
        "prompt_version": "v4",
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "labellers": {
            "claude-opus-4.6": {
                "cli": "agy",
                "model": "claude-opus-4-6-thinking",
            },
        },
        "chunk_size": 25,
        "concurrency": 4,
    }


def test_batch_hash_covers_content_and_column_order() -> None:
    module = load_label_drive_module()
    batch = pd.DataFrame(
        [
            {
                "post_id": "post-1",
                "text": "original",
                "stratum": "sample",
                "metadata": "m1",
            }
        ]
    )
    reordered = batch[["metadata", "stratum", "text", "post_id"]]
    changed_text = batch.assign(text="changed")
    changed_metadata = batch.assign(metadata="m2")

    assert module.batch_sha256(batch) != module.batch_sha256(reordered)
    assert module.batch_sha256(batch) != module.batch_sha256(changed_text)
    assert module.batch_sha256(batch) != module.batch_sha256(changed_metadata)


def test_batch_hash_preserves_exact_float64_bits() -> None:
    module = load_label_drive_module()
    lower = pd.DataFrame({"value": [1.0000000000000002]})
    upper = pd.DataFrame({"value": [1.0000000000000004]})

    assert module.batch_sha256(lower) != module.batch_sha256(upper)


def test_batch_hash_distinguishes_null_from_nan_when_represented() -> None:
    module = load_label_drive_module()
    nan_value = pd.DataFrame({"value": pd.Series([float("nan")], dtype=object)})
    null_value = pd.DataFrame({"value": pd.Series([None], dtype=object)})

    assert module.batch_sha256(nan_value) != module.batch_sha256(null_value)


def test_batch_hash_includes_dtype_and_row_order() -> None:
    module = load_label_drive_module()
    integers = pd.DataFrame({"value": pd.Series([1, 2], dtype="int64")})
    nullable = pd.DataFrame({"value": pd.Series([1, 2], dtype="Int64")})
    reversed_rows = integers.iloc[::-1].reset_index(drop=True)

    assert module.batch_sha256(integers) != module.batch_sha256(nullable)
    assert module.batch_sha256(integers) != module.batch_sha256(reversed_rows)


def manifest_identity() -> dict:
    return {
        "tag": "test-run",
        "created_at": "2026-07-21T12:34:56Z",
        "input_path": "/source/label_batch_001.parquet",
        "input_name": "label_batch_001.parquet",
        "rows": 1,
        "input_sha256": "input-hash",
        "prompt_filename": "label_v4.md",
        "prompt_version": "v4",
        "prompt_sha256": "prompt-hash",
        "labellers": {
            "claude-opus-4.6": {
                "cli": "agy",
                "model": "claude-opus-4-6-thinking",
            }
        },
        "chunk_size": 25,
        "concurrency": 4,
    }


def test_validate_run_manifest_accepts_matching_resume() -> None:
    module = load_label_drive_module()
    existing = manifest_identity()
    current = {**existing, "created_at": "2026-07-21T13:00:00Z"}

    module.validate_run_manifest(existing, current)


def test_validate_run_manifest_allows_different_checkout_path() -> None:
    module = load_label_drive_module()
    existing = manifest_identity()
    current = {
        **existing,
        "input_path": "/another/checkout/label_batch_001.parquet",
    }

    module.validate_run_manifest(existing, current)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("rows", 2),
        ("input_sha256", "different-input-hash"),
        ("prompt_filename", "label_v5.md"),
        ("prompt_version", "v5"),
        ("prompt_sha256", "different-prompt-hash"),
        (
            "labellers",
            {
                "claude-opus-4.6": {
                    "cli": "agy",
                    "model": "other-model",
                }
            },
        ),
        ("chunk_size", 50),
        ("concurrency", 8),
    ],
)
def test_validate_run_manifest_rejects_changed_identity(
    field: str, changed: object
) -> None:
    module = load_label_drive_module()
    existing = manifest_identity()
    current = {**existing, field: changed}

    with pytest.raises(ValueError, match=field):
        module.validate_run_manifest(existing, current)


def test_main_writes_manifest_before_labelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_label_drive_module()
    out = tmp_path / "out"
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "label_v4.md").write_bytes(b"prompt v4")
    df = pd.DataFrame(
        [{"post_id": "post-1", "text": "text", "stratum": "sample"}]
    )
    chunk = tmp_path / "chunk_000.jsonl"
    chunk.write_text('{"post_id":"post-1","text":"text"}')

    monkeypatch.setattr(module, "OUT", out)
    monkeypatch.setattr(module, "PROMPT_DIR", prompt_dir)
    monkeypatch.setattr(module.pd, "read_parquet", lambda _: df)
    monkeypatch.setattr(module, "write_chunks", lambda _df, _path: [chunk])
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index: Path(path).write_bytes(b"batch"),
    )

    def label_chunk(*args, **kwargs):
        manifest_path = out / "labels" / "test-run" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["labellers"] == {
            "claude-opus-4.6": {
                "cli": "agy",
                "model": "claude-opus-4-6-thinking",
            }
        }
        return {"chunk": "chunk_000", "rows": 1, "seconds": 0.1, "attempts": 1}

    monkeypatch.setattr(module, "label_chunk", label_chunk)
    monkeypatch.setattr(
        "sys.argv",
        [
            "13_label_drive.py",
            "--tag",
            "test-run",
            "--labellers",
            "claude-opus-4.6",
            "--prompt-version",
            "v4",
        ],
    )

    module.main()


def test_main_rejects_conflicting_manifest_before_model_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_label_drive_module()
    out = tmp_path / "out"
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    prompt_path = prompt_dir / "label_v4.md"
    prompt_path.write_bytes(b"prompt v4")
    df = pd.DataFrame(
        [{"post_id": "post-1", "text": "text", "stratum": "sample"}]
    )
    root = out / "labels" / "test-run"
    root.mkdir(parents=True)
    existing = module.run_manifest(
        tag="test-run",
        prompt_path=prompt_path,
        labellers=["claude-opus-4.6"],
        input_name=str(out / "label_batch_001.parquet"),
        batch=df.assign(text="changed"),
        concurrency=4,
    )
    (root / "manifest.json").write_text(json.dumps(existing))
    model_called = False

    def label_chunk(*args, **kwargs):
        nonlocal model_called
        model_called = True
        raise AssertionError("model must not run")

    monkeypatch.setattr(module, "OUT", out)
    monkeypatch.setattr(module, "PROMPT_DIR", prompt_dir)
    monkeypatch.setattr(module.pd, "read_parquet", lambda _: df)
    monkeypatch.setattr(module, "label_chunk", label_chunk)
    monkeypatch.setattr(
        module,
        "write_chunks",
        lambda _df, _path: pytest.fail("chunks must not change on conflict"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "13_label_drive.py",
            "--tag",
            "test-run",
            "--labellers",
            "claude-opus-4.6",
            "--prompt-version",
            "v4",
        ],
    )

    with pytest.raises(ValueError, match="input_sha256"):
        module.main()

    assert model_called is False


def test_main_resumes_when_manifest_identity_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_label_drive_module()
    out = tmp_path / "out"
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    prompt_path = prompt_dir / "label_v4.md"
    prompt_path.write_bytes(b"prompt v4")
    df = pd.DataFrame(
        [{"post_id": "post-1", "text": "text", "stratum": "sample"}]
    )
    chunk = tmp_path / "chunk_000.jsonl"
    chunk.write_text('{"post_id":"post-1","text":"text"}')
    root = out / "labels" / "test-run"
    root.mkdir(parents=True)
    existing = module.run_manifest(
        tag="test-run",
        prompt_path=prompt_path,
        labellers=["claude-opus-4.6"],
        input_name=str(out / "label_batch_001.parquet"),
        batch=df,
        created_at=datetime(2026, 7, 21, 12, 34, 56, tzinfo=timezone.utc),
        concurrency=4,
    )
    manifest_path = root / "manifest.json"
    manifest_text = json.dumps(existing, indent=2) + "\n"
    manifest_path.write_text(manifest_text)
    model_called = False

    def label_chunk(*args, **kwargs):
        nonlocal model_called
        model_called = True
        return {"chunk": "chunk_000", "rows": 1, "seconds": 0.1, "attempts": 1}

    monkeypatch.setattr(module, "OUT", out)
    monkeypatch.setattr(module, "PROMPT_DIR", prompt_dir)
    monkeypatch.setattr(module.pd, "read_parquet", lambda _: df)
    monkeypatch.setattr(module, "write_chunks", lambda _df, _path: [chunk])
    monkeypatch.setattr(module, "label_chunk", label_chunk)
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index: Path(path).write_bytes(b"batch"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "13_label_drive.py",
            "--tag",
            "test-run",
            "--labellers",
            "claude-opus-4.6",
            "--prompt-version",
            "v4",
        ],
    )

    module.main()

    assert model_called is True
    assert manifest_path.read_text() == manifest_text

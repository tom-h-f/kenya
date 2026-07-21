from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

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

    command = module.build_cmd("cursor", "sonnet-4.5", "label these", "10m")

    assert "--mode" in command
    assert command[command.index("--mode") + 1] == "ask"


def test_claude_labeller_uses_print_mode() -> None:
    module = load_label_drive_module()

    command = module.build_cmd("claude", "opus", "label these", "10m")

    assert command == ["claude", "-p", "label these", "--model", "opus"]


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


def test_run_manifest_captures_reproducible_provenance(tmp_path: Path) -> None:
    module = load_label_drive_module()
    prompt_path = tmp_path / "label_v4.md"
    prompt_bytes = b"# Prompt v4\n\xcf\x80 exact bytes\n"
    prompt_path.write_bytes(prompt_bytes)
    created_at = datetime(2026, 7, 21, 12, 34, 56, tzinfo=timezone.utc)
    manifest = module.run_manifest(
        tag="opus-v4",
        prompt_path=prompt_path,
        labellers=["claude-opus-4.6"],
        input_name="/source/label_batch_001.parquet",
        rows=125,
        created_at=created_at,
        chunk_size=25,
        concurrency=4,
    )

    assert manifest == {
        "tag": "opus-v4",
        "created_at": "2026-07-21T12:34:56Z",
        "input_path": "/source/label_batch_001.parquet",
        "input_name": "label_batch_001.parquet",
        "rows": 125,
        "prompt_filename": "label_v4.md",
        "prompt_version": "v4",
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "labellers": {
            "claude-opus-4.6": {
                "cli": "agy",
                "model": "Claude Opus 4.6 (Thinking)",
            },
        },
        "chunk_size": 25,
        "concurrency": 4,
    }


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
                "model": "Claude Opus 4.6 (Thinking)",
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

from __future__ import annotations

import importlib.util
from pathlib import Path


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

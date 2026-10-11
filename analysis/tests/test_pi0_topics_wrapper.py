"""The pi0 topics cron wrapper, run end to end against a stub ssh.

The gauge `kma_topics_assigned_posts` read 0 after an idle run, which hides the
last real assignment; `kma_topics_last_run_assigned_posts` now carries the
per-run number. Each wrapper also logs the git sha it synced mike-wsl to."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOPICS = ROOT / "deploy/pi0/kenya-topics-cron.sh"
INCITEMENT = ROOT / "deploy/pi0/kenya-incitement-cron.sh"
SHA = "0123456789abcdef0123456789abcdef01234567"

SSH_STUB = f"""#!/usr/bin/env bash
in=$(cat)
case "$in" in
  *"echo up"*) echo up ;;
  *"nvidia-smi"*) echo 0 ;;
  *"awake status"*) echo "awake: on"; ;;
  *"git checkout"*) echo "SHA {SHA}"; echo SYNC_OK ;;
  *"base64 -d"*) wc -c <<<"$in" | tr -d ' ' ;;
  *"tmux new-session"*) echo STARTED ;;
  *"sed -n 's/^exit"*) echo 0 ;;
  *"tmux has-session"*) ;;
esac
exit 0
"""


def _setup(tmp_path: Path, assigned: int) -> tuple[dict[str, str], Path]:
    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    textfile = tmp_path / "textfile"
    textfile.mkdir()
    ssh = home / "bin" / "ssh"
    ssh.write_text(SSH_STUB)
    ssh.chmod(0o755)
    wake = home / "bin" / "wake-lan"
    wake.write_text("#!/bin/sh\nexit 0\n")
    wake.chmod(0o755)
    (home / "kenya_topics_r2.py").write_text(
        "import sys\n"
        "if sys.argv[1] == 'manifest':\n    print('{}')\n"
        f"else:\n    print('{{\"assigned\": {assigned}}}')\n"
    )
    env = {
        **os.environ,
        "KMA_TOPICS_HOME": str(home),
        "KMA_TOPICS_SSH": str(ssh),
        "KMA_TOPICS_PATH_PREPEND": str(home / "bin"),
        "KMA_TOPICS_TEXTFILE_DIR": str(textfile),
        "KMA_TOPICS_POLL_S": "0",
        "KMA_TOPICS_DEADLINE_S": "5",
    }
    return env, textfile


def _run(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=60)


def _gauges(textfile: Path) -> dict[str, float]:
    out = {}
    for line in (textfile / "kma_topics.prom").read_text().splitlines():
        if line and not line.startswith("#"):
            name, value = line.split()
            out[name] = float(value)
    return out


def _log(env: dict[str, str]) -> str:
    return "".join(p.read_text() for p in (Path(env["KMA_TOPICS_HOME"]) / "cron-logs").glob("kenya-topics-*.log"))


def test_idle_run_keeps_last_nonidle_assigned_and_reports_zero_for_the_run(tmp_path):
    env, textfile = _setup(tmp_path, assigned=1234)
    assert _run(TOPICS, env).returncode == 0
    first = _gauges(textfile)
    assert first["kma_topics_assigned_posts"] == 1234
    assert first["kma_topics_last_run_assigned_posts"] == 1234

    env2, _ = _setup(tmp_path / "again", assigned=0)
    env2["KMA_TOPICS_TEXTFILE_DIR"] = str(textfile)
    assert _run(TOPICS, env2).returncode == 0
    idle = _gauges(textfile)
    assert idle["kma_topics_assigned_posts"] == 1234
    assert idle["kma_topics_last_run_assigned_posts"] == 0
    assert idle["kma_topics_last_attempt_ok"] == 1


def test_topics_wrapper_logs_the_resolved_sha(tmp_path):
    env, _ = _setup(tmp_path, assigned=5)
    _run(TOPICS, env)
    assert re.search(rf"repo at {SHA} \(master\)", _log(env))


def test_incitement_wrapper_logs_the_sha_it_resolves_in_the_repo_dir():
    text = INCITEMENT.read_text()
    block = text[text.index("git checkout -q -B"):text.index("echo SYNC_OK")]
    assert block.index("git rev-parse HEAD") < block.index('cd "\\$B"\n')
    assert 'log "repo at ' in text


@pytest.mark.parametrize("script", [TOPICS, INCITEMENT])
def test_wrappers_parse(script):
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0

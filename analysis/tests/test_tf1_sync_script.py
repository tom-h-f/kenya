"""deploy/tf1/sync.sh dry run against a local file clone: it must report the
rsync diff and write nothing to the tree."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy/tf1/sync.sh"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_script_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck_clean():
    subprocess.run(["shellcheck", str(SCRIPT)], check=True)


def test_dry_run_writes_nothing(tmp_path):
    origin = tmp_path / "origin"
    (origin / "analysis/src/kma").mkdir(parents=True)
    (origin / "deploy/tf1").mkdir(parents=True)
    (origin / "analysis/src/kma/new.py").write_text("x = 1\n")
    for name in ("Dockerfile", ".dockerignore", ".python-version", "pyproject.toml",
                 "uv.lock", "README.md", "compose.yaml"):
        (origin / "analysis" / name).write_text("a\n")
    shutil.copy(ROOT / "deploy/tf1/docker-compose.override.yaml", origin / "deploy/tf1/")
    _git(origin, "init", "-q", "-b", "master")
    _git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")

    tree = tmp_path / "tree"
    (tree / "analysis/src/kma").mkdir(parents=True)
    (tree / "analysis/src/kma/stale.py").write_text("y = 2\n")
    (tree / ".env").write_text("K=v\n")

    env = {
        "PATH": subprocess.os.environ["PATH"],
        "HOME": str(tmp_path),
        "REPO_URL": str(origin),
        "CLONE": str(tmp_path / "clone"),
        "TREE": str(tree),
        "BACKUPS": str(tmp_path / "backups"),
    }
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], env=env, check=True, capture_output=True, text=True
    ).stdout

    assert "new.py" in out
    assert "deleting" in out and "stale.py" in out
    assert (tree / "analysis/src/kma/stale.py").exists()
    assert not (tree / "analysis/src/kma/new.py").exists()
    assert not (tree / "analysis/docker-compose.override.yaml").exists()
    assert not (tmp_path / "backups").exists()

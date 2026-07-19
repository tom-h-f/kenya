"""Shared bits for the 2026-07-18 hate-speech fine-tune scripts.

Every script here runs standalone via `uv run <script>` (PEP 723 inline deps,
no analysis env needed) so the same files work on Colab after one pip cell.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

LABELS = ["neither", "offensive", "hate"]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

SEED = 42


def default_csv() -> Path:
    """HateSpeech_Kenya.csv from repo root (worktree or main checkout)."""
    for parent in HERE.parents:
        candidate = parent / "HateSpeech_Kenya.csv"
        if candidate.exists():
            return candidate
    return HERE / "HateSpeech_Kenya.csv"


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_split(name: str):
    import pandas as pd

    path = OUT / f"{name}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} missing - run 00_prep.py first")
    return pd.read_parquet(path)

"""The text a relevance model sees: one function for training and for serving.

Standard library only, so the same file runs on the GPU box and in the Modal
image. A model trained on one preparation and served another is a mismatch the
2026-09-12 retrain already paid for.

Mentions are stripped by default: a post with no words of its own otherwise
rides on the handles it replies to ("@SomeForcePolice [emoji]" scored 0.80). That
also removed the handle-carried true positives (`@Ruto_tutam2027 <link>`), so a
variant can keep a mention when the handle itself says Kenya.

    strip       every mention removed
    known       a mention is kept when its handle is in the model's Kenyan-handle list
    known_lex   ... or when the handle contains a Kenyan political term
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MENTION = re.compile(r"@(\w+)", re.UNICODE)
VARIANTS = ("strip", "known", "known_lex")
HANDLES_FILE = "kenyan_handles.txt"
PREP_FILE = "prep.json"

# Kept in step with `measure._KENYA_RE` by `test_relevance_prep`.
KENYA_TERMS = re.compile(
    r"\b("
    r"kenya|kenyan|kenyans|"
    r"ruto|raila|gachagua|kasongo|kindiki|murkomen|matiangi|sifuna|"
    r"wantam|tutam|iebc|ncic|"
    r"nairobi|kisumu|mombasa|nakuru|eldoret|kakamega|bungoma|"
    r"luo|luos|kikuyu|kikuyus|kalenjin|kalenjins|luhya|luhyas|"
    r"kamba|kisii|meru|somali|gusii|abagusii|"
    r"mt\.?\s*kenya|murima|wamunyoro|"
    r"odm|uda|dcp|"
    r"githeri|mbogi"
    r")\b",
    re.IGNORECASE,
)
_HANDLE_WORDS = re.compile(r"[_\d]+|(?<=[a-z])(?=[A-Z])")


def load_handles(path: Path) -> frozenset[str]:
    return frozenset(h.strip().lower() for h in Path(path).read_text().splitlines() if h.strip())


def handle_is_kenyan(handle: str, handles: frozenset[str], variant: str) -> bool:
    if variant == "strip":
        return False
    if handle.lower() in handles:
        return True
    return variant == "known_lex" and bool(KENYA_TERMS.search(_HANDLE_WORDS.sub(" ", handle)))


def prepare(text: str | None, variant: str = "strip", handles: frozenset[str] = frozenset()) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return MENTION.sub(
        lambda m: m.group(0) if handle_is_kenyan(m.group(1), handles, variant) else " ", text or "")


def load_for_model(model_dir: Path) -> tuple[str, frozenset[str]]:
    """The preparation a published model was trained with; `strip` when it records none."""
    model_dir = Path(model_dir)
    spec = model_dir / PREP_FILE
    variant = json.loads(spec.read_text())["variant"] if spec.exists() else "strip"
    handles_path = model_dir / HANDLES_FILE
    return variant, load_handles(handles_path) if handles_path.exists() else frozenset()

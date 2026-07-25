from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def load_module():
    path = Path(__file__).with_name("23_opus_partial_merge.py")
    spec = importlib.util.spec_from_file_location("opus_partial_merge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )


def sample_label(post_id: str, label: str = "neither", **extra) -> dict:
    flags = extra.pop("flags", [])
    row = {
        "post_id": post_id,
        "label": label,
        "flags": flags,
        "target_group": "Kikuyu" if "ethnic_targeting" in flags else None,
        "confidence": extra.pop("confidence", "high"),
        "rationale": extra.pop("rationale", "A rationale."),
        "warnings": [],
    }
    row.update(extra)
    return row


def test_collect_labels_unions_automated_and_manual(tmp_path: Path) -> None:
    module = load_module()
    auto = tmp_path / "auto"
    manual = tmp_path / "manual"
    write_jsonl(auto / "chunk_000.jsonl", [sample_label("1", "hate", flags=["ethnic_targeting"])])
    write_jsonl(manual / "batch_000.jsonl", [sample_label("2", "offensive")])

    frame = module.collect_labels(auto, manual)

    assert set(frame["post_id"]) == {"1", "2"}
    assert frame.set_index("post_id").loc["1", "label_origin"] == "automated"
    assert frame.set_index("post_id").loc["2", "label_origin"] == "manual"


def test_collect_labels_rejects_conflicting_duplicates(tmp_path: Path) -> None:
    module = load_module()
    auto = tmp_path / "auto"
    manual = tmp_path / "manual"
    write_jsonl(auto / "chunk_000.jsonl", [sample_label("1", "hate", flags=["ethnic_targeting"])])
    write_jsonl(manual / "batch_000.jsonl", [sample_label("1", "offensive")])

    with pytest.raises(ValueError, match="conflicting labels"):
        module.collect_labels(auto, manual)


def test_merge_partial_preserves_batch_and_prior_split(tmp_path: Path) -> None:
    module = load_module()
    batch = pd.DataFrame(
        {
            "post_id": ["1", "2", "3"],
            "text": ["a", "b", "c"],
            "stratum": ["lexicon", "nli_tail", "p_hate_top"],
            "author_handle": ["x", "y", "z"],
            "created_at": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "p_hate": [0.1, 0.2, 0.9],
            "p_offensive": [0.2, 0.3, 0.1],
            "is_lexicon": [True, False, False],
            "is_nli_tail": [False, True, False],
        }
    )
    labels = pd.DataFrame(
        [
            sample_label("1", "hate", flags=["ethnic_targeting"])
            | {"label_origin": "automated"},
            sample_label("2", "neither") | {"label_origin": "manual"},
        ]
    )
    baseline = pd.DataFrame(
        {
            "post_id": ["1", "2", "3"],
            "label": ["offensive", "neither", "hate"],
            "split": ["train", "challenge", "gold"],
        }
    )

    merged, report = module.merge_partial(
        batch=batch,
        labels=labels,
        baseline=baseline,
        prompt_version="v4",
        prompt_sha256="abc",
        labeller="claude-opus-code",
        model="opus",
    )

    assert len(merged) == 2
    assert set(merged["post_id"]) == {"1", "2"}
    assert merged.set_index("post_id").loc["1", "text"] == "a"
    assert merged.set_index("post_id").loc["1", "split"] == "train"
    assert merged.set_index("post_id").loc["2", "split"] == "challenge"
    assert merged["label_source"].unique().tolist() == ["single_labeller_claude"]
    assert report["rows"] == 2
    assert report["class_counts"]["hate"] == 1
    assert report["prior_split_counts"]["train"] == 1
    assert report["prior_split_counts"]["challenge"] == 1
    assert report["vs_baseline_overlap"]["n"] == 2
    assert report["vs_baseline_overlap"]["label_changed"] == 1
    assert "gold" not in report["prior_split_counts"]

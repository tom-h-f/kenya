# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2", "pyarrow>=18"]
# ///
"""Import manual outbox JSONL into the opus-v4-full claude-opus-code tree.

Rebuilds only missing driver chunks from outbox labels + existing chunk files.
Does not overwrite existing successful chunk_*.jsonl files.

    uv run 22_import_manual_labels.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from _common import OUT

TAG = "opus-v4-full"
LABELLER = "claude-opus-code"
CHUNK_SIZE = 25


def main() -> None:
    root = OUT / "labels" / TAG
    label_dir = root / LABELLER
    label_dir.mkdir(parents=True, exist_ok=True)
    batch = pd.read_parquet(root / "batch.parquet")
    batch["post_id"] = batch["post_id"].astype(str)
    expected = batch["post_id"].tolist()

    labels: dict[str, dict] = {}
    for path in sorted(label_dir.glob("chunk_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                labels[str(row["post_id"])] = row

    outbox = OUT / "opus_v4_full_manual" / "outbox"
    imported = 0
    for path in sorted(outbox.glob("batch_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pid = str(row["post_id"])
            if pid not in labels:
                labels[pid] = row
                imported += 1

    written = 0
    for i in range(0, len(expected), CHUNK_SIZE):
        chunk_ids = expected[i : i + CHUNK_SIZE]
        out_path = label_dir / f"chunk_{i // CHUNK_SIZE:03d}.jsonl"
        if out_path.exists():
            continue
        if not all(pid in labels for pid in chunk_ids):
            missing = [pid for pid in chunk_ids if pid not in labels]
            print(f"skip {out_path.name}: missing {len(missing)}")
            continue
        out_path.write_text(
            "\n".join(json.dumps(labels[pid], ensure_ascii=False) for pid in chunk_ids)
            + "\n"
        )
        written += 1
        print(f"wrote {out_path.name}")

    print(json.dumps({
        "existing_or_imported_labels": len(labels),
        "newly_imported_from_outbox": imported,
        "chunks_written": written,
        "complete": all(pid in labels for pid in expected),
        "labelled": sum(pid in labels for pid in expected),
        "total": len(expected),
    }, indent=2))


if __name__ == "__main__":
    main()

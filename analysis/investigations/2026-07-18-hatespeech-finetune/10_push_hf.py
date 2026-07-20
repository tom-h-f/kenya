# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface_hub>=0.30",
# ]
# ///
"""Push a model dir to a private HF hub repo.

Needs HF_TOKEN in the env. On Colab: set it from the secrets UI first
(os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")). Bare repo names
resolve under the token's namespace.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"not a directory: {model_dir}")

    files = [
        p for p in sorted(model_dir.rglob("*"))
        if p.is_file() and "checkpoints" not in p.relative_to(model_dir).parts
    ]
    total = sum(p.stat().st_size for p in files)
    for p in files:
        print(f"  {p.relative_to(model_dir)}  {p.stat().st_size / 1e6:.1f}MB")
    print(f"{len(files)} files, {total / 1e9:.2f}GB -> {args.repo_id} (private)")
    if args.dry_run:
        print("dry run - nothing uploaded")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    repo = api.create_repo(args.repo_id, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo.repo_id,
        ignore_patterns=["checkpoints/**"],
    )
    print(f"pushed: https://huggingface.co/{repo.repo_id}")


if __name__ == "__main__":
    main()

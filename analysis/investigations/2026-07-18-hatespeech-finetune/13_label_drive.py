# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2",
#     "pyarrow>=18",
# ]
# ///
"""Drive dual-agent labelling of mined candidates through `agy` (Plan A3).

    uv run 13_label_drive.py --pilot 200 --tag pilot     # stratified pilot
    uv run 13_label_drive.py --tag full                  # everything

Two independent model families label identical chunks; agreement/kappa comes
later from 14_label_merge.py. Resumable and idempotent: an existing chunk
output is schema- and ID-validated before it is skipped.

Validation is strict about structure (exact schema, ordered string IDs, typed
fields, and label/flag consistency) and those failures are retried then parked
in out/labels/<tag>/failed/. Flag-consistency oddities (`neither` carrying
`violence_call`) are recorded per row as `warnings` rather than retried -
they are a labelling signal for the merge step, not a transport error, and
retrying them would just burn spend on a model that meant what it said.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from _common import OUT, SEED

HERE = Path(__file__).resolve().parent
PROMPT_DIR = HERE / "prompts"
CHUNK_SIZE = 25
MAX_ATTEMPTS = 3

"""Labeller name -> (cli, model). Two CLIs, deliberately: agy and the Cursor
agent draw on separate quotas, so a run that exhausts one can continue on the
other. Names are kept distinct per model version - never write two model
versions into one labeller bucket, it hides which model produced a label."""
LABELLERS = {
    "gemini-3.1-pro": ("agy", "Gemini 3.1 Pro (Low)"),
    "claude-sonnet-4.6": ("agy", "Claude Sonnet 4.6 (Thinking)"),
    "cursor-sonnet-4.5": ("cursor", "sonnet-4.5"),
    "claude-opus-4.6": ("agy", "claude-opus-4-6-thinking"),
    "claude-opus-code": ("claude", "opus"),
}
DEFAULT_LABELLERS = ["gemini-3.1-pro", "claude-sonnet-4.6"]


def build_cmd(cli: str, model: str, timeout: str) -> list[str]:
    if cli == "agy":
        return ["agy", "-p", "--model", model, "--print-timeout", timeout]
    if cli == "cursor":
        return ["agent", "-p", "--trust", "--model", model,
                "--mode", "ask", "--output-format", "text"]
    if cli == "claude":
        return ["claude", "-p", "--model", model]
    raise ValueError(f"unknown cli: {cli}")


VALID_LABELS = {"hate", "offensive", "neither"}
VALID_FLAGS = {"dehumanisation", "violence_call", "ethnic_targeting", "coded_language"}
VALID_CONFIDENCE = {"high", "medium", "low"}
RESPONSE_FIELDS = {
    "post_id",
    "label",
    "flags",
    "target_group",
    "confidence",
    "rationale",
}
IMMUTABLE_MANIFEST_FIELDS = (
    "tag",
    "rows",
    "input_sha256",
    "prompt_filename",
    "prompt_version",
    "prompt_sha256",
    "labellers",
    "chunk_size",
    # Keep the recorded execution contract simple: resume concurrency is fixed.
    "concurrency",
)


def batch_sha256(df: pd.DataFrame) -> str:
    """Hash every selected value with canonical column and row ordering."""
    columns = sorted(df.columns)
    records = json.loads(
        df.loc[:, columns].to_json(
            orient="records",
            date_format="iso",
            date_unit="ns",
            double_precision=15,
            force_ascii=False,
        )
    )
    payload = json.dumps(
        {"columns": columns, "records": records},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@contextmanager
def atomic_target(path: Path):
    """Yield a same-directory temp path, then durably replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        yield temp_path
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, content: str) -> None:
    with atomic_target(path) as temp_path:
        temp_path.write_text(content)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    with atomic_target(path) as temp_path:
        df.to_parquet(temp_path, index=False)


@contextmanager
def tag_lock(path: Path):
    """Hold a non-blocking OS lock for one tag; the lock file may persist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            holder = handle.read().strip() or "unknown process"
            raise RuntimeError(
                f"label run already locked: {path} (holder {holder})"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def run_manifest(
    *,
    tag: str,
    prompt_path: Path,
    labellers: list[str],
    input_name: str,
    batch: pd.DataFrame,
    created_at: datetime | None = None,
    chunk_size: int = CHUNK_SIZE,
    concurrency: int = 4,
) -> dict:
    """Build a serialisable run manifest without writing run state."""
    match = re.fullmatch(r"label_(v[0-9]+)\.md", prompt_path.name)
    if match is None:
        raise ValueError(f"cannot derive prompt version from {prompt_path.name!r}")
    created_at = created_at or datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")
    created_utc = created_at.astimezone(timezone.utc)
    input_path = Path(input_name)
    return {
        "tag": tag,
        "created_at": created_utc.isoformat().replace("+00:00", "Z"),
        "input_path": str(input_path),
        "input_name": input_path.name,
        "rows": len(batch),
        "input_sha256": batch_sha256(batch),
        "prompt_filename": prompt_path.name,
        "prompt_version": match.group(1),
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "labellers": {
            name: {"cli": LABELLERS[name][0], "model": LABELLERS[name][1]}
            for name in labellers
        },
        "chunk_size": chunk_size,
        "concurrency": concurrency,
    }


def validate_run_manifest(existing: dict, current: dict) -> None:
    """Reject a resume if any recorded run-identity field has changed."""
    for field in IMMUTABLE_MANIFEST_FIELDS:
        if existing.get(field) != current.get(field):
            raise ValueError(
                f"manifest conflict for {field}: "
                f"existing={existing.get(field)!r}, current={current.get(field)!r}"
            )


def stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Proportional-by-stratum subsample, at least one row per stratum."""
    frac = n / len(df)
    picks = [
        g.sample(max(1, round(len(g) * frac)), random_state=SEED)
        for _, g in df.groupby("stratum", sort=False)
    ]
    return pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)


def write_chunks(df: pd.DataFrame, chunk_dir: Path) -> list[Path]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(0, len(df), CHUNK_SIZE):
        path = chunk_dir / f"chunk_{i // CHUNK_SIZE:03d}.jsonl"
        rows = df.iloc[i : i + CHUNK_SIZE]
        payload = "\n".join(
            json.dumps({"post_id": r.post_id, "text": r.text}, ensure_ascii=False)
            for r in rows.itertuples()
        )
        if not path.exists() or path.read_text() != payload:
            atomic_write_text(path, payload)
        paths.append(path)
    return paths


def parse_response(
    raw: str,
    expected_ids: list[str],
    *,
    allow_warnings: bool = False,
) -> list[dict]:
    """Parse JSONL, tolerating fences and surrounding prose.

    Agent CLIs differ in how literally they take "no prose": agy obeys, the
    Cursor agent prefixes a sentence and fences the block. Skipping non-`{`
    lines absorbs that without weakening validation - the ordered post_id
    check below still fails loudly if any row is missing, extra or malformed.
    """
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        rows.append(json.loads(line))

    for r in rows:
        if not isinstance(r, dict):
            raise ValueError(f"response row must be an object, got {type(r).__name__}")
        expected_fields = RESPONSE_FIELDS | ({"warnings"} if allow_warnings else set())
        missing = expected_fields - r.keys()
        extra = r.keys() - expected_fields
        if missing:
            raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
        if extra:
            raise ValueError(f"extra fields: {', '.join(sorted(extra))}")
        if not isinstance(r["post_id"], str):
            raise ValueError(f"bad post_id {r['post_id']!r}: must be a string")
        if not isinstance(r["label"], str) or r["label"] not in VALID_LABELS:
            raise ValueError(f"bad label {r['label']!r}")
        flags = r["flags"]
        if not isinstance(flags, list):
            raise ValueError(f"bad flags {flags!r}")
        if any(not isinstance(flag, str) for flag in flags):
            raise ValueError(f"bad flags {flags!r}")
        if len(flags) != len(set(flags)):
            raise ValueError(f"duplicate flags {flags!r}")
        if set(flags) - VALID_FLAGS:
            raise ValueError(f"bad flags {flags!r}")
        if (
            not isinstance(r["confidence"], str)
            or r["confidence"] not in VALID_CONFIDENCE
        ):
            raise ValueError(f"bad confidence {r['confidence']!r}")
        if not isinstance(r["rationale"], str) or not r["rationale"].strip():
            raise ValueError(f"bad rationale {r['rationale']!r}")
        target_group = r["target_group"]
        if target_group is not None and (
            not isinstance(target_group, str) or not target_group.strip()
        ):
            raise ValueError(f"bad target_group {target_group!r}")
        ethnic_targeting = "ethnic_targeting" in flags
        if (r["label"] == "hate") != ethnic_targeting:
            raise ValueError(
                "hate label and ethnic_targeting flag must appear together"
            )
        if target_group is not None and not ethnic_targeting:
            raise ValueError("target_group requires ethnic_targeting")
        warnings = []
        if r["label"] == "neither" and "violence_call" in flags:
            warnings.append("neither_with_violence_call")
        if allow_warnings and r["warnings"] != warnings:
            raise ValueError(
                f"bad warnings {r['warnings']!r}: expected derived warnings {warnings!r}"
            )
        r["flags"] = flags
        r["warnings"] = warnings

    got = [r["post_id"] for r in rows]
    if got != expected_ids:
        raise ValueError(f"post_id order mismatch: got {got!r}, want {expected_ids!r}")
    return rows


def chunk_expected_ids(chunk: Path) -> list[str]:
    return [
        json.loads(line)["post_id"]
        for line in chunk.read_text().splitlines()
        if line
    ]


def pending_chunks(chunks: list[Path], out_dir: Path) -> list[Path]:
    todo = []
    for chunk in chunks:
        output_path = out_dir / f"{chunk.stem}.jsonl"
        if not output_path.exists():
            todo.append(chunk)
            continue
        try:
            parse_response(
                output_path.read_text(),
                chunk_expected_ids(chunk),
                allow_warnings=True,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid existing output {output_path}: {exc}") from exc
    return todo


def label_chunk(labeller: str, cli: str, model: str, chunk: Path, out_dir: Path,
                failed_dir: Path, timeout: str, prompt_path: Path) -> dict:
    prompt = f"{prompt_path.read_text()}\n\n{chunk.read_text()}"
    expected = chunk_expected_ids(chunk)
    start = time.time()
    last_error = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        proc = subprocess.run(
            build_cmd(cli, model, timeout),
            input=prompt, capture_output=True, text=True,
        )
        try:
            if proc.returncode != 0:
                raise ValueError(f"{cli} exit {proc.returncode}: {proc.stderr[:200]}")
            rows = parse_response(proc.stdout, expected)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = f"attempt {attempt}: {exc}"
            print(f"  [{labeller}/{chunk.stem}] {last_error}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            out_dir / f"{chunk.stem}.jsonl",
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
        )
        elapsed = time.time() - start
        print(f"  [{labeller}/{chunk.stem}] ok, {len(rows)} rows, {elapsed:.0f}s")
        return {"chunk": chunk.stem, "rows": len(rows), "seconds": round(elapsed, 1),
                "attempts": attempt}

    failed_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        failed_dir / f"{labeller}-{chunk.stem}.txt",
        f"{last_error}\n\n--- last stdout ---\n{proc.stdout}"
    )
    print(f"  [{labeller}/{chunk.stem}] PARKED after {MAX_ATTEMPTS} attempts")
    return {"chunk": chunk.stem, "failed": True, "error": last_error,
            "seconds": round(time.time() - start, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="label_batch_001.parquet")
    ap.add_argument("--tag", required=True, help="run name; isolates chunks + labels")
    ap.add_argument("--pilot", type=int, default=None,
                    help="stratified subsample of this many rows")
    ap.add_argument("--limit-chunks", type=int, default=None, help="smoke: first N chunks")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--print-timeout", default="10m")
    ap.add_argument("--labellers", default=",".join(DEFAULT_LABELLERS))
    ap.add_argument("--prompt-version", default="v2")
    args = ap.parse_args()

    prompt_path = PROMPT_DIR / f"label_{args.prompt_version}.md"
    if not prompt_path.exists():
        raise SystemExit(f"{prompt_path} missing")
    print(f"prompt: {prompt_path.name}")

    input_path = OUT / args.input
    df = pd.read_parquet(input_path)
    if args.pilot:
        df = stratified_sample(df, args.pilot)
    print(f"{len(df)} rows -> {args.tag}")
    print(df["stratum"].value_counts().to_string())

    labeller_names = args.labellers.split(",")
    selected_labellers = {name: LABELLERS[name] for name in labeller_names}
    root = OUT / "labels" / args.tag
    lock_path = OUT / "labels" / f".{args.tag}.lock"
    with tag_lock(lock_path):
        manifest_path = root / "manifest.json"
        manifest = run_manifest(
            tag=args.tag,
            prompt_path=prompt_path,
            labellers=labeller_names,
            input_name=str(input_path),
            batch=df,
            chunk_size=CHUNK_SIZE,
            concurrency=args.concurrency,
        )
        if manifest_path.exists():
            validate_run_manifest(json.loads(manifest_path.read_text()), manifest)
        else:
            root.mkdir(parents=True, exist_ok=True)
            atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")

        chunks = write_chunks(df, OUT / "chunks" / args.tag)
        if args.limit_chunks:
            chunks = chunks[: args.limit_chunks]
        atomic_write_parquet(df, root / "batch.parquet")
        print(f"{len(chunks)} chunks of {CHUNK_SIZE}")

        for labeller, (cli, model) in selected_labellers.items():
            out_dir, failed_dir = root / labeller, root / "failed"
            state_path = root / f"state-{labeller}.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}

            todo = pending_chunks(chunks, out_dir)
            print(
                f"\n{labeller} ({model}): "
                f"{len(todo)} of {len(chunks)} chunks to run"
            )
            if not todo:
                continue

            start = time.time()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                results = list(pool.map(
                    lambda c: label_chunk(
                        labeller, cli, model, c, out_dir, failed_dir,
                        args.print_timeout, prompt_path,
                    ),
                    todo,
                ))

            for r in results:
                state[r["chunk"]] = r
            atomic_write_text(state_path, json.dumps(state, indent=2) + "\n")

            done = [r for r in results if not r.get("failed")]
            failed = len(results) - len(done)
            wall = time.time() - start
            secs = [r["seconds"] for r in done]
            print(
                f"{labeller}: {len(done)} ok, {failed} parked, "
                f"{wall / 60:.1f} min wall"
            )
            if secs:
                print(
                    f"  per chunk: mean {sum(secs) / len(secs):.0f}s, "
                    f"max {max(secs):.0f}s"
                )
                print(
                    f"  throughput: "
                    f"{len(done) * CHUNK_SIZE / wall * 60:.0f} posts/min"
                )


if __name__ == "__main__":
    main()

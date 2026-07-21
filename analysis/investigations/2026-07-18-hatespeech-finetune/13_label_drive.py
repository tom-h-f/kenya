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
later from 14_label_merge.py. Resumable and idempotent: a chunk already
present in the labeller's state file is skipped, so re-running costs nothing.

Validation is strict about structure (JSONL parses, every input post_id back
exactly once, label/flags in enum) and those failures are retried then parked
in out/labels/<tag>/failed/. Flag-consistency oddities (`neither` carrying
`violence_call`) are recorded per row as `warnings` rather than retried -
they are a labelling signal for the merge step, not a transport error, and
retrying them would just burn spend on a model that meant what it said.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
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
    "claude-opus-4.6": ("agy", "Claude Opus 4.6 (Thinking)"),
    "claude-opus-code": ("claude", "opus"),
}
DEFAULT_LABELLERS = ["gemini-3.1-pro", "claude-sonnet-4.6"]


def build_cmd(cli: str, model: str, prompt: str, timeout: str) -> list[str]:
    if cli == "agy":
        return ["agy", "-p", prompt, "--model", model, "--print-timeout", timeout]
    if cli == "cursor":
        return ["agent", "-p", "--trust", "--model", model,
                "--mode", "ask", "--output-format", "text", prompt]
    if cli == "claude":
        return ["claude", "-p", prompt, "--model", model]
    raise ValueError(f"unknown cli: {cli}")


VALID_LABELS = {"hate", "offensive", "neither"}
VALID_FLAGS = {"dehumanisation", "violence_call", "ethnic_targeting", "coded_language"}
VALID_CONFIDENCE = {"high", "medium", "low"}
REQUIRED_RESPONSE_FIELDS = {"confidence", "rationale", "target_group"}


def run_manifest(
    *,
    tag: str,
    prompt_path: Path,
    labellers: list[str],
    input_name: str,
    rows: int,
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
        "rows": rows,
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
            path.write_text(payload)
        paths.append(path)
    return paths


def parse_response(raw: str, expected_ids: list[str]) -> list[dict]:
    """Parse JSONL, tolerating fences and surrounding prose.

    Agent CLIs differ in how literally they take "no prose": agy obeys, the
    Cursor agent prefixes a sentence and fences the block. Skipping non-`{`
    lines absorbs that without weakening validation - the post_id set check
    below still fails loudly if any row is missing, extra or malformed.
    """
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        rows.append(json.loads(line))

    got = [r.get("post_id") for r in rows]
    if sorted(map(str, got)) != sorted(expected_ids):
        raise ValueError(f"post_id mismatch: got {len(got)}, want {len(expected_ids)}")

    for r in rows:
        if r.get("label") not in VALID_LABELS:
            raise ValueError(f"bad label {r.get('label')!r}")
        flags = r.get("flags") or []
        if not isinstance(flags, list) or set(flags) - VALID_FLAGS:
            raise ValueError(f"bad flags {flags!r}")
        missing = REQUIRED_RESPONSE_FIELDS - r.keys()
        if missing:
            raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
        if r["confidence"] not in VALID_CONFIDENCE:
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
        r["flags"] = flags
        r["warnings"] = warnings
    return rows


def label_chunk(labeller: str, cli: str, model: str, chunk: Path, out_dir: Path,
                failed_dir: Path, timeout: str, prompt_path: Path) -> dict:
    prompt = f"{prompt_path.read_text()}\n\n{chunk.read_text()}"
    expected = [json.loads(l)["post_id"] for l in chunk.read_text().splitlines() if l]
    start = time.time()
    last_error = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        proc = subprocess.run(
            build_cmd(cli, model, prompt, timeout),
            capture_output=True, text=True, stdin=subprocess.DEVNULL,
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
        (out_dir / f"{chunk.stem}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
        )
        elapsed = time.time() - start
        print(f"  [{labeller}/{chunk.stem}] ok, {len(rows)} rows, {elapsed:.0f}s")
        return {"chunk": chunk.stem, "rows": len(rows), "seconds": round(elapsed, 1),
                "attempts": attempt}

    failed_dir.mkdir(parents=True, exist_ok=True)
    (failed_dir / f"{labeller}-{chunk.stem}.txt").write_text(
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

    root = OUT / "labels" / args.tag
    chunks = write_chunks(df, OUT / "chunks" / args.tag)
    if args.limit_chunks:
        chunks = chunks[: args.limit_chunks]
    root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / "batch.parquet", index=False)
    print(f"{len(chunks)} chunks of {CHUNK_SIZE}")

    labeller_names = args.labellers.split(",")
    selected_labellers = {name: LABELLERS[name] for name in labeller_names}
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        manifest = run_manifest(
            tag=args.tag,
            prompt_path=prompt_path,
            labellers=labeller_names,
            input_name=str(input_path),
            rows=len(df),
            chunk_size=CHUNK_SIZE,
            concurrency=args.concurrency,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    for labeller, (cli, model) in selected_labellers.items():
        out_dir, failed_dir = root / labeller, root / "failed"
        state_path = root / f"state-{labeller}.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}

        todo = [c for c in chunks if not (out_dir / f"{c.stem}.jsonl").exists()]
        print(f"\n{labeller} ({model}): {len(todo)} of {len(chunks)} chunks to run")
        if not todo:
            continue

        start = time.time()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(
                lambda c: label_chunk(labeller, cli, model, c, out_dir, failed_dir,
                                      args.print_timeout, prompt_path),
                todo,
            ))

        for r in results:
            state[r["chunk"]] = r
        state_path.write_text(json.dumps(state, indent=2))

        done = [r for r in results if not r.get("failed")]
        failed = len(results) - len(done)
        wall = time.time() - start
        secs = [r["seconds"] for r in done]
        print(f"{labeller}: {len(done)} ok, {failed} parked, {wall / 60:.1f} min wall")
        if secs:
            print(f"  per chunk: mean {sum(secs) / len(secs):.0f}s, max {max(secs):.0f}s")
            print(f"  throughput: {len(done) * CHUNK_SIZE / wall * 60:.0f} posts/min")


if __name__ == "__main__":
    main()

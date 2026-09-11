"""A2's size-matched adjudication, judged by headless Claude Code instead of the API.

    cd analysis && uv run python investigations/2026-09-11-textsim-sensitivity/05_a2_headless.py \\
        out/a2_matched/a2_sample.parquet --dry-run
    ... --probe                  # what does the reader load besides the rubric?
    ... --limit 1                # one real case, not persisted
    ... --persist                # all cases, verdicts written to R2

There is no ANTHROPIC_API_KEY for this project - no Modal `anthropic` secret, and
by the 2026-09-08 decision none on the hosts - so the reader is `claude -p` under
the user's Claude Code login, the route `13_label_drive.py` already uses for
labelling. Everything the reader is told comes from `kma.adjudicate`: `SYSTEM` as
the system prompt, `prompt(packet)` as the only message, `_parse` for the reply.

Blinding is enforced in the reader's environment, not only in the prompt. Each
case is a fresh process started in an empty temporary directory, so no project
memory loads (the kenya memory describes v1's and v2's findings), with no tools,
no MCP servers, hooks off, no session persistence, and `--setting-sources local`
so the user's global CLAUDE.md and rules stay out too. It sees one dossier, the
rubric, and the harness's own session context (email, environment, date). Which method produced the case stays in `a2_key.csv`, which this script
never reads. `--probe` prints what the reader's own init message says it loaded.
"""

import argparse
import json
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from kma import adjudicate, coordination, dossier
from kma.db import connect

PROBE = (
    "Before anything else: list every instruction, preference, memory or piece of "
    "context you were given other than this message and your system prompt. Quote "
    "short items verbatim and name the source of long ones. If there are none, reply "
    "with the single word NONE."
)


def reader_command(model: str) -> list[str]:
    return [
        "claude", "-p",
        "--model", model,
        "--system-prompt", adjudicate.SYSTEM,
        "--tools", "",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}',
        "--settings", '{"disableAllHooks": true}',
        # Without this the reader loads the user's global CLAUDE.md and rules
        # (measured 2026-09-11 with --probe); with it, only the harness's own
        # session context remains - email, environment, date, model name.
        "--setting-sources", "local",
        "--no-session-persistence",
        "--output-format", "json",
    ]


def ask(text: str, model: str, workdir: str, timeout: int) -> tuple[dict, dict]:
    """One fresh reader process. Returns (result message, init message)."""
    proc = subprocess.run(
        reader_command(model), input=text, capture_output=True, text=True, cwd=workdir, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr.strip()[-400:]}")
    parsed = json.loads(proc.stdout)
    # Claude Code 2.1.x prints the whole message list; older versions print only
    # the final result object. Either way the verdict lives in the `result` entry.
    messages = [m for m in (parsed if isinstance(parsed, list) else [parsed]) if isinstance(m, dict)]
    results = [m for m in messages if m.get("type") == "result"]
    if not results:
        raise RuntimeError(f"no result message in claude output: {proc.stdout[:300]}")
    result = results[-1]
    if result.get("is_error"):
        raise RuntimeError(f"claude reported an error: {str(result.get('result'))[:400]}")
    init = next((m for m in messages if m.get("type") == "system" and m.get("subtype") == "init"), {})
    return result, init


def judge(packet: dict, model: str, workdir: str, timeout: int, replies: Path) -> dict:
    case = packet.get("cluster_id")
    try:
        result, _ = ask(adjudicate.prompt(packet), model, workdir, timeout)
        (replies / f"{case}.txt").write_text(result["result"])
        verdict = adjudicate._parse(result["result"], case)
        # Claude Code also bills a small background model; the verdict is the
        # requested model's, so that is what `reader_model` records.
        verdict["reader_model"] = model
        verdict["models_billed"] = ",".join(result.get("modelUsage") or {})
        verdict["cost_usd"] = result.get("total_cost_usd")
        return verdict
    except Exception as exc:  # one failed case must not cost the rest
        return {
            "cluster_id": case,
            "cluster_type": "unclear",
            "kenya_relevant": None,
            "confidence": "low",
            "rationale": f"adjudication failed: {exc}",
            "error": f"{type(exc).__name__}: {exc}",
        }


def probe(model: str, timeout: int) -> None:
    with tempfile.TemporaryDirectory() as empty:
        result, init = ask(PROBE, model, empty, timeout)
    print("== what the reader's init message says it loaded ==")
    for field in ("model", "cwd", "memory_paths", "mcp_servers", "output_style", "agents", "apiKeySource"):
        print(f"  {field}: {init.get(field)!r}")
    print("== the reader's own answer ==")
    print(result["result"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sample", type=Path)
    ap.add_argument("--model", default=adjudicate.MODEL)
    ap.add_argument("--dry-run", action="store_true", help="build dossiers and prompts, call nothing")
    ap.add_argument("--probe", action="store_true", help="show what context the reader loads, then stop")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--persist", action="store_true", help="write the verdicts to R2")
    args = ap.parse_args()

    if args.probe:
        probe(args.model, args.timeout)
        return

    out = args.sample.parent / "headless"
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    (out / "replies").mkdir(parents=True, exist_ok=True)

    con = connect()
    members = pd.read_parquet(args.sample)[["cluster_id", "author_id"]]
    cases = sorted(members["cluster_id"].unique().tolist())
    if args.limit:
        cases = cases[: args.limit]
    start = time.perf_counter()
    packets = dossier.build(con, members, cluster_ids=cases)
    print(f"built {len(packets)} dossiers in {time.perf_counter() - start:.0f}s")
    for packet in packets:
        (out / "prompts" / f"{packet['cluster_id']}.txt").write_text(adjudicate.prompt(packet))

    if args.dry_run:
        for packet in packets:
            print(f"  case {packet['cluster_id']:>2}: {packet.get('size'):>3} accounts, "
                  f"{len(packet.get('shared_objects', []))} shared objects, "
                  f"{len(adjudicate.prompt(packet)):,} prompt chars")
        return

    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as empty, ThreadPoolExecutor(max_workers=args.workers) as pool:
        verdicts = list(pool.map(
            lambda packet: judge(packet, args.model, empty, args.timeout, out / "replies"), packets
        ))
    frame = pd.DataFrame(verdicts)
    adjudicator = f"claude-code-headless/{args.model}"
    frame["adjudicator"] = adjudicator
    frame["unit"] = "sample"
    frame["sample"] = str(args.sample)
    frame.to_parquet(out / f"verdicts{'_limit' + str(args.limit) if args.limit else ''}.parquet")

    failed = int(frame["error"].notna().sum()) if "error" in frame else 0
    cost = float(frame["cost_usd"].sum()) if "cost_usd" in frame else 0.0
    print(f"judged {len(frame)} cases in {time.perf_counter() - start:.0f}s, {failed} failed, "
          f"reported cost ${cost:.2f}")
    print(frame["cluster_type"].value_counts().to_string())

    if args.persist:
        if failed:
            raise SystemExit("not persisting a run with failed cases - rerun them first")
        key = coordination.persist_verdicts(con, frame, members, adjudicator=adjudicator)
        print(f"persisted verdicts: {key}")


if __name__ == "__main__":
    main()

"""Label the drawn posts kenya / offdomain / unclear with headless Claude Code.

    cd analysis && uv run python investigations/2026-09-12-relevance-classifier/02_label.py --limit 200
    ... 02_label.py                       # everything in out/to_label.parquet

The definitions are `kma.measure_eval`'s, word for word, so these labels mean
what the B2 labels meant. The labeller is claude-opus-5, the model whose labels
agreed with Tom's blind pass on 90 of 100 posts (kappa 0.83, no disagreement on
any Kenya call). The reader is isolated as in the A2 runs: empty working
directory, no tools, MCP servers, hooks or user settings.

Chunks of `CHUNK` posts, one fresh process each, several in parallel. Each
chunk's reply is validated - every id back, in order, with a known label - and
written to `out/labels/`; a rerun skips chunks already done, so an interrupted
pass resumes.
"""

import argparse
import json
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CHUNK = 50
LABELS = ("kenya", "offdomain", "unclear")
MODEL = "claude-opus-5"

SYSTEM = """You label social media posts, mostly Kenyan English, Swahili and Sheng, for whether they \
are about Kenya.

  kenya      the post is about Kenyan politics, society or the election, including posts that
             never say a Kenyan word but are unmistakably about it in context
  offdomain  the post is about somewhere else
  unclear    you cannot tell without more context

Judge each post on its own text."""


def command() -> list[str]:
    return ["claude", "-p", "--model", MODEL, "--system-prompt", SYSTEM, "--tools", "",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}',
            "--settings", '{"disableAllHooks": true}', "--setting-sources", "local",
            "--no-session-persistence", "--output-format", "json"]


def parse(reply: str, ids: list[str]) -> list[str]:
    body = reply[reply.find("[") : reply.rfind("]") + 1]
    rows = json.loads(body)
    got = [str(r["id"]) for r in rows]
    if got != ids:
        raise ValueError(f"ids back {len(got)}, expected {len(ids)}, or out of order")
    labels = [r["label"] for r in rows]
    if bad := set(labels) - set(LABELS):
        raise ValueError(f"unknown labels {bad}")
    return labels


def label_chunk(chunk: pd.DataFrame, path: Path, workdir: str, attempts: int = 3) -> str:
    ids = chunk["post_id"].tolist()
    body = "\n\n".join(f"ID {r.post_id}\n{str(r.text).strip()}" for r in chunk.itertuples())
    prompt = (f"{body}\n\nReturn ONLY a JSON array, one object per post, in the order given: "
              '{"id": "<the ID>", "label": "kenya" | "offdomain" | "unclear"}')
    error = ""
    for _ in range(attempts):
        proc = subprocess.run(command(), input=prompt, capture_output=True, text=True, cwd=workdir, timeout=900)
        try:
            if proc.returncode:
                raise RuntimeError(proc.stderr.strip()[-300:])
            out = json.loads(proc.stdout)
            result = next(m for m in (out if isinstance(out, list) else [out]) if m.get("type") == "result")
            labels = parse(result["result"], ids)
            chunk.assign(label=labels, labeller=MODEL)[["post_id", "label", "labeller"]].to_parquet(path)
            return "ok"
        except Exception as exc:  # retried, then reported; one bad chunk must not stop the pass
            error = f"{type(exc).__name__}: {exc}"
    return f"failed: {error}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    posts = pd.read_parquet(HERE / "out" / "to_label.parquet")
    if args.limit:
        posts = posts.head(args.limit)
    out = HERE / "out" / "labels"
    out.mkdir(parents=True, exist_ok=True)
    chunks = [(k, posts.iloc[s : s + CHUNK]) for k, s in enumerate(range(0, len(posts), CHUNK))]
    todo = [(k, c) for k, c in chunks if not (out / f"chunk_{k:04d}.parquet").exists()]
    print(f"{len(posts):,} posts in {len(chunks)} chunks; {len(todo)} to do", flush=True)

    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as empty, ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda kc: (kc[0], label_chunk(kc[1], out / f"chunk_{kc[0]:04d}.parquet", empty)),
                                todo))
    failed = [(k, r) for k, r in results if r != "ok"]
    print(f"{len(todo) - len(failed)} chunks labelled in {time.perf_counter() - start:.0f}s, {len(failed)} failed")
    for k, r in failed:
        print(f"  chunk {k}: {r}")

    done = pd.concat([pd.read_parquet(p) for p in sorted(out.glob("chunk_*.parquet"))], ignore_index=True)
    merged = posts.merge(done, on="post_id")
    merged.to_parquet(HERE / "out" / "labelled.parquet")
    print(f"labelled {len(merged):,}: {merged['label'].value_counts().to_dict()}")
    print(pd.crosstab(merged["bucket"], merged["label"]).to_string())


if __name__ == "__main__":
    main()

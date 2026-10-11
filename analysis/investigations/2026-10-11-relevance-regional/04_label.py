"""Label a pool of posts kenya / offdomain / unclear, plus the country, with headless Claude Code.

    cd analysis && uv run python investigations/2026-10-11-relevance-regional/04_label.py out/pool_train.parquet --name train

Same reader as `2026-09-12-relevance-classifier/02_label.py`: claude-opus-5, the
definitions word for word, empty working directory, no tools, MCP servers, hooks
or user settings. The one addition is a `country` field, so the regional false
positive rate can be broken out by country. Chunks of 50, validated (every id
back, in order, known label), cached under `out/labels_<name>/`; a rerun skips
chunks already done.
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

Judge each post on its own text. Also give the country the post is about, lowercase one word or
two joined by an underscore (kenya, tanzania, uganda, nigeria, south_africa, usa, uk, india, ...),
or "none" when it is about no country."""


def command() -> list[str]:
    return ["claude", "-p", "--model", MODEL, "--system-prompt", SYSTEM, "--tools", "",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}',
            "--settings", '{"disableAllHooks": true}', "--setting-sources", "local",
            "--no-session-persistence", "--output-format", "json"]


def parse(reply: str, ids: list[str]) -> list[tuple[str, str]]:
    rows = json.loads(reply[reply.find("[") : reply.rfind("]") + 1])
    if [str(r["id"]) for r in rows] != ids:
        raise ValueError(f"ids back {len(rows)}, expected {len(ids)}, or out of order")
    if bad := {r["label"] for r in rows} - set(LABELS):
        raise ValueError(f"unknown labels {bad}")
    return [(r["label"], str(r.get("country", "none")).lower()) for r in rows]


def label_chunk(chunk: pd.DataFrame, path: Path, workdir: str, attempts: int = 3) -> str:
    ids = chunk["post_id"].tolist()
    body = "\n\n".join(f"ID {r.post_id}\n{str(r.text).strip()}" for r in chunk.itertuples())
    prompt = (f"{body}\n\nReturn ONLY a JSON array, one object per post, in the order given: "
              '{"id": "<the ID>", "label": "kenya" | "offdomain" | "unclear", "country": "<country>"}')
    error = ""
    for _ in range(attempts):
        proc = subprocess.run(command(), input=prompt, capture_output=True, text=True, cwd=workdir, timeout=900)
        try:
            if proc.returncode:
                raise RuntimeError(proc.stderr.strip()[-300:])
            out = json.loads(proc.stdout)
            result = next(m for m in (out if isinstance(out, list) else [out]) if m.get("type") == "result")
            got = parse(result["result"], ids)
            chunk.assign(label=[g[0] for g in got], country=[g[1] for g in got], labeller=MODEL)[
                ["post_id", "label", "country", "labeller"]].to_parquet(path)
            return "ok"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if len(chunk) > 1:
        # A chunk the reader answers short or out of order three times running is
        # split and asked again in halves, so one odd post cannot hold up fifty.
        half = len(chunk) // 2
        parts = [path.with_suffix(f".{i}.tmp") for i in (0, 1)]
        results = [label_chunk(c, q, workdir, attempts) for c, q in zip((chunk.iloc[:half], chunk.iloc[half:]), parts)]
        if all(r == "ok" for r in results):
            pd.concat([pd.read_parquet(q) for q in parts]).to_parquet(path)
            for q in parts:
                q.unlink()
            return "ok"
        return f"failed in halves: {results}"
    return f"failed: {error}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pool", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    posts = pd.read_parquet(args.pool)
    posts["post_id"] = posts["post_id"].astype(str)
    if args.limit:
        posts = posts.head(args.limit)
    out = HERE / "out" / f"labels_{args.name}"
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
    merged.to_parquet(HERE / "out" / f"labelled_{args.name}.parquet")
    print(f"labelled {len(merged):,}: {merged['label'].value_counts().to_dict()}")
    print(pd.crosstab(merged["stratum"], merged["label"]).to_string() if "stratum" in merged else "")


if __name__ == "__main__":
    main()

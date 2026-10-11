"""Blind labels for the Tanzanian-politics check: 200 posts, 10 per headless call, 20 calls.

    uv run python 41_tz_label.py build     # writes out/tz_batches/*.txt and out/tz_key.parquet
    uv run python 41_tz_label.py run       # calls claude -p per batch, writes out/tz_replies/*.json
    uv run python 41_tz_label.py report    # joins labels to persisted model scores

The reader is the isolated headless setup of 2026-09-11-textsim-sensitivity/05_a2_headless.py:
empty cwd, no tools, no MCP, hooks off, no session persistence, user settings not loaded.
It sees post text only: no scores, no keyword sets, no community ids.
"""

import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "out"
MODEL = "claude-sonnet-5"
N_EACH = 100
PER_CALL = 10
SYSTEM = (
    "You label short social media posts. For each numbered post choose exactly one label:\n"
    "kenya: the post is about Kenyan politics, government, public figures, events or Kenyan "
    "audiences, or explicitly ties its subject to Kenya.\n"
    "tanzania_only: the post is about Tanzania (its politics, government, courts, protests, "
    "elections) with no link to Kenya.\n"
    "other: about another country or topic, or too little content to tell.\n"
    "Judge only from the text. Reply with one JSON array and nothing else: "
    '[{"i": 1, "label": "kenya"}, ...].'
)


def build() -> None:
    hits = pd.read_parquet(OUT / "tz_hits.parquet")
    mem = pd.read_parquet(OUT / "c8_member_posts.parquet")
    a = hits[~hits["kenya_anchor"] & ~hits["is_retweet"].astype(bool)].drop_duplicates("post_id")
    a = a.sample(min(N_EACH, len(a)), random_state=0).assign(source="tz_keyword")
    b = mem[~mem["is_retweet"].astype(bool) & ~mem["post_id"].isin(a["post_id"])].drop_duplicates("post_id")
    b = b.sample(min(N_EACH, len(b)), random_state=0).assign(source="c8_member")
    key = pd.concat([a, b], ignore_index=True)[["post_id", "text", "p", "source"]]
    key = key.sample(frac=1, random_state=1).reset_index(drop=True)
    key["batch"] = key.index // PER_CALL
    key["i"] = key.index % PER_CALL + 1
    key.to_parquet(OUT / "tz_key.parquet")
    (OUT / "tz_batches").mkdir(exist_ok=True)
    for batch, g in key.groupby("batch"):
        body = "\n\n".join(f"[{r.i}] {re.sub(r'\\s+', ' ', r.text)}" for r in g.itertuples())
        (OUT / "tz_batches" / f"{batch}.txt").write_text(body)
    print(key["source"].value_counts().to_dict(), key["batch"].nunique(), "batches")


def command() -> list[str]:
    return [
        "claude", "-p", "--model", MODEL, "--system-prompt", SYSTEM, "--tools", "",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}',
        "--settings", '{"disableAllHooks": true}', "--setting-sources", "local",
        "--no-session-persistence", "--output-format", "json",
    ]


def ask(batch: Path, workdir: str) -> None:
    target = OUT / "tz_replies" / f"{batch.stem}.json"
    if target.exists():
        return
    proc = subprocess.run(command(), input=batch.read_text(), capture_output=True, text=True,
                          cwd=workdir, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-300:])
    parsed = json.loads(proc.stdout)
    msgs = [m for m in (parsed if isinstance(parsed, list) else [parsed]) if isinstance(m, dict)]
    result = [m for m in msgs if m.get("type") == "result"][-1]
    target.write_text(json.dumps({"result": result["result"], "cost": result.get("total_cost_usd")}))


def run() -> None:
    (OUT / "tz_replies").mkdir(exist_ok=True)
    batches = sorted((OUT / "tz_batches").glob("*.txt"), key=lambda p: int(p.stem))
    with tempfile.TemporaryDirectory() as empty, ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda b: ask(b, empty), batches))


def report() -> None:
    key = pd.read_parquet(OUT / "tz_key.parquet")
    labels = {}
    cost = 0.0
    for path in (OUT / "tz_replies").glob("*.json"):
        d = json.loads(path.read_text())
        cost += d["cost"] or 0
        arr = json.loads(re.search(r"\[.*\]", d["result"], re.S).group(0))
        for item in arr:
            labels[(int(path.stem), int(item["i"]))] = item["label"]
    key["label"] = [labels.get((b, i)) for b, i in zip(key["batch"], key["i"])]
    key.to_parquet(OUT / "tz_labelled.parquet")
    key["kenya_model"] = key["p"] >= 0.5
    print("reported cost", round(cost, 2), "unlabelled", int(key["label"].isna().sum()))
    print(pd.crosstab(key["label"], key["kenya_model"], margins=True))
    print(pd.crosstab([key["source"], key["label"]], key["kenya_model"]))
    t = key[key["label"] == "tanzania_only"]
    print("tanzania_only:", len(t), "scored Kenya by model:", int(t["kenya_model"].sum()),
          "rate", round(float(t["kenya_model"].mean()), 3), "mean p", round(float(t["p"].mean()), 3))
    k = key[key["label"] == "kenya"]
    print("kenya:", len(k), "scored Kenya:", int(k["kenya_model"].sum()))


if __name__ == "__main__":
    {"build": build, "run": run, "report": report}[sys.argv[1]]()

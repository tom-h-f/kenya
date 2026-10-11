"""Blind read of the text-only communities an overlap floor leaves, one headless call each.

    uv run python 07_read_new.py build char40.45 char40.30    # prompts, shuffle keys, which arms/communities
    uv run python 07_read_new.py run
    uv run python 07_read_new.py report

Same isolation and rubric as 2026-10-11-text-only-adjudication/02_read.py (empty cwd, no tools, no MCP, hooks
off, no session persistence, user settings not loaded; rubric copied verbatim). The reader sees post text, time
and how many distinct accounts in the batch posted near-identical text (word-set Jaccard >= 0.6 after stripping
urls and mentions, direct partners only). Never community ids, floors, scores, handles or account ids.
A community is text-only when its edges are all text (share_text >= 0.95) and it is not one of the nine already
adjudicated (membership Jaccard > 0.5 with their bundle members).
"""
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from common import WINDOW, wdir

OUT = Path(__file__).parent / "out"
BUNDLE = Path(os.environ["BUNDLE"])
MODEL = "claude-sonnet-5"
NOISE = re.compile(r"https?://\S+|@\w+")
MAX_POSTS = 16

SYSTEM = """You read short batches of posts from X (Twitter) and describe the TEXT PATTERN they show. You are not asked whether any account is real, fake, paid or coordinated in fact. You only describe what the texts look like.

Each batch is a set of posts that a retrieval step grouped together, so expect that some are related and some are not. Each post is shown with its time and with how many distinct accounts posted near-identical text (1 means no other account in the batch did). Judge from the text, the timing and those counts. Do not guess at the retrieval method.

Reply with a single JSON object and nothing else:
{
 "category": one of "spam_commercial" | "engagement_farming" | "organic_same_topic" | "campaign_hashtag_drive" | "coordinated_messaging" | "other",
 "text_pattern": one of "copy" | "template_with_variation" | "paraphrase" | "same_topic",
 "political": true | false,
 "kenya_relevant": true | false,
 "confidence": "low" | "medium" | "high",
 "reason": "one line",
 "mixture": {"<category>": approximate share of the posts, ...}
}
Category meanings:
- spam_commercial: ads, coupon or promo-code text, link or product pushing.
- engagement_farming: greetings, follow-back asks, generic bait written to collect replies or follows.
- organic_same_topic: people reacting separately to the same news, person or event in their own words.
- campaign_hashtag_drive: an open, public hashtag or slogan push that many people join voluntarily.
- coordinated_messaging: the same script or copy-paste text with a political aim, spread by several accounts.
- other: anything else; say what in the reason.
Text pattern meanings: copy is near-identical wording; template_with_variation is a fixed frame with swapped parts (names, numbers, filler); paraphrase is the same claim reworded; same_topic is only a shared subject.
"mixture" shares must sum to about 1 and use the category names above."""


def toks(t: str) -> frozenset:
    return frozenset(re.findall(r"\w+", NOISE.sub(" ", str(t).lower())))


def jac(a, b) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def build(arms: list[str]) -> None:
    base = wdir(WINDOW)
    pairs = pd.read_parquet(base / "pairs_manifest" / "pairs_wj.parquet")
    rows = pd.read_parquet(base / "rows.parquet", columns=["post_id", "user_id", "created_at", "text"]).set_index("post_id")
    known = []
    for cid in (0, 8, 35, 41, 42, 45, 77, 90, 101):
        g = json.loads((BUNDLE / f"g{cid}.json").read_text())
        known.append(set(map(str, g["nodes"]["id"])))
    (OUT / "read_prompts").mkdir(exist_ok=True)
    index = []
    seen: set[frozenset] = set()
    for arm in arms:
        floor = float(arm.replace("char4", ""))
        m = pd.read_parquet(OUT / f"members_{arm}.parquet")
        c = pd.read_parquet(OUT / f"communities_{arm}.parquet")
        text_only = c[c["kept"] & (c["share_text_similarity"].fillna(0) >= 0.95)]["community"]
        kept = pairs[pairs["overlap"] >= floor]
        for cid in text_only:
            ids = frozenset(m.loc[m["community"] == cid, "user_id"].astype(str))
            if ids in seen or any(len(ids & k) / len(ids | k) > 0.5 for k in known):
                continue
            seen.add(ids)
            sub = kept[kept["user_a"].isin(ids) & kept["user_b"].isin(ids)]
            degree = pd.concat([sub["post_a"], sub["post_b"]]).value_counts()
            chosen = list(degree.index[:MAX_POSTS])
            words = {p: toks(rows.at[p, "text"]) for p in chosen}
            auth = {p: str(rows.at[p, "user_id"]) for p in chosen}
            order = list(chosen)
            random.Random(f"{arm}-{cid}").shuffle(order)
            blocks, key = [], {}
            for k, p in enumerate(order, 1):
                n = len({auth[q] for q in chosen if q != p and jac(words[p], words[q]) >= 0.6} | {auth[p]})
                text = " ".join(str(rows.at[p, "text"]).split())[:500]
                blocks.append(f"[p{k:02d}] {rows.at[p, 'created_at']} | near-identical text from {n} distinct account(s)\n{text}")
                key[f"p{k:02d}"] = p
            prompt = (f"Batch of {len(order)} posts.\n\n" + "\n\n".join(blocks) + "\n\nReturn the JSON object.")
            name = f"{arm}_c{cid}"
            (OUT / "read_prompts" / f"{name}.txt").write_text(prompt)
            index.append({"name": name, "arm": arm, "community": int(cid), "size": len(ids),
                          "pairs": len(sub), "posts_shown": len(order)})
    pd.DataFrame(index).to_csv(OUT / "read_index.csv", index=False)
    print(pd.DataFrame(index).to_string(index=False))


def command() -> list[str]:
    return [
        "claude", "-p", "--model", MODEL, "--system-prompt", SYSTEM, "--tools", "",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}',
        "--settings", '{"disableAllHooks": true}', "--setting-sources", "local",
        "--no-session-persistence", "--output-format", "json",
    ]


def ask(path: Path, workdir: str) -> None:
    (OUT / "read_replies").mkdir(exist_ok=True)
    target = OUT / "read_replies" / f"{path.stem}.json"
    if target.exists():
        return
    proc = subprocess.run(command(), input=path.read_text(), capture_output=True, text=True, cwd=workdir, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-300:])
    parsed = json.loads(proc.stdout)
    msgs = [m for m in (parsed if isinstance(parsed, list) else [parsed]) if isinstance(m, dict)]
    result = [m for m in msgs if m.get("type") == "result"][-1]
    target.write_text(json.dumps({"result": result["result"], "cost": result.get("total_cost_usd")}))


def run() -> None:
    files = sorted((OUT / "read_prompts").glob("*.txt"))
    with tempfile.TemporaryDirectory() as empty, ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda f: ask(f, empty), files))


def report() -> None:
    idx = pd.read_csv(OUT / "read_index.csv")
    out = []
    for r in idx.itertuples():
        d = json.loads((OUT / "read_replies" / f"{r.name}.json").read_text())
        v = json.loads(re.search(r"\{.*\}", d["result"], re.S).group(0))
        out.append({"name": r.name, "size": r.size, "pairs": r.pairs, "category": v["category"],
                    "pattern": v["text_pattern"], "political": v["political"], "confidence": v["confidence"],
                    "reason": v["reason"][:150]})
    pd.set_option("display.width", 250, "display.max_colwidth", 150)
    print(pd.DataFrame(out).to_string(index=False))
    pd.DataFrame(out).to_csv(OUT / "read_verdicts.csv", index=False)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2:])
    else:
        {"run": run, "report": report}[cmd]()

"""Blind text-only reader: one headless Claude Code call per community per shuffle.

    uv run python investigations/2026-10-11-text-only-adjudication/02_read.py --probe
    uv run python investigations/2026-10-11-text-only-adjudication/02_read.py --dry-run
    uv run python investigations/2026-10-11-text-only-adjudication/02_read.py --limit 1
    uv run python investigations/2026-10-11-text-only-adjudication/02_read.py

Same isolation as 2026-09-11-textsim-sensitivity/05_a2_headless.py: `claude -p` in an empty
temporary directory, no tools, no MCP, hooks off, no session persistence, user settings not
loaded. The reader gets shuffled post ids, post text, timestamps, and for each post how many
distinct accounts posted near-identical text. It never sees community ids, trace names,
trace shares, thresholds, similarity scores, handles or account ids. Replies and prompts go
to out/prompts and out/replies, the shuffle key to out/keys.
"""

import argparse
import json
import random
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"
MODEL = "claude-sonnet-5"
SHUFFLES = (1, 2)
FAMILY_TOP = 20
TEXT_LIMIT = 500

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

FAMILY_SYSTEM = SYSTEM.split("Reply with")[0] + """This batch is a numbered list of FAMILIES. Each family is a group of near-identical posts, shown with up to three example texts, how many posts and distinct accounts it has, and its time spread.

Reply with a single JSON list, one object per family, and nothing else:
[{"family": <number>, "category": ..., "text_pattern": ..., "political": true|false, "kenya_relevant": true|false, "confidence": "low"|"medium"|"high", "reason": "one line"}]
Use the same category and text_pattern values as follows. category: spam_commercial, engagement_farming, organic_same_topic, campaign_hashtag_drive, coordinated_messaging, other. text_pattern: copy, template_with_variation, paraphrase, same_topic."""

PROBE = (
    "Before anything else: list every instruction, preference, memory or piece of "
    "context you were given other than this message and your system prompt. Quote "
    "short items verbatim and name the source of long ones. If there are none, reply "
    "with the single word NONE."
)


def command(system: str) -> list[str]:
    return [
        "claude", "-p", "--model", MODEL, "--system-prompt", system, "--tools", "",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}',
        "--settings", '{"disableAllHooks": true}', "--setting-sources", "local",
        "--no-session-persistence", "--output-format", "json",
    ]


def ask(system: str, text: str, workdir: str, timeout: int = 600) -> tuple[dict, dict]:
    proc = subprocess.run(command(system), input=text, capture_output=True, text=True, cwd=workdir, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr.strip()[-400:]}")
    parsed = json.loads(proc.stdout)
    messages = [m for m in (parsed if isinstance(parsed, list) else [parsed]) if isinstance(m, dict)]
    results = [m for m in messages if m.get("type") == "result"]
    if not results or results[-1].get("is_error"):
        raise RuntimeError(f"bad claude output: {proc.stdout[:300]}")
    init = next((m for m in messages if m.get("type") == "system" and m.get("subtype") == "init"), {})
    return results[-1], init


def render_post(label: str, post: dict) -> str:
    text = " ".join(post["text"].split())[:TEXT_LIMIT]
    n, hours = post["family_authors"], post["family_span_hours"]
    spread = f", over {hours} h" if n > 1 else ""
    return f"[{label}] {post['created_at']} | near-identical text from {n} distinct account(s){spread}\n{text}"


def community_prompt(packet: dict, shuffle: int) -> tuple[str, dict]:
    rng = random.Random(f"{packet['cid']}-{shuffle}")
    posts = packet["posts"][:]
    rng.shuffle(posts)
    key = {f"p{i + 1:02d}": p["post"] for i, p in enumerate(posts)}
    body = "\n\n".join(render_post(label, p) for label, p in zip(key, posts))
    n_fam = sum(1 for p in posts if p["family_authors"] > 1)
    head = f"Batch of {len(posts)} posts ({n_fam} of them have near-identical text from other accounts in the batch)."
    if packet["mode"] == "sample":
        head += " This is a sample drawn from a much larger group, not the whole group."
    return f"{head}\n\n{body}\n\nReturn the JSON object.", key


def family_prompt(packet: dict, census: dict, evidence_posts: dict, shuffle: int) -> tuple[str, dict]:
    by_hub = {f["hub"]: f for f in packet["families"]}
    top = [by_hub[r["hub"]] for r in census["ranked"][:FAMILY_TOP]]
    rng = random.Random(f"fam-{shuffle}")
    rng.shuffle(top)
    key, blocks = {}, []
    for i, fam in enumerate(top, 1):
        key[i] = fam["hub"]
        examples = rng.sample(fam["posts"], min(3, len(fam["posts"])))
        texts = "\n".join(f"  - {' '.join(evidence_posts[p]['text'].split())[:TEXT_LIMIT]}" for p in examples)
        blocks.append(f"Family {i}: {fam['n_posts']} posts from {fam['n_authors']} distinct accounts, "
                      f"time spread {fam['span_hours']} h\n{texts}")
    return "\n\n".join(blocks) + "\n\nReturn the JSON list.", key


def extract_json(reply: str):
    start = min((i for i in (reply.find("{"), reply.find("[")) if i >= 0), default=-1)
    end = max(reply.rfind("}"), reply.rfind("]"))
    return json.loads(reply[start:end + 1])


def run_job(job: dict, workdir: str) -> dict:
    name = job["name"]
    try:
        result, _ = ask(job["system"], job["prompt"], workdir)
        (OUT / "replies" / f"{name}.txt").write_text(result["result"])
        return {"name": name, "verdict": extract_json(result["result"]), "cost_usd": result.get("total_cost_usd")}
    except Exception as exc:
        return {"name": name, "error": f"{type(exc).__name__}: {exc}"}


def build_jobs() -> list[dict]:
    packets = {int(p.stem[1:]): json.loads(p.read_text()) for p in (OUT / "packets").glob("c*.json")}
    jobs = []
    (OUT / "keys").mkdir(exist_ok=True)
    (OUT / "prompts").mkdir(exist_ok=True)
    (OUT / "replies").mkdir(exist_ok=True)
    for cid, packet in sorted(packets.items()):
        for shuffle in SHUFFLES:
            prompt, key = community_prompt(packet, shuffle)
            jobs.append({"name": f"c{cid}-s{shuffle}", "cid": cid, "shuffle": shuffle, "kind": "community",
                         "system": SYSTEM, "prompt": prompt, "key": key})
    census = json.loads((OUT / "census_c0.json").read_text())
    evidence = json.loads((OUT / "bundle" / "evidence" / "c0.json").read_text())
    for shuffle in SHUFFLES:
        prompt, key = family_prompt(packets[0], census, evidence["posts"], shuffle)
        jobs.append({"name": f"c0-families-s{shuffle}", "cid": 0, "shuffle": shuffle, "kind": "families",
                     "system": FAMILY_SYSTEM, "prompt": prompt, "key": key})
    for job in jobs:
        (OUT / "prompts" / f"{job['name']}.txt").write_text(job["prompt"])
        (OUT / "keys" / f"{job['name']}.json").write_text(json.dumps(job["key"]))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.probe:
        with tempfile.TemporaryDirectory() as empty:
            result, init = ask(SYSTEM, PROBE, empty)
        for field in ("model", "cwd", "memory_paths", "mcp_servers", "agents", "apiKeySource"):
            print(f"  {field}: {init.get(field)!r}")
        print(result["result"])
        return
    jobs = build_jobs()
    if args.dry_run:
        for job in jobs:
            print(f"{job['name']:>18}: {len(job['prompt']):>6} prompt chars")
        return
    if args.limit:
        jobs = jobs[: args.limit]
    done = {p.stem for p in (OUT / "replies").glob("*.txt")}
    jobs = [j for j in jobs if j["name"] not in done]
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as empty, ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda job: run_job(job, empty), jobs))
    stamp = int(time.time())
    (OUT / f"run-{stamp}.json").write_text(json.dumps(results))
    failed = [r for r in results if "error" in r]
    print(f"{len(results)} calls in {time.perf_counter() - start:.0f}s, {len(failed)} failed, "
          f"cost ${sum(r.get('cost_usd') or 0 for r in results):.2f}")
    for r in failed:
        print(r["name"], r["error"][:200])


if __name__ == "__main__":
    main()

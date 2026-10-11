"""Tabulate reader verdicts: per-community table, self-agreement, community 0 family mass by category.

    uv run python investigations/2026-10-11-text-only-adjudication/03_report.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"
BARE = re.compile(r'("(?:category|text_pattern)":\s*)([a-z_]+)(\s*[,}])')
FIELDS = ("category", "text_pattern", "political", "kenya_relevant")


def load_reply(name: str):
    text = (OUT / "replies" / f"{name}.txt").read_text()
    text = BARE.sub(r'\1"\2"\3', text)
    start = min(i for i in (text.find("{"), text.find("[")) if i >= 0)
    return json.loads(text[start:max(text.rfind("}"), text.rfind("]")) + 1])


def main() -> None:
    census = json.loads((OUT / "census_c0.json").read_text())
    packets = {int(p.stem[1:]): json.loads(p.read_text()) for p in (OUT / "packets").glob("c[0-9]*.json")}
    print("| cid | posts | families(authors) | run | category | pattern | political | kenya | conf | reason |")
    agree = {}
    for cid in sorted(packets):
        runs = [load_reply(f"c{cid}-s{s}") for s in (1, 2)]
        fams = sorted((f["n_authors"] for f in packets[cid]["families"]), reverse=True)[:3]
        for s, v in enumerate(runs, 1):
            print(f"| {cid} | {len(packets[cid]['posts'])} | {fams} | s{s} | {v['category']} | {v['text_pattern']} | "
                  f"{v['political']} | {v['kenya_relevant']} | {v['confidence']} | {v['reason']} |")
        agree[cid] = {f: runs[0][f] == runs[1][f] for f in FIELDS}
        print(f"mixture c{cid}: {runs[0].get('mixture')} || {runs[1].get('mixture')}")
    print("\nself-agreement (field-level, same verdict in both shuffles)")
    for cid, a in agree.items():
        print(cid, {k: int(v) for k, v in a.items()})
    print("total", {f: sum(a[f] for a in agree.values()) for f in FIELDS}, "of", len(agree))

    print("\ncommunity 0 families (top 20 by edge mass)")
    runs = [load_reply(f"c0-families-s{s}") for s in (1, 2)] if (OUT / "replies/c0-families-s2.txt").exists() else []
    ranked = census["ranked"][:20]
    keys = [json.loads((OUT / "keys" / f"c0-families-s{s}.json").read_text()) for s in (1, 2)]
    mass_by_cat = [defaultdict(int), defaultdict(int)]
    per_hub = defaultdict(list)
    for s, (run, key) in enumerate(zip(runs, keys)):
        by_hub = {r["hub"]: r for r in ranked}
        for item in run:
            hub = key[str(item["family"])]
            per_hub[hub].append(item)
            mass_by_cat[s][item["category"]] += by_hub[hub]["edges"]
    same_cat = sum(1 for h, v in per_hub.items() if len(v) == 2 and v[0]["category"] == v[1]["category"])
    same_pat = sum(1 for h, v in per_hub.items() if len(v) == 2 and v[0]["text_pattern"] == v[1]["text_pattern"])
    print(f"families judged both runs: {sum(len(v) == 2 for v in per_hub.values())}, category agree {same_cat}, pattern agree {same_pat}")
    for s in range(2):
        print(f"run s{s + 1} edge mass by category (top-20 families):", dict(mass_by_cat[s]))
    print("pattern counts:", [Counter(i["text_pattern"] for i in r) for r in runs])
    top20 = sum(r["edges"] for r in ranked)
    print(f"top20 edges {top20}; copy-family edges {census['copy_edges']}; text edges {census['text_edges']}; "
          f"evidenced {census['text_edges_with_evidence']}; graph edges {census['graph_edges']}")
    print("jaccard fifths", census["jaccard_fifths_of_evidenced_edges"])
    for r, in zip(ranked):
        v = per_hub[r["hub"]]
        print(f"edges {r['edges']:>5} posts {r['n_posts']:>4} authors {r['n_authors']:>4} span {r['span_hours']:>7}h",
              [(i["category"], i["text_pattern"]) for i in v])


if __name__ == "__main__":
    main()

"""Build one blind packet per community from the bundle: sampled posts, near-identical families, timing.

    uv run python investigations/2026-10-11-text-only-adjudication/01_packets.py

Writes out/packets/c<cid>.json (unshuffled, includes author ids for counting only) and
out/census_c0.json (text-edge mass of community 0 by family). Nothing here is shown to the
reader directly; 02_read.py shuffles, strips ids and renders.
"""

import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from kma.families import hub_families

HERE = Path(__file__).parent
BUNDLE = HERE / "out" / "bundle"
PACKETS = HERE / "out" / "packets"
SMALL = [41, 42, 45, 77, 90, 101]
BIG_SAMPLE = 40
FAMILY_TOP = 20
SEED = 0
COPY_JACCARD = 0.6
NOISE = re.compile(r"https?://\S+|@\w+")


def load(kind: str, cid: int) -> dict:
    return json.loads((BUNDLE / kind / f"c{cid}.json").read_text())


def parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", NOISE.sub(" ", text.lower())))


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


def text_pairs(evidence: dict) -> list[tuple[str, str, float, str]]:
    pairs = {}
    for key, edge in evidence["edges"].items():
        for item in edge["items"]:
            if item["trace"] == "text_similarity":
                a, b = item["post_a"], item["post_b"]
                if a in evidence["posts"] and b in evidence["posts"]:
                    pairs[tuple(sorted((a, b)))] = (item["similarity"], key)
    return [(a, b, s, k) for (a, b), (s, k) in pairs.items()]


def families(evidence: dict, pairs: list) -> dict[str, dict]:
    """Hub-star families of the pairs whose word sets overlap at least COPY_JACCARD: literal near-copies only.

    Not connected components: a chain of near-copies would merge into one family. Every member is a
    direct near-copy of the family's hub (kma.families)."""
    pairs = [p for p in pairs if jaccard(evidence["posts"][p[0]]["text"], evidence["posts"][p[1]]["text"]) >= COPY_JACCARD]
    ids = sorted({p for a, b, _, _ in pairs for p in (a, b)})
    index = {p: i for i, p in enumerate(ids)}
    labels, _ = hub_families(
        np.array([index[a] for a, _, _, _ in pairs], dtype=int),
        np.array([index[b] for _, b, _, _ in pairs], dtype=int),
        len(ids),
    )
    root_of = {p: ids[labels[index[p]]] for p in ids}
    members: dict[str, set[str]] = defaultdict(set)
    for p in ids:
        members[root_of[p]].add(p)
    members = {root: posts for root, posts in members.items() if len(posts) > 1}
    pairs = [p for p in pairs if root_of[p[0]] == root_of[p[1]]]
    out = {}
    pair_count = Counter(root_of[a] for a, _, _, _ in pairs)
    for root, posts in members.items():
        records = [evidence["posts"][p] for p in posts]
        times = sorted(parse(r["created_at"]) for r in records)
        out[root] = {
            "posts": sorted(posts),
            "n_posts": len(posts),
            "n_authors": len({r["author_id"] for r in records}),
            "first": times[0].isoformat(),
            "last": times[-1].isoformat(),
            "span_hours": round((times[-1] - times[0]).total_seconds() / 3600, 1),
            "pairs": pair_count[root],
            "hub": root,
        }
    return out


def post_view(evidence: dict, pid: str, family: dict | None) -> dict:
    record = evidence["posts"][pid]
    view = {"post": pid, "text": record["text"], "created_at": record["created_at"]}
    view["family_posts"] = family["n_posts"] if family else 1
    view["family_authors"] = family["n_authors"] if family else 1
    view["family_span_hours"] = family["span_hours"] if family else 0.0
    return view


def small_packet(cid: int) -> dict:
    evidence = load("evidence", cid)
    pairs = text_pairs(evidence)
    fams = families(evidence, pairs)
    owner = {p: root for root, f in fams.items() for p in f["posts"]}
    posts = [post_view(evidence, p, fams.get(owner.get(p))) for p in evidence["posts"]]
    return {"cid": cid, "mode": "all", "posts": posts, "families": list(fams.values()),
            "n_pairs": len(pairs)}


def stratified_pairs(pairs: list, rng: random.Random, n_posts: int) -> list[str]:
    ordered = sorted(pairs, key=lambda p: (p[2], p[0], p[1]))
    strata = [ordered[i * len(ordered) // 4:(i + 1) * len(ordered) // 4] for i in range(4)]
    per = n_posts // 8
    chosen: list[str] = []
    for stratum in strata:
        picked = 0
        for a, b, _, _ in rng.sample(stratum, len(stratum)):
            if a in chosen or b in chosen:
                continue
            chosen += [a, b]
            picked += 1
            if picked == per:
                break
    return chosen


def big_packet(cid: int) -> dict:
    rng = random.Random(SEED + cid)
    evidence = load("evidence", cid)
    pairs = text_pairs(evidence)
    fams = families(evidence, pairs)
    owner = {p: root for root, f in fams.items() for p in f["posts"]}
    chosen = stratified_pairs(pairs, rng, BIG_SAMPLE)
    if cid == 8:
        pool = [p for p in evidence["posts"] if p not in chosen]
        chosen += rng.sample(pool, min(BIG_SAMPLE - len(chosen), len(pool)))
    posts = [post_view(evidence, p, fams.get(owner.get(p))) for p in chosen]
    return {"cid": cid, "mode": "sample", "posts": posts, "n_pairs": len(pairs),
            "families": list(fams.values()), "n_family_total": len(fams)}


def census(cid: int, packet: dict) -> dict:
    """Attribute each full-graph text edge with evidence to its family, rank families by edge mass."""
    evidence, graph = load("evidence", cid), load("graph", cid)
    fams = {f["hub"]: f for f in packet["families"]}
    owner = {p: f["hub"] for f in packet["families"] for p in f["posts"]}
    text_edges = [i for i, m in enumerate(graph["edges"]["mask"]) if m & 16]
    mass: Counter = Counter()
    weight: Counter = Counter()
    attributed = 0
    buckets: Counter = Counter()
    for i in text_edges:
        edge = evidence["edges"].get(f"e{i}")
        if not edge:
            continue
        item = next((x for x in edge["items"] if x["trace"] == "text_similarity"), None)
        if not item:
            continue
        attributed += 1
        jac = jaccard(evidence["posts"][item["post_a"]]["text"], evidence["posts"][item["post_b"]]["text"])
        buckets[min(int(jac * 5), 4)] += 1
        if item["post_a"] in owner and jac >= COPY_JACCARD:
            hub = owner[item["post_a"]]
            mass[hub] += 1
            weight[hub] += graph["edges"]["w"][i]
    ranked = []
    for hub, n in mass.most_common():
        f = fams[hub]
        ranked.append({"hub": hub, "edges": n, "weight": round(weight[hub], 2), **{
            k: f[k] for k in ("n_posts", "n_authors", "span_hours")}})
    sizes = Counter(min(f["n_posts"], 50) for f in packet["families"])
    return {"graph_edges": len(graph["edges"]["s"]), "text_edges": len(text_edges),
            "text_edges_with_evidence": attributed,
            "jaccard_fifths_of_evidenced_edges": sorted(buckets.items()),
            "copy_edges": sum(mass.values()), "families": len(fams), "family_size_hist_capped_50": sorted(sizes.items()),
            "ranked": ranked}


if __name__ == "__main__":
    PACKETS.mkdir(parents=True, exist_ok=True)
    for cid in SMALL:
        packet = small_packet(cid)
        (PACKETS / f"c{cid}.json").write_text(json.dumps(packet))
        print(cid, len(packet["posts"]), "posts", len(packet["families"]), "families")
    for cid in (0, 8):
        packet = big_packet(cid)
        (PACKETS / f"c{cid}.json").write_text(json.dumps(packet))
        print(cid, len(packet["posts"]), "sampled of", packet["n_pairs"], "pairs;", len(packet["families"]), "families")
        if cid == 0:
            info = census(cid, packet)
            (HERE / "out" / "census_c0.json").write_text(json.dumps(info))
            print({k: v for k, v in info.items() if k != "ranked"})
            print([(r["edges"], r["n_posts"], r["n_authors"]) for r in info["ranked"][:10]])
    evidence = load("evidence", 35)
    posts = list(evidence["posts"])
    rng = random.Random(SEED + 35)
    chosen = rng.sample(posts, min(BIG_SAMPLE, len(posts)))
    packet = {"cid": 35, "mode": "sample", "n_pairs": 0, "families": [],
              "posts": [post_view(evidence, p, None) for p in chosen]}
    (PACKETS / "c35.json").write_text(json.dumps(packet))
    print(35, len(chosen), "posts")

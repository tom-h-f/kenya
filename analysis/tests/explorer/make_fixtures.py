#!/usr/bin/env python3
"""Write the synthetic explorer fixture bundle.

Usage: uv run python tests/explorer/make_fixtures.py [OUT_DIR]

Default OUT_DIR is docs/plans/explorer-contract/fixtures. The layout mirrors the
bucket: OUT_DIR/explorer/v1/manifest.json and OUT_DIR/explorer/v1/runs/{run}/...
Everything is synthetic and seeded: ids start at 9000000000000000001, handles are
user_001. Nothing is read from live data. Output is byte-stable for a given seed.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / "docs" / "plans" / "explorer-contract" / "fixtures"

SEED = 20261010
RUNS = [
    {"run_id": "20261010T031500Z", "window_end": "2026-10-09", "computed_at": "2026-10-10T03:41:07Z"},
    {"run_id": "20261009T031500Z", "window_end": "2026-10-08", "computed_at": "2026-10-09T03:39:52Z"},
]
WINDOW_DAYS = 21
SIZES = {0: 60, 1: 30, 2: 12}
DROPPED = {3: 8}
FIRST_ID = 9000000000000000001
FIRST_POST = 9100000000000000001
FIRST_ORIGIN = 9000000000000002001
N_POSTS = 2000

TAG_POOL = [
    "budgetwatch", "countyfunds", "roadworks", "schoolfees", "healthaccess", "farmgate",
    "waterdrive", "jobsnow", "transitfares", "localnews", "townhall", "harvesttime",
    "powerbills", "ballotlogistics", "youthvoice", "marketprices", "clinicqueue", "bridgeopen",
    "libraryweek", "cleanupday", "festivalweek", "sportsfinal", "weathernote", "rentcosts",
]
TOPIC_TERMS = [
    ["budget", "county", "funds", "allocation"],
    ["road", "bridge", "works", "tender"],
    ["school", "fees", "bursary", "term"],
    ["clinic", "queue", "drugs", "supply"],
    ["market", "prices", "maize", "flour"],
    ["power", "bills", "outage", "tariff"],
]
WORDS = ["update", "report", "notice", "thread", "summary", "note", "question", "reply", "view", "photo"]
DOMAINS = ["news.example", "blog.example", "gazette.example", "video.example", "docs.example", "paper.example"]
TRACES = {1: "co_retweet", 2: "co_url", 4: "hashtag_sequence", 8: "fast_retweet", 16: "text_similarity"}

PARAMS = {
    "text_threshold": 0.9, "text_overlap": 0.5, "min_entities": 3, "min_kenya": 0.6,
    "relevance_model": "fixture-relevance-v0", "leiden_resolution": 1.0, "leiden_seed": 0,
    "min_community_size": 8, "listing_budget": 500, "listing_per_group": 20,
    "edge_percentile": {"co_retweet": 80, "co_url": 80, "hashtag_sequence": 65, "fast_retweet": None, "text_similarity": 80},
}
EDGE_RULE = "Edges drawn where two accounts retweeted at least 2 of the same posts, or are linked by any other trace. Single-shared-retweet links are opt-in."


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dumps(obj) -> bytes:
    return (json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def account_shard(aid: str) -> int:
    return int(hashlib.sha1(aid.encode()).hexdigest()[:8], 16) % 64


def tag_shard(tag: str) -> int:
    return int(hashlib.sha1(tag.lower().encode()).hexdigest()[0], 16)


def lineage(n: int) -> str:
    return hashlib.sha1(f"fixture-lineage-{n}".encode()).hexdigest()[:10]


class World:
    """All synthetic entities, built once and shared by both runs."""

    def __init__(self):
        self.rng = random.Random(SEED)
        r = self.rng
        self.members: dict[int, list[str]] = {}
        self.handle: dict[str, str] = {}
        self.display: dict[str, str] = {}
        self.cid_of: dict[str, int] = {}
        k = 0
        for cid, size in SIZES.items():
            ids = []
            for _ in range(size):
                aid = str(FIRST_ID + k)
                k += 1
                ids.append(aid)
                self.handle[aid] = f"user_{k:03d}"
                self.display[aid] = f"Synthetic User {k:03d}" if k % 7 else None
                self.cid_of[aid] = cid
            self.members[cid] = ids
        self.origins = [str(FIRST_ORIGIN + i) for i in range(8)]
        for i, o in enumerate(self.origins):
            self.handle[o] = f"origin_{i + 1:02d}"
        self.all_members = [a for c in SIZES for a in self.members[c]]
        self.score = {a: round(r.uniform(0.05, 1.0), 4) for a in self.all_members}
        self.centrality = {a: round(r.uniform(0.0, 1.0), 4) for a in self.all_members}
        self.kenya = {a: (round(r.uniform(0.65, 0.98), 3) if r.random() > 0.06 else None) for a in self.all_members}
        self.build_posts()
        self.build_edges()
        self.build_layout()

    def window(self, end: str):
        end_d = date.fromisoformat(end)
        return [end_d - timedelta(days=WINDOW_DAYS - 1 - i) for i in range(WINDOW_DAYS)]

    def build_posts(self):
        r = self.rng
        days = self.window(RUNS[0]["window_end"])
        community_tags = {c: r.sample(TAG_POOL, 8) for c in SIZES}
        self.posts = []
        for i in range(N_POSTS):
            aid = r.choice(self.all_members)
            cid = self.cid_of[aid]
            day = r.choice(days)
            ts = datetime(day.year, day.month, day.day, r.randrange(24), r.randrange(60), r.randrange(60), tzinfo=timezone.utc)
            kind = r.choices(["original", "reply", "repost"], [5, 3, 4])[0]
            tags = []
            if kind != "repost" and r.random() < 0.7:
                tags = sorted(set(r.choices(community_tags[cid], k=r.randint(1, 3))))
            reposted = r.choice(self.origins) if kind == "repost" and r.random() < 0.85 else None
            pid = str(FIRST_POST + i)
            self.posts.append({
                "id": pid, "author_id": aid, "created_at": ts, "kind": kind,
                "text": f"Synthetic {r.choice(WORDS)} {i:04d} on {tags[0] if tags else 'general matters'}",
                "like_count": r.randrange(0, 400), "repost_count": r.randrange(0, 120), "reply_count": r.randrange(0, 40),
                "hashtags": tags, "reposted_author_id": reposted,
            })
        self.posts.sort(key=lambda p: p["created_at"])
        self.post_by_id = {p["id"]: p for p in self.posts}
        self.origin_posts = {}
        for i in range(30):
            pid = str(9200000000000000001 + i)
            o = self.origins[i % len(self.origins)]
            day = days[i % WINDOW_DAYS]
            self.origin_posts[pid] = {
                "id": pid, "author_id": o, "author_handle": self.handle[o],
                "created_at": iso(datetime(day.year, day.month, day.day, 8 + i % 10, 5, 0, tzinfo=timezone.utc)),
                "text": f"Synthetic source post {i:02d} for the fixture corpus",
                "url": f"https://x.example/{self.handle[o]}/status/{pid}",
                "like_count": r.randrange(100, 900), "repost_count": r.randrange(50, 300), "reply_count": r.randrange(0, 60),
                "counts_as_of": "2026-10-09T22:10:00Z",
            }

    def build_edges(self):
        r = self.rng
        self.edges: dict[tuple[str, str], dict] = {}

        def add(a, b, mask, shared, w):
            key = (a, b) if a < b else (b, a)
            e = self.edges.setdefault(key, {"mask": 0, "shared": 0, "w": 0.0})
            e["mask"] |= mask
            e["shared"] = max(e["shared"], shared)
            e["w"] = max(e["w"], w)

        trace_cycle = [1, 2, 4, 8, 16]
        for cid, ids in self.members.items():
            n = len(ids)
            for i, a in enumerate(ids):
                b = ids[(i + 1) % n]
                add(a, b, 1, r.choice([1, 2, 2, 3, 4, 6]), round(r.uniform(0.6, 0.99), 3))
            for _ in range(n * 2):
                a, b = r.sample(ids, 2)
                bit = r.choice(trace_cycle)
                mask = bit | (1 if r.random() < 0.3 else 0)
                shared = r.choice([1, 1, 2, 3, 5]) if mask & 1 else 0
                add(a, b, mask, shared, round(r.uniform(0.5, 0.99), 3))
        for c1, c2 in [(0, 1), (1, 2), (0, 2)]:
            for _ in range(6):
                a, b = r.choice(self.members[c1]), r.choice(self.members[c2])
                add(a, b, 1, r.choice([1, 2, 3]), round(r.uniform(0.5, 0.9), 3))
        # every trace type appears in the first community's main graph
        ids = self.members[0]
        for j, bit in enumerate([2, 4, 8, 16]):
            add(ids[2 * j], ids[2 * j + 5], bit, 0, 0.8)
        add(ids[20], ids[30], 1 | 16, 4, 0.91)

    def build_layout(self):
        r = self.rng
        centres = {0: (-6.0, 1.0), 1: (5.0, 4.0), 2: (1.0, -6.0)}
        self.xy = {}
        for cid, ids in self.members.items():
            cx, cy = centres[cid]
            for a in ids:
                self.xy[a] = (round(cx + r.gauss(0, 1.6), 3), round(cy + r.gauss(0, 1.6), 3))
        self.meta_xy = {cid: (round(cx, 3), round(cy, 3)) for cid, (cx, cy) in centres.items()}

    def is_thin(self, e):
        return e["mask"] == 1 and e["shared"] == 1

    def degree_all(self):
        d = Counter()
        for (a, b) in self.edges:
            d[a] += 1
            d[b] += 1
        return d


def build_graph(w: World, stamp, ids: list[str], scope: str, cid, edge_keys, layout: str):
    idx = {a: i for i, a in enumerate(ids)}
    deg = Counter()
    s, t, ww, mask, shared = [], [], [], [], []
    for (a, b) in edge_keys:
        e = w.edges[(a, b)]
        s.append(idx[a]); t.append(idx[b]); ww.append(e["w"]); mask.append(e["mask"]); shared.append(e["shared"])
        deg[a] += 1; deg[b] += 1
    g = {
        "kind": "graph", "stamp": stamp, "scope": scope,
        "edge_rule": EDGE_RULE if scope != "community_thin" else "Edges resting on exactly one shared retweet and no other trace.",
        "layout": layout,
        "nodes": {
            "id": ids, "handle": [w.handle[a] for a in ids], "cid": [w.cid_of[a] for a in ids],
            "x": [w.xy[a][0] for a in ids], "y": [w.xy[a][1] for a in ids], "degree": [deg[a] for a in ids],
        },
        "edges": {"s": s, "t": t, "w": ww, "mask": mask, "shared": shared},
    }
    if cid is not None:
        g["cid"] = cid
    return g


def evidence_for(w: World, cid: int, main_keys, thin_keys):
    r = random.Random(SEED + cid)
    posts: dict[str, dict] = {}
    member_posts = defaultdict(list)
    for p in w.posts:
        if w.cid_of[p["author_id"]] == cid and p["kind"] != "repost":
            member_posts[p["author_id"]].append(p)

    def ref(p):
        a = p["author_id"]
        return {
            "id": p["id"], "author_id": a, "author_handle": w.handle[a], "created_at": iso(p["created_at"]),
            "text": p["text"], "url": f"https://x.example/{w.handle[a]}/status/{p['id']}",
            "like_count": p["like_count"], "repost_count": p["repost_count"], "reply_count": p["reply_count"],
            "counts_as_of": "2026-10-09T22:10:00Z",
        }

    def items_for(a, b, e):
        out = []
        for bit, name in TRACES.items():
            if not e["mask"] & bit:
                continue
            if name == "co_retweet":
                total = e["shared"]
                chosen = r.sample(sorted(w.origin_posts), min(total, 5))
                for pid in chosen:
                    posts[pid] = w.origin_posts[pid]
                out.append((name, total, [{"trace": name, "post": pid} for pid in chosen]))
            elif name == "co_url":
                n = r.randint(1, 3)
                out.append((name, n, [{"trace": name, "url": f"https://{r.choice(DOMAINS)}/a/{r.randrange(1000, 9999)}"} for _ in range(n)]))
            elif name == "hashtag_sequence":
                tags = r.sample(TAG_POOL, 3)
                out.append((name, 1, [{"trace": name, "tags": tags}]))
            elif name == "fast_retweet":
                pid = r.choice(sorted(w.origin_posts))
                posts[pid] = w.origin_posts[pid]
                out.append((name, 1, [{"trace": name, "post": pid, "gap_seconds": r.randrange(2, 55)}]))
            else:
                pa_, pb_ = r.choice(member_posts[a] or w.posts), r.choice(member_posts[b] or w.posts)
                posts[pa_["id"]] = ref(pa_); posts[pb_["id"]] = ref(pb_)
                out.append((name, 1, [{"trace": name, "post_a": pa_["id"], "post_b": pb_["id"], "similarity": round(r.uniform(0.9, 0.99), 3)}]))
        return out

    edges = {}
    for prefix, keys in (("e", main_keys), ("t", thin_keys)):
        for i, (a, b) in enumerate(keys):
            groups = items_for(a, b, w.edges[(a, b)])
            items = [it for _, _, its in groups for it in its][:5]
            total = sum(tot for _, tot, _ in groups)
            edges[f"{prefix}{i}"] = {"total": max(total, len(items)), "items": items}
    for item in edges.values():
        left_out = next((it["post"] for it in item["items"] if it["trace"] == "co_retweet" and it["post"] in posts), None)
        if left_out:
            del posts[left_out]
            break
    return edges, dict(sorted(posts.items()))


def day_label(d: date) -> str:
    return d.isoformat()


def build_run(w: World, root: Path, run: dict, is_latest: bool, prev_run_id) -> dict[str, bytes]:
    r = random.Random(SEED + int(run["run_id"][6:8]))
    run_id = run["run_id"]
    end = date.fromisoformat(run["window_end"])
    stamp = {
        "schema_version": 1, "run_id": run_id, "window_id": f"daily-{run['window_end']}-{WINDOW_DAYS}d",
        "window_start": (end - timedelta(days=WINDOW_DAYS - 1)).isoformat(), "window_end": run["window_end"],
        "window_days": WINDOW_DAYS, "computed_at": run["computed_at"],
        "generated_at": iso(datetime.fromisoformat(run["computed_at"].replace("Z", "+00:00")) + timedelta(minutes=11)),
        "params": PARAMS,
    }
    files: dict[str, bytes] = {}

    def put(rel, obj):
        files[rel] = dumps(obj)

    days = w.window(RUNS[0]["window_end"])
    posts_by_cid = defaultdict(list)
    for p in w.posts:
        posts_by_cid[w.cid_of[p["author_id"]]].append(p)

    tag_posts = defaultdict(set); tag_accts = defaultdict(set)
    cid_tags = {c: Counter() for c in SIZES}; cid_tag_acc = {c: defaultdict(set) for c in SIZES}
    for p in w.posts:
        c = w.cid_of[p["author_id"]]
        for t in p["hashtags"]:
            tag_posts[t].add(p["id"]); tag_accts[t].add(p["author_id"])
            cid_tags[c][t] += 1; cid_tag_acc[c][t].add(p["author_id"])

    main_by_c: dict[int, list] = {}
    thin_by_c: dict[int, list] = {}
    cross_main = []
    for key, e in sorted(w.edges.items()):
        ca, cb = w.cid_of[key[0]], w.cid_of[key[1]]
        if ca == cb:
            (thin_by_c if w.is_thin(e) else main_by_c).setdefault(ca, []).append(key)
        elif not w.is_thin(e):
            cross_main.append(key)
    for c in SIZES:
        main_by_c.setdefault(c, []); thin_by_c.setdefault(c, [])

    rows = []
    for cid in list(SIZES) + list(DROPPED):
        if cid in SIZES:
            ids = w.members[cid]
            edges_n = len(main_by_c[cid]) + len(thin_by_c[cid])
            tags = cid_tags[cid].most_common(10)
            row = {
                "cid": cid, "lineage": lineage(cid), "label": f"Community {cid + 1:02d}", "size": len(ids), "edges": edges_n,
                "eigenvalue": round(1.5 + 0.8 * (3 - cid), 3),
                "trace_share": {"co_retweet": 0.6, "co_url": 0.1, "hashtag_sequence": 0.1, "fast_retweet": 0.1, "text_similarity": 0.1},
                "kenya_model": round(sum(v for v in (w.kenya[a] for a in ids) if v is not None) / max(1, sum(1 for a in ids if w.kenya[a] is not None)), 3),
                "relevance_coverage": round(0.82 - 0.05 * cid, 3), "relevance_scored": True, "kept": True, "listed": min(len(ids), 20),
                "template": TEMPLATES[cid % len(TEMPLATES)],
                "top_hashtags": [{"tag": t, "posts": n, "accounts": len(cid_tag_acc[cid][t])} for t, n in tags],
                "top_domains": [{"domain": d, "posts": 40 - 7 * i - 3 * cid, "accounts": 12 - 2 * i} for i, d in enumerate(DOMAINS[:5])],
            }
        else:
            row = {
                "cid": cid, "lineage": lineage(cid), "label": f"Community {cid + 1:02d}", "size": DROPPED[cid], "edges": 21,
                "eigenvalue": 1.1, "trace_share": {"co_retweet": 1.0}, "kenya_model": 0.12, "relevance_coverage": 0.64,
                "relevance_scored": True, "kept": False,
            }
        rows.append(row)
    kept_members = [a for c in SIZES for a in w.members[c]]
    put("communities.json", {"kind": "community_index", "stamp": stamp, "communities": rows})

    def row_of(cid):
        return next(x for x in rows if x["cid"] == cid)

    reposted = defaultdict(Counter)
    for p in w.posts:
        if p["kind"] == "repost" and p["reposted_author_id"]:
            reposted[p["author_id"]][p["reposted_author_id"]] += 1

    for cid in SIZES:
        ids = sorted(w.members[cid], key=lambda a: -w.score[a])
        count = Counter(p["author_id"] for p in posts_by_cid[cid])
        daily = []
        for d in days:
            k = Counter(p["kind"] for p in posts_by_cid[cid] if p["created_at"].date() == d)
            daily.append({"dt": day_label(d), "originals": k["original"], "replies": k["reply"], "reposts": k["repost"]})
        amp = Counter(); amp_members = defaultdict(set)
        for a in ids:
            for o, n in reposted[a].items():
                amp[o] += n; amp_members[o].add(a)
        total_reposts = sum(1 for p in posts_by_cid[cid] if p["kind"] == "repost")
        put(f"communities/{cid}.json", {
            "kind": "community_detail", "stamp": stamp, "community": row_of(cid),
            "members": {
                "id": ids, "handle": [w.handle[a] for a in ids], "within_score": [w.score[a] for a in ids],
                "centrality": [w.centrality[a] for a in ids], "posts_in_window": [count[a] for a in ids],
                "kenya_model": [w.kenya[a] for a in ids],
            },
            "daily_posts": daily,
            "amplified": [{"id": o, "handle": w.handle[o], "reposts": n, "members": len(amp_members[o])} for o, n in amp.most_common(20)],
            "amplified_coverage": round(sum(amp.values()) / total_reposts, 3) if total_reposts else None,
            "graph_available": True,
        })
        main_keys, thin_keys = main_by_c[cid], thin_by_c[cid]
        put(f"graph/c{cid}.json", build_graph(w, stamp, w.members[cid], "community", cid, main_keys, "synthetic seed 0"))
        put(f"graph/c{cid}.thin.json", build_graph(w, stamp, w.members[cid], "community_thin", cid, thin_keys, "synthetic seed 0"))
        ev_edges, ev_posts = evidence_for(w, cid, main_keys, thin_keys)
        put(f"evidence/c{cid}.json", {"kind": "evidence", "stamp": stamp, "cid": cid, "edges": ev_edges, "posts": ev_posts})

    all_main = [k for c in SIZES for k in main_by_c[c]] + cross_main
    put("graph/all.json", build_graph(w, stamp, kept_members, "all", None, all_main, "synthetic seed 0"))

    meta_edges = defaultdict(lambda: [0, 0])
    for key, e in w.edges.items():
        ca, cb = sorted((w.cid_of[key[0]], w.cid_of[key[1]]))
        if ca != cb:
            meta_edges[(ca, cb)][0] += 1
            meta_edges[(ca, cb)][1] += 1 if (e["shared"] >= 2 or e["mask"] != 1) else 0
    put("graph/meta.json", {
        "kind": "graph_meta", "stamp": stamp,
        "nodes": [{"cid": c, "size": SIZES[c], "x": w.meta_xy[c][0], "y": w.meta_xy[c][1]} for c in SIZES],
        "edges": [{"a": a, "b": b, "edges": n, "edges_shared_ge_2": m} for (a, b), (n, m) in sorted(meta_edges.items())],
    })

    deg_all = w.degree_all()
    nbrs = defaultdict(list)
    for (a, b), e in w.edges.items():
        if w.is_thin(e):
            continue
        nbrs[a].append((b, e)); nbrs[b].append((a, e))
    shards: dict[int, dict] = {i: {} for i in range(64)}
    posts_of = defaultdict(list)
    for p in w.posts:
        posts_of[p["author_id"]].append(p)
    for a in kept_members:
        mine = posts_of[a]
        day_n = Counter(p["created_at"].date() for p in mine)
        tag_n = Counter(t for p in mine for t in p["hashtags"])
        recent = [p for p in reversed(mine) if p["kind"] != "repost"][:10]
        top = sorted(nbrs[a], key=lambda x: (-x[1]["shared"], -x[1]["w"], x[0]))[:25]
        k = int(w.handle[a][5:])
        shards[account_shard(a)][a] = {
            "id": a, "handle": w.handle[a], "cid": w.cid_of[a], "lineage": lineage(w.cid_of[a]),
            "within_score": w.score[a], "centrality": w.centrality[a], "kenya_model": w.kenya[a],
            "posts_scored": int(len(mine) * 0.8), "posts_in_window": len(mine),
            "reposts_in_window": sum(1 for p in mine if p["kind"] == "repost"),
            "profile": {
                "display_name": w.display[a], "bio": f"Synthetic bio for fixture account {k:03d}." if k % 5 else None,
                "followers": 100 + k * 37, "following": 50 + k * 11, "statuses": 1000 + k * 91,
                "created_at": "2019-03-14T09:00:00Z" if k % 3 else None, "profile_as_of": "2026-10-08T21:30:00Z",
            },
            "neighbours": [{"id": b, "handle": w.handle[b], "cid": w.cid_of[b], "mask": e["mask"], "shared": e["shared"], "w": e["w"]} for b, e in top],
            "neighbours_total": deg_all[a],
            "hashtags": [{"tag": t, "posts": n, "accounts": 1} for t, n in tag_n.most_common(10)],
            "amplifies": [{"id": o, "handle": w.handle[o], "reposts": n} for o, n in reposted[a].most_common(10)],
            "daily_posts": [{"dt": day_label(d), "n": day_n[d]} for d in days],
            "recent_posts": [{
                "id": p["id"], "author_id": a, "author_handle": w.handle[a], "created_at": iso(p["created_at"]), "text": p["text"],
                "url": f"https://x.example/{w.handle[a]}/status/{p['id']}", "like_count": p["like_count"],
                "repost_count": p["repost_count"], "reply_count": p["reply_count"], "counts_as_of": "2026-10-09T22:10:00Z",
            } for p in recent],
        }
    for i in range(64):
        put(f"accounts/{i}.json", {"kind": "account_shard", "stamp": stamp, "shard": i, "accounts": shards[i]})

    srt = sorted(kept_members, key=lambda a: w.handle[a].lower())
    put("search/accounts.json", {
        "kind": "search_accounts", "stamp": stamp, "id": srt, "handle": [w.handle[a] for a in srt],
        "display_name": [w.display[a] for a in srt], "cid": [w.cid_of[a] for a in srt],
    })

    index_tags = sorted(tag_posts, key=lambda t: (-len(tag_posts[t]), t))
    emerging = [{"tag": t, "recent_rate": round(0.004 + 0.001 * i, 4), "prior_rate": round(0.001 + 0.0002 * i, 4),
                 "emergence": None if i == 0 else round(3.0 + 0.5 * i, 2), "posts": len(tag_posts[t]), "censused": bool(i % 2)} for i, t in enumerate(index_tags[-3:])]
    put("hashtags/index.json", {
        "kind": "hashtag_index", "stamp": stamp,
        "basis": "Rows in the collected corpus for members of kept communities in the window. Counts, not prevalence.",
        "tags": [{"tag": t, "posts": len(tag_posts[t]), "accounts": len(tag_accts[t]), "shard": tag_shard(t),
                  "first_seen": stamp["window_start"], **({"emerging": True} if t in {e["tag"] for e in emerging} else {})} for t in index_tags],
        "emerging": emerging, "emerging_run": run_id,
    })
    by_shard = {i: {} for i in range(16)}
    for t in index_tags:
        tp = [w.post_by_id[i] for i in tag_posts[t]]
        daily = []
        for d in days:
            dp = [p for p in tp if p["created_at"].date() == d]
            daily.append({"dt": day_label(d), "posts": len(dp), "accounts": len({p["author_id"] for p in dp})})
        comm = Counter(w.cid_of[p["author_id"]] for p in tp)
        examples = [p for p in sorted(tp, key=lambda p: p["id"]) if p["kind"] != "repost"][:5]
        by_shard[tag_shard(t)][t] = {
            "tag": t, "daily": daily, "communities": [{"cid": c, "posts": n} for c, n in sorted(comm.items())],
            "examples": [{
                "id": p["id"], "author_id": p["author_id"], "author_handle": w.handle[p["author_id"]], "created_at": iso(p["created_at"]),
                "text": p["text"], "url": f"https://x.example/{w.handle[p['author_id']]}/status/{p['id']}",
                "like_count": p["like_count"], "repost_count": p["repost_count"], "reply_count": p["reply_count"], "counts_as_of": "2026-10-09T22:10:00Z",
            } for p in examples],
        }
    for n in range(16):
        put(f"hashtags/s{n}.json", {"kind": "hashtag_shard", "stamp": stamp, "shard": n, "tags": by_shard[n]})

    put("links/index.json", {
        "kind": "link_index", "stamp": stamp,
        "domains": [{"domain": d, "posts": 120 - 15 * i, "accounts": 40 - 5 * i} for i, d in enumerate(DOMAINS)],
        "urls": [{"url": f"https://{DOMAINS[i % len(DOMAINS)]}/a/{1000 + i}", "posts": 30 - i, "accounts": 12 - i // 3, "top_cids": [i % 3]} for i in range(12)],
    })

    topics = []
    for tid, terms in enumerate(TOPIC_TERMS):
        daily = [r.randrange(20, 80) for _ in days]
        knn = sum(daily) // 3
        topics.append({"tid": tid, "name": " ".join(terms[:3]), "terms": terms, "posts_in_window": sum(daily),
                       "fit_posts_in_window": sum(daily) - knn, "knn_posts_in_window": knn,
                       "accounts_in_window": 20 + 7 * tid, "daily": daily})
    put("topics/index.json", {
        "kind": "topic_index", "stamp": stamp,
        "fit": {"fitted_at": "2026-09-04T00:00:00Z", "fit_posts": 15000, "topics": len(topics), "noise_share": 0.31,
                "params": "synthetic fixture parameters", "assignment": "nearest fitted posts (kNN), k=10, share cut 0.40",
                "knn_posts": 41000, "window_posts": 60000, "window_noise_posts": 17000, "listed_topics": len(topics)},
        "topics": topics,
    })
    sample_posts = [p for p in w.posts if p["kind"] != "repost"][:10]
    for t in topics:
        detail = {
            "kind": "topic_detail", "stamp": stamp, "tid": t["tid"], "name": t["name"], "terms": t["terms"],
            "posts_in_window": t["posts_in_window"], "fit_posts_in_window": t["fit_posts_in_window"],
            "knn_posts_in_window": t["knn_posts_in_window"], "daily": t["daily"],
            "examples": [{
                "id": p["id"], "author_id": p["author_id"], "author_handle": w.handle[p["author_id"]], "created_at": iso(p["created_at"]),
                "text": p["text"], "url": f"https://x.example/{w.handle[p['author_id']]}/status/{p['id']}",
                "like_count": p["like_count"], "repost_count": p["repost_count"], "reply_count": p["reply_count"], "counts_as_of": "2026-10-09T22:10:00Z",
            } for p in sample_posts[: 3 + t["tid"]]],
            "communities": [{"cid": c, "posts": 40 + 10 * t["tid"] + c, "community_posts": len(posts_by_cid[c])} for c in SIZES],
            "hashtags": [{"tag": g, "posts": 20 - i, "accounts": 8 - i} for i, g in enumerate(TAG_POOL[t["tid"]: t["tid"] + 3])],
        }
        if t["tid"] == 0:
            detail["flags"] = {"scope": "baseline", "scored_posts": 400, "flagged_posts": 12, "model": "fixture-classifier-v0",
                               "threshold_note": "synthetic numbers, aggregate within topic only"}
        put(f"topics/{t['tid']}.json", detail)

    series = []
    gap = days[6]
    for d in days:
        if d == gap:
            continue
        search = r.randrange(5000, 6500); replies = r.randrange(1500, 2500); reposts = r.randrange(900, 1600)
        series.append({"dt": day_label(d), "rows": search + replies + reposts, "by_type": {"search": search, "replies": replies, "reposts": reposts}})
    put("time/collection.json", {
        "kind": "time_collection", "stamp": stamp, "series": series, "gap_days": [day_label(gap)],
        "control_arm": {"runs": 14, "posts": 29213, "truncated_runs": 1, "first": "2026-09-14", "last": run["window_end"]},
    })
    history = []
    for rr in RUNS:
        history.append({
            "run_id": rr["run_id"], "window_id": f"daily-{rr['window_end']}-{WINDOW_DAYS}d", "computed_at": rr["computed_at"],
            "window_days": WINDOW_DAYS, "accounts_in_graph": 160, "edges_in_graph": len(w.edges) + 55, "communities": 4, "communities_kept": 3,
            "stage_seconds": {"read_posts": 281.4, "networks": 190.2, "communities": 12.7, "export": 44.0},
            "edges": {"co_retweet": 700, "co_url": 40, "hashtag_sequence": 55, "fast_retweet": 30, "text_similarity": 25}, "params": PARAMS,
        })
    put("time/pipeline.json", {"kind": "time_pipeline", "stamp": stamp, "runs": sorted((h for h in history if h["run_id"] <= run_id), key=lambda h: h["run_id"], reverse=True)})

    if is_latest:
        matches = [
            {"status": "continuing", "lineage": lineage(0), "cid": 0, "previous_cid": 0, "size": 60, "previous_size": 57, "jaccard": 0.92, "joined": 4, "left": 1},
            {"status": "split", "lineage": lineage(1), "cid": 1, "previous_cid": 1, "size": 30, "previous_size": 44, "jaccard": 0.55, "joined": 0, "left": 14},
            {"status": "merged", "lineage": lineage(2), "cid": 2, "previous_cid": 2, "size": 12, "previous_size": 7, "jaccard": 0.58, "joined": 5, "left": 0},
            {"status": "new", "lineage": lineage(3), "cid": 3, "previous_cid": None, "size": DROPPED[3], "previous_size": None, "jaccard": None, "joined": DROPPED[3], "left": 0},
            {"status": "vanished", "lineage": lineage(9), "cid": None, "previous_cid": 4, "size": None, "previous_size": 9, "jaccard": None, "joined": 0, "left": 9},
        ]
        changes = {"kind": "changes", "stamp": stamp, "previous_run": prev_run_id, "min_jaccard": 0.3, "matches": matches}
    else:
        changes = {"kind": "changes", "stamp": stamp, "previous_run": None, "min_jaccard": 0.3, "matches": []}
    put("time/changes.json", changes)

    if is_latest:
        files.pop("time/coverage.json", None)
    else:
        put("time/coverage.json", {
            "kind": "time_coverage", "stamp": stamp, "denominator_posts": 90000,
            "basis": "Distinct non-repost posts with text in the window. Counts of rows, not a rate.",
            "stages": [{"stage": "embeddings", "covered_posts": 88000, "latest_dt": run["window_end"]},
                       {"stage": "relevance", "covered_posts": 71000, "latest_dt": run["window_end"]},
                       {"stage": "labels", "covered_posts": 12000, "latest_dt": None}],
        })

    overview = {
        "kind": "overview", "stamp": stamp, "caveat_id": "caveat-v1",
        "surfaced": {"accounts_in_graph": 160, "edges_in_graph": len(w.edges) + 55, "communities": len(rows), "communities_kept": len(SIZES),
                     "communities_unscored": 0, "accounts_in_kept_communities": len(kept_members)},
        "trace_edges": {"co_retweet": 700, "co_url": 40, "hashtag_sequence": 55, "fast_retweet": 30, "text_similarity": 25},
        "edge_basis": {"co_retweet_edges_in_kept": sum(1 for e in w.edges.values() if e["mask"] & 1),
                       "with_shared_ge_2": sum(1 for e in w.edges.values() if e["mask"] & 1 and e["shared"] >= 2),
                       "with_shared_ge_3": sum(1 for e in w.edges.values() if e["mask"] & 1 and e["shared"] >= 3)},
        "window_corpus": {"post_rows": 105000, "distinct_posts": 90000, "distinct_authors": 3100, "embedding_coverage": 0.978, "relevance_coverage": None},
    }
    put("overview.json", overview)
    return files


TEMPLATES = (
    {"label": "template_promo", "rate": 0.81, "promo_rate": 0.93, "eligible_posts": 412, "authors": 18},
    {"label": None, "rate": 0.04, "promo_rate": 0.0, "eligible_posts": 260, "authors": 22},
    None,
)


def write_slices(w: World, run_dir: Path) -> dict[str, bytes]:
    out = {}
    sl = run_dir / "slices"
    sl.mkdir(parents=True, exist_ok=True)
    ids = w.all_members
    pos = {a: i for i, a in enumerate(ids)}
    tables = {
        "accounts": pa.table({
            "id": ids, "handle": [w.handle[a] for a in ids], "display_name": [w.display[a] for a in ids],
            "cid": pa.array([w.cid_of[a] for a in ids], pa.int32()),
            "within_score": [w.score[a] for a in ids], "posts_in_window": pa.array([sum(1 for p in w.posts if p["author_id"] == a) for a in ids], pa.int32()),
        }),
        "edges": pa.table({
            "source": pa.array([pos[a] for (a, b) in sorted(w.edges)], pa.int32()),
            "target": pa.array([pos[b] for (a, b) in sorted(w.edges)], pa.int32()),
            "mask": pa.array([w.edges[k]["mask"] for k in sorted(w.edges)], pa.int8()),
            "weight": [w.edges[k]["w"] for k in sorted(w.edges)],
            "shared_retweets": pa.array([w.edges[k]["shared"] for k in sorted(w.edges)], pa.int32()),
        }),
        "posts": pa.table({
            "id": [p["id"] for p in w.posts], "author_id": [p["author_id"] for p in w.posts],
            "created_at": pa.array([p["created_at"] for p in w.posts], pa.timestamp("s", tz="UTC")),
            "kind": [p["kind"] for p in w.posts], "text": [p["text"] for p in w.posts],
            "like_count": pa.array([p["like_count"] for p in w.posts], pa.int32()),
            "repost_count": pa.array([p["repost_count"] for p in w.posts], pa.int32()),
            "reply_count": pa.array([p["reply_count"] for p in w.posts], pa.int32()),
            "hashtags": pa.array([p["hashtags"] for p in w.posts], pa.list_(pa.string())),
        }),
        "hashtags": None,
        "communities": pa.table({
            "cid": pa.array(list(SIZES) + list(DROPPED), pa.int32()),
            "label": [f"Community {c + 1:02d}" for c in list(SIZES) + list(DROPPED)],
            "size": pa.array(list(SIZES.values()) + list(DROPPED.values()), pa.int32()),
            "kept": [True] * len(SIZES) + [False] * len(DROPPED),
        }),
    }
    rows = Counter()
    for p in w.posts:
        for t in p["hashtags"]:
            rows[(t, p["created_at"].date().isoformat())] += 1
    keys = sorted(rows)
    tables["hashtags"] = pa.table({"tag": [k[0] for k in keys], "dt": [k[1] for k in keys], "posts": pa.array([rows[k] for k in keys], pa.int32())})
    for name, tbl in tables.items():
        path = sl / f"{name}.parquet"
        pq.write_table(tbl, path, compression="zstd", write_statistics=False)
        out[f"slices/{name}.parquet"] = (path.read_bytes(), tbl.num_rows)
    return out


def main(argv=None) -> int:
    out = Path((argv or sys.argv[1:] or [str(DEFAULT_OUT)])[0])
    if out.exists():
        shutil.rmtree(out)
    base = out / "explorer" / "v1"
    w = World()
    latest_files: dict[str, bytes] = {}
    meta_rows: dict[str, int] = {}
    for i, run in enumerate(RUNS):
        run_dir = base / "runs" / run["run_id"]
        prev = RUNS[i + 1]["run_id"] if i + 1 < len(RUNS) else None
        files = build_run(w, out, run, i == 0, prev)
        for rel, data in files.items():
            p = run_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        slices = write_slices(w, run_dir)
        if i == 0:
            latest_files = dict(files)
            for rel, (data, n) in slices.items():
                latest_files[rel] = data
                meta_rows[rel] = n
    listing = {}
    for rel, data in sorted(latest_files.items()):
        entry = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if rel in meta_rows:
            entry["rows"] = meta_rows[rel]
        listing[rel] = entry
    manifest = {
        "kind": "manifest", "schema_version": 1, "latest_run": RUNS[0]["run_id"], "generated_at": "2026-10-10T03:52:00Z",
        "method_url": "/methodology",
        "runs": [{"run_id": r["run_id"], "window_id": f"daily-{r['window_end']}-{WINDOW_DAYS}d", "window_end": r["window_end"], "computed_at": r["computed_at"]} for r in RUNS],
        "files": listing,
        "unavailable": [{"view": "time_coverage", "reason": "embedding and relevance listings were not readable this run"}],
    }
    (base / "manifest.json").write_bytes(dumps(manifest))
    print(f"wrote {sum(1 for _ in base.rglob('*') if _.is_file())} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

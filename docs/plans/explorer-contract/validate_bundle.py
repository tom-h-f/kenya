#!/usr/bin/env python3
"""Validate an explorer bundle directory against schema.json.

Usage: validate_bundle.py BUNDLE_ROOT [--denylist FILE]

BUNDLE_ROOT is the directory that contains `explorer/v1/` (the fixture layout
and the producer's --out-dir layout). Exit 0 when clean, 1 with one problem per
line otherwise. Needs jsonschema with format support; pyarrow is optional and
only used to check parquet column names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "schema.json"
PREFIX = "explorer/v1/"

FILE_KEY_TO_DEF = {
    "manifest": "Manifest",
    "overview": "Overview",
    "community_index": "CommunityIndex",
    "community_detail": "CommunityDetail",
    "graph_meta": "GraphMeta",
    "graph_all": "Graph",
    "graph_community": "Graph",
    "graph_community_thin": "Graph",
    "evidence": "Evidence",
    "account_shard": "AccountShard",
    "search_accounts": "SearchAccounts",
    "hashtag_index": "HashtagIndex",
    "hashtag_shard": "HashtagShard",
    "link_index": "LinkIndex",
    "topic_index": "TopicIndex",
    "topic_detail": "TopicDetail",
    "time_collection": "TimeCollection",
    "time_pipeline": "TimePipeline",
    "time_coverage": "TimeCoverage",
    "changes": "Changes",
}

PLACEHOLDERS = {
    "run": r"(?P<run>\d{8}T\d{6}Z)",
    "cid": r"(?P<cid>\d+)",
    "shard": r"(?P<shard>\d+)",
    "n": r"(?P<n>\d+)",
    "tid": r"(?P<tid>\d+)",
    "file": r"(?P<file>[A-Za-z0-9._-]+)",
}

FORBIDDEN_KEY = re.compile(r"hate|incit|toxic|delet|suspend|gone|banned|prevalence", re.I)

VIEW_TO_FILE_KEYS = {
    "time_coverage": ["time_coverage"],
    "topic_index": ["topic_index", "topic_detail"],
    "time_collection": ["time_collection"],
    "time_pipeline": ["time_pipeline"],
    "changes": ["changes"],
    "hashtag_index": ["hashtag_index", "hashtag_shard"],
    "link_index": ["link_index"],
}


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def compile_file_patterns(schema: dict) -> dict[str, re.Pattern]:
    out = {}
    for key, spec in schema["$defs"]["Files"]["properties"].items():
        pattern = re.escape(spec["const"])
        for name, rx in PLACEHOLDERS.items():
            pattern = pattern.replace(re.escape("{" + name + "}"), rx)
        out[key] = re.compile("^" + pattern + "$")
    return out


def classify(rel: str, patterns: dict[str, re.Pattern]):
    for key, rx in patterns.items():
        m = rx.match(rel)
        if m:
            return key, m.groupdict()
    return None, {}


def def_validator(schema: dict, name: str) -> Draft202012Validator:
    wrapper = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": f"#/$defs/{name}",
    }
    return Draft202012Validator(wrapper, format_checker=FormatChecker())


def walk_keys(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            here = f"{path}/{k}"
            yield here, k
            yield from walk_keys(v, here)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk_keys(v, f"{path}/{i}")


def account_shard_of(account_id: str) -> int:
    return int(hashlib.sha1(account_id.encode()).hexdigest()[:8], 16) % 64


def hashtag_shard_of(tag: str) -> int:
    return int(hashlib.sha1(tag.lower().encode()).hexdigest()[0], 16)


class Report:
    def __init__(self):
        self.problems: list[str] = []

    def add(self, where: str, msg: str):
        self.problems.append(f"{where}: {msg}")


def check_schema_file(rel, doc, key, validators, rep):
    name = FILE_KEY_TO_DEF[key]
    errs = sorted(validators[name].iter_errors(doc), key=lambda e: list(e.absolute_path))
    for e in errs[:10]:
        loc = "/".join(str(p) for p in e.absolute_path)
        rep.add(rel, f"schema: {loc or '<root>'}: {e.message[:200]}")
    if len(errs) > 10:
        rep.add(rel, f"schema: {len(errs) - 10} more errors")
    return not errs


DATA_KEYED = {"hashtag_shard": re.compile(r"^/tags/[^/]+$")}
"""Per file kind, the one place a key is a hashtag the author typed rather than a
field name. The key itself is data and may spell any word; every key inside the
value is still a field and is checked."""


def check_forbidden_keys(rel, doc, rep, kind=None):
    data_key = DATA_KEYED.get(kind)
    for path, key in walk_keys(doc):
        if data_key and data_key.match(path):
            continue
        if FORBIDDEN_KEY.search(key):
            rep.add(rel, f"forbidden field name {key!r} at {path}")


def check_graph(rel, g, cids_kept, thin, rep):
    nodes, edges = g["nodes"], g["edges"]
    n = len(nodes["id"])
    for col, arr in nodes.items():
        if len(arr) != n:
            rep.add(rel, f"nodes.{col} length {len(arr)} != {n}")
    m = len(edges["s"])
    for col, arr in edges.items():
        if len(arr) != m:
            rep.add(rel, f"edges.{col} length {len(arr)} != {m}")
    if len(set(nodes["id"])) != n:
        rep.add(rel, "duplicate node ids")
    for c in set(nodes["cid"]):
        if c not in cids_kept:
            rep.add(rel, f"node in community {c} which is not kept")
    for i in range(m):
        s, t = edges["s"][i], edges["t"][i]
        if not (0 <= s < n and 0 <= t < n):
            rep.add(rel, f"edge {i} points outside nodes")
            continue
        if s == t:
            rep.add(rel, f"edge {i} is a self loop")
        mask, shared = edges["mask"][i], edges["shared"][i]
        has_rt = bool(mask & 1)
        if has_rt and shared < 1:
            rep.add(rel, f"edge {i} has co_retweet but shared < 1")
        if not has_rt and shared != 0:
            rep.add(rel, f"edge {i} has no co_retweet but shared != 0")
        if thin:
            if not (mask == 1 and shared == 1):
                rep.add(rel, f"thin edge {i} is not a single shared retweet")
        elif mask == 1 and shared < 2:
            rep.add(rel, f"edge {i} is a single shared retweet and belongs in thin")


def check_evidence(rel, ev, graph, thin_graph, rep):
    edge_count = len(graph["edges"]["s"]) if graph else 0
    thin_count = len(thin_graph["edges"]["s"]) if thin_graph else 0
    for key, item in ev["edges"].items():
        m = re.fullmatch(r"([et])(\d+)", key)
        if not m:
            rep.add(rel, f"bad evidence key {key!r}")
            continue
        idx = int(m.group(2))
        limit = edge_count if m.group(1) == "e" else thin_count
        if idx >= limit:
            rep.add(rel, f"evidence key {key} outside graph ({limit} edges)")
        if len(item["items"]) > 5:
            rep.add(rel, f"{key} has more than 5 items")
        if item["total"] < len(item["items"]):
            rep.add(rel, f"{key} total below item count")
        for it in item["items"]:
            t = it["trace"]
            need = {
                "co_retweet": ["post"],
                "co_url": ["url"],
                "hashtag_sequence": ["tags"],
                "fast_retweet": ["post", "gap_seconds"],
                "text_similarity": ["post_a", "post_b", "similarity"],
            }[t]
            for f in need:
                if f not in it:
                    rep.add(rel, f"{key} {t} item lacks {f}")
    for pid, post in ev["posts"].items():
        if post["id"] != pid:
            rep.add(rel, f"posts map key {pid} != post.id {post['id']}")


def read_parquet_columns(path: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None, None
    f = pq.ParquetFile(path)
    return f.schema_arrow.names, f.metadata.num_rows


def validate_bundle(root: Path, denylist: set[str] | None = None) -> list[str]:
    rep = Report()
    schema = load_schema()
    patterns = compile_file_patterns(schema)
    validators = {n: def_validator(schema, n) for n in set(FILE_KEY_TO_DEF.values())}
    base = root / "explorer" / "v1"
    if not base.is_dir():
        return [f"{root}: no explorer/v1 directory"]

    found: dict[str, tuple[str, dict]] = {}
    for p in sorted(base.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(root).as_posix()
        key, parts = classify(rel, patterns)
        if key is None:
            rep.add(rel, "path matches no pattern in $defs.Files")
            continue
        found[rel] = (key, parts)

    manifest_rel = PREFIX + "manifest.json"
    if manifest_rel not in found:
        return [f"{manifest_rel}: missing"]
    manifest = json.loads((root / manifest_rel).read_text())
    if not check_schema_file(manifest_rel, manifest, "manifest", validators, rep):
        return rep.problems

    run_ids = [r["run_id"] for r in manifest["runs"]]
    latest = manifest["latest_run"]
    if len(run_ids) > 30:
        rep.add(manifest_rel, "more than 30 runs (retention)")
    if latest not in run_ids or run_ids[0] != latest:
        rep.add(manifest_rel, "latest_run must be runs[0]")

    runs_on_disk: dict[str, dict[str, tuple[str, dict, object]]] = {}
    for rel, (key, parts) in found.items():
        if key == "manifest":
            continue
        run = parts.get("run")
        if run is not None and run not in run_ids:
            rep.add(rel, f"run {run} is not listed in the manifest")
        if key.startswith("slice_"):
            runs_on_disk.setdefault(run, {})[rel] = (key, parts, None)
            continue
        if key == "vendor":
            continue
        doc = json.loads((root / rel).read_text())
        ok = check_schema_file(rel, doc, key, validators, rep)
        check_forbidden_keys(rel, doc, rep, key)
        if ok and doc["stamp"]["run_id"] != run:
            rep.add(rel, f"stamp.run_id {doc['stamp']['run_id']} != path run {run}")
        if ok:
            expected_kind = {
                "graph_all": "graph", "graph_community": "graph", "graph_community_thin": "graph",
            }.get(key, key)
            if doc["kind"] != expected_kind:
                rep.add(rel, f"kind {doc['kind']!r} does not match path")
        runs_on_disk.setdefault(run, {})[rel] = (key, parts, doc if ok else None)

    # manifest files listing (latest run only)
    listed = manifest["files"]
    latest_prefix = f"{PREFIX}runs/{latest}/"
    for rel, (key, parts, _) in runs_on_disk.get(latest, {}).items():
        sub = rel[len(latest_prefix):]
        if sub not in listed:
            rep.add(rel, "not listed in manifest.files")
    for sub, meta in listed.items():
        p = root / (latest_prefix + sub)
        if not p.is_file():
            rep.add(manifest_rel, f"files lists {sub} but it does not exist")
            continue
        data = p.read_bytes()
        if len(data) != meta["bytes"]:
            rep.add(manifest_rel, f"{sub}: bytes {meta['bytes']} != actual {len(data)}")
        if hashlib.sha256(data).hexdigest() != meta["sha256"]:
            rep.add(manifest_rel, f"{sub}: sha256 mismatch")

    # unavailable views: absent files, valid names
    file_keys = set(schema["$defs"]["Files"]["properties"])
    present_keys = {k for (k, _, _) in runs_on_disk.get(latest, {}).values()}
    for u in manifest.get("unavailable", []):
        if u["view"] not in file_keys:
            rep.add(manifest_rel, f"unavailable view {u['view']!r} is not a file key")
        for fk in VIEW_TO_FILE_KEYS.get(u["view"], [u["view"]]):
            if fk in present_keys:
                rep.add(manifest_rel, f"{u['view']} is marked unavailable but {fk} exists")
        if not u["reason"].strip():
            rep.add(manifest_rel, f"unavailable {u['view']} has an empty reason")

    for run, files in runs_on_disk.items():
        check_run(run, files, manifest, root, rep, denylist or set())
    return rep.problems


def check_run(run, files, manifest, root, rep, denylist):
    by_key: dict[str, list] = {}
    for rel, (key, parts, doc) in files.items():
        by_key.setdefault(key, []).append((rel, parts, doc))
    if any(doc is None and not key.startswith("slice_") for rel, (key, parts, doc) in files.items()):
        return

    def one(key):
        items = by_key.get(key, [])
        return items[0] if items else (None, None, None)

    idx_rel, _, cidx = one("community_index")
    if cidx is None:
        rep.add(f"runs/{run}", "no community index")
        return
    rows = {c["cid"]: c for c in cidx["communities"]}
    kept = {cid for cid, c in rows.items() if c["kept"]}
    dropped = set(rows) - kept
    for cid in dropped:
        c = rows[cid]
        if "top_hashtags" in c or "top_domains" in c or "template" in c:
            rep.add(idx_rel, f"dropped community {cid} carries content beyond counts")
    labels = [c["label"] for c in rows.values()]
    if len(set(labels)) != len(labels):
        rep.add(idx_rel, "duplicate community labels")
    for c in rows.values():
        if not re.fullmatch(r"Community \d{2,}", c["label"]):
            rep.add(idx_rel, f"label {c['label']!r} is not neutral 'Community NN'")

    kept_members: set[str] = set()
    graphs: dict[int, dict] = {}
    thin_graphs: dict[int, dict] = {}
    for rel, parts, doc in by_key.get("community_detail", []):
        cid = int(parts["cid"])
        if cid not in kept:
            rep.add(rel, f"detail exists for community {cid} which is not kept")
            continue
        if doc["community"] != rows[cid]:
            rep.add(rel, "community row differs from communities.json")
        mem = doc["members"]
        n = len(mem["id"])
        for col, arr in mem.items():
            if len(arr) != n:
                rep.add(rel, f"members.{col} length {len(arr)} != {n}")
        if mem["within_score"] != sorted(mem["within_score"], reverse=True):
            rep.add(rel, "members not sorted by within_score descending")
        if n != rows[cid]["size"]:
            rep.add(rel, f"members {n} != community size {rows[cid]['size']}")
        kept_members.update(mem["id"])
    for cid in kept:
        if not any(int(p["cid"]) == cid for _, p, _ in by_key.get("community_detail", [])):
            rep.add(idx_rel, f"kept community {cid} has no detail file")

    for rel, parts, doc in by_key.get("graph_community", []):
        graphs[int(parts["cid"])] = doc
        check_graph(rel, doc, kept, False, rep)
        if doc["scope"] != "community" or doc.get("cid") != int(parts["cid"]):
            rep.add(rel, "scope/cid do not match path")
    for rel, parts, doc in by_key.get("graph_community_thin", []):
        thin_graphs[int(parts["cid"])] = doc
        check_graph(rel, doc, kept, True, rep)
        if doc["scope"] != "community_thin":
            rep.add(rel, "scope must be community_thin")
    for rel, parts, doc in by_key.get("graph_all", []):
        check_graph(rel, doc, kept, False, rep)
        if doc["scope"] != "all":
            rep.add(rel, "scope must be all")
        for i in doc["nodes"]["id"]:
            if i not in kept_members:
                rep.add(rel, f"node {i} is not a member of a kept community")
    for rel, parts, doc in by_key.get("graph_meta", []):
        for nd in doc["nodes"]:
            if nd["cid"] not in kept:
                rep.add(rel, f"meta node {nd['cid']} not kept")
        for e in doc["edges"]:
            if e["a"] not in kept or e["b"] not in kept:
                rep.add(rel, f"meta edge {e['a']}-{e['b']} touches a dropped community")
    for rel, parts, doc in by_key.get("evidence", []):
        cid = int(parts["cid"])
        if cid not in kept:
            rep.add(rel, f"evidence exists for community {cid} which is not kept")
        check_evidence(rel, doc, graphs.get(cid), thin_graphs.get(cid), rep)

    seen_cards: set[str] = set()
    for rel, parts, doc in by_key.get("account_shard", []):
        shard = int(parts["shard"])
        if doc["shard"] != shard:
            rep.add(rel, "shard field != path")
        for aid, card in doc["accounts"].items():
            if card["id"] != aid:
                rep.add(rel, f"key {aid} != card.id")
            if account_shard_of(aid) != shard:
                rep.add(rel, f"account {aid} is in the wrong shard")
            if card["cid"] not in kept:
                rep.add(rel, f"account {aid} belongs to a dropped community")
            if aid not in kept_members:
                rep.add(rel, f"account {aid} is not in any kept community detail")
            seen_cards.add(aid)
    for rel, parts, doc in by_key.get("search_accounts", []):
        n = len(doc["id"])
        for col in ("handle", "cid", "display_name"):
            if col in doc and len(doc[col]) != n:
                rep.add(rel, f"{col} length != id length")
        for i in doc["id"]:
            if i not in kept_members:
                rep.add(rel, f"search lists {i} which is not in a kept community")
        for c in doc["cid"]:
            if c not in kept:
                rep.add(rel, f"search row in dropped community {c}")
        lowered = [h.lower() for h in doc["handle"]]
        if lowered != sorted(lowered):
            rep.add(rel, "handles not sorted by lower-cased handle")
    if by_key.get("account_shard") and seen_cards != kept_members:
        rep.add(f"runs/{run}", f"account cards cover {len(seen_cards)} of {len(kept_members)} kept members")

    for rel, parts, doc in by_key.get("hashtag_shard", []):
        n = int(parts["n"])
        for tag in doc["tags"]:
            if hashtag_shard_of(tag) != n:
                rep.add(rel, f"tag {tag!r} is in the wrong shard")
            if tag != tag.lower():
                rep.add(rel, f"tag key {tag!r} is not lower case")
    for rel, parts, doc in by_key.get("hashtag_index", []):
        for t in doc["tags"]:
            if hashtag_shard_of(t["tag"]) != t["shard"]:
                rep.add(rel, f"tag {t['tag']!r} lists the wrong shard")
        posts = [t["posts"] for t in doc["tags"]]
        if posts != sorted(posts, reverse=True):
            rep.add(rel, "tags not sorted by posts descending")
    for rel, parts, doc in by_key.get("hashtag_shard", []):
        for tag, t in doc["tags"].items():
            for c in t["communities"]:
                if c["cid"] is not None and c["cid"] not in kept:
                    rep.add(rel, f"tag {tag!r} lists dropped community {c['cid']}")

    ov = one("overview")[2]
    if ov:
        s = ov["surfaced"]
        if s["communities"] != len(rows):
            rep.add("overview", f"surfaced.communities {s['communities']} != {len(rows)} rows")
        if s["communities_kept"] != len(kept):
            rep.add("overview", "surfaced.communities_kept != kept rows")
        if s["accounts_in_kept_communities"] != len(kept_members):
            rep.add("overview", "accounts_in_kept_communities != members of kept details")

    ch = one("changes")[2]
    if ch is not None:
        runs = [r["run_id"] for r in manifest["runs"]]
        if run in runs:
            i = runs.index(run)
            expect = runs[i + 1] if i + 1 < len(runs) else None
            if ch["previous_run"] is not None and ch["previous_run"] != expect:
                rep.add("changes", f"previous_run {ch['previous_run']} != {expect}")
        if ch["previous_run"] is None and ch["matches"]:
            rep.add("changes", "matches present but previous_run is null")

    ti = one("topic_index")[2]
    if ti is not None:
        tids = {t["tid"] for t in ti["topics"]}
        for rel, parts, doc in by_key.get("topic_detail", []):
            if doc["tid"] != int(parts["tid"]) or doc["tid"] not in tids:
                rep.add(rel, "tid not in topic index or differs from path")
            for c in doc["communities"]:
                if c["cid"] not in kept:
                    rep.add(rel, f"topic lists dropped community {c['cid']}")
                if c["posts"] > c["community_posts"]:
                    rep.add(rel, "topic posts exceed community posts")
            fl = doc.get("flags")
            if fl and fl["flagged_posts"] > fl["scored_posts"]:
                rep.add(rel, "flagged_posts exceed scored_posts")

    for rel, (key, parts, _) in files.items():
        if key.startswith("slice_"):
            names, nrows = read_parquet_columns(root / rel)
            if names is None:
                continue
            for c in names:
                if FORBIDDEN_KEY.search(c):
                    rep.add(rel, f"forbidden column {c!r}")
            sub = rel.split(f"runs/{run}/", 1)[1]
            meta = manifest["files"].get(sub)
            if meta and "rows" in meta and meta["rows"] != nrows:
                rep.add(rel, f"manifest rows {meta['rows']} != parquet rows {nrows}")

    if denylist:
        for rel in files:
            text = (root / rel).read_bytes() if rel.endswith(".json") else b""
            for d in denylist:
                if d.encode() in text:
                    rep.add(rel, f"denylisted id {d} present")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--denylist", type=Path)
    args = ap.parse_args(argv)
    deny = set()
    if args.denylist:
        deny = {l.strip() for l in args.denylist.read_text().splitlines() if l.strip() and not l.startswith("#")}
    problems = validate_bundle(args.root, deny)
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)" if problems else "ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

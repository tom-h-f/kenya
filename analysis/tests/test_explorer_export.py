"""The explorer producer on a synthetic world: no R2, no network.

Two kept communities, one dropped for relevance, a non-member whose posts the
members repost, and traces rich enough to give every edge kind some evidence.
The staged bundle is judged by the contract's own validator and language lint.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from kma import explorer_build as eb
from kma import explorer_export as ex

RUN_ID = "20261010T031500Z"
NOW = datetime(2026, 10, 10, 3, 52, tzinfo=timezone.utc)
END = date(2026, 10, 9)
DAYS = 7
START = END - timedelta(days=DAYS - 1)

A = [f"9000000000000000{i:03d}" for i in range(1, 7)]
B = [f"9000000000000000{i:03d}" for i in range(11, 16)]
D = [f"9000000000000000{i:03d}" for i in range(21, 25)]
OUTSIDER = "9000000000000000999"


def _run_row() -> dict:
    return {
        "run_id": RUN_ID, "window_id": f"daily-{END}-{DAYS}d", "window_end": END.isoformat(),
        "window_days": DAYS, "computed_at": pd.Timestamp("2026-10-10T03:41:07Z"),
        "accounts_in_graph": 20, "edges_in_graph": 20, "communities": 3, "communities_kept": 2,
        "communities_unscored": 0, "listed_accounts": 8, "listed_communities": 2,
        "text_threshold": 0.85, "text_overlap": 0.10, "min_entities": 2, "min_kenya": 0.5,
        "edges_co_retweet": 15, "edges_co_url": 1, "edges_hashtag_sequence": 1,
        "edges_fast_retweet": 1, "edges_text_similarity": 1,
        "networks_s": 12.3, "fuse_centrality_s": 1.0, "relevance_s": 2.0, "communities_s": 0.5,
    }


def _communities() -> pd.DataFrame:
    rows = [
        {"community": 0, "size": 6, "edges": 9, "eigenvalue": 3.1, "share_co_retweet": 0.8,
         "share_co_url": 0.2, "kenya_model": 0.9, "relevance_coverage": 0.7,
         "relevance_scored": True, "kept": True, "listed": 6},
        {"community": 1, "size": 5, "edges": 6, "eigenvalue": 2.4, "share_co_retweet": 0.7,
         "share_hashtag_sequence": 0.15, "share_text_similarity": 0.15, "kenya_model": 0.8,
         "relevance_coverage": 0.6, "relevance_scored": True, "kept": True, "listed": 5},
        {"community": 2, "size": 4, "edges": 3, "eigenvalue": 1.9, "share_co_retweet": 1.0,
         "kenya_model": 0.1, "relevance_coverage": 0.5, "relevance_scored": True, "kept": False,
         "listed": 0},
    ]
    return pd.DataFrame(rows)


def _members() -> pd.DataFrame:
    rows = []
    for cid, ids in ((0, A), (1, B), (2, D)):
        for k, uid in enumerate(ids):
            rows.append({"user_id": uid, "community": cid, "within_score": 0.9 - 0.1 * k,
                         "centrality": 0.01 * (k + 1)})
    return pd.DataFrame(rows)


def _edges() -> pd.DataFrame:
    rows = []

    def add(a, b, mask, w, shared):
        lo, hi = sorted((a, b))
        rows.append({"source": lo, "target": hi, "mask": mask, "weight_max": w,
                     "shared_retweets": shared})

    add(A[0], A[1], 1, 0.9, 3)
    add(A[0], A[2], 1, 0.8, 1)
    add(A[1], A[2], 1 | 2, 0.85, 1)
    add(A[2], A[3], 1, 0.7, 2)
    add(A[3], A[4], 8, 0.6, 0)
    add(B[0], B[1], 1, 0.9, 4)
    add(B[1], B[2], 4, 0.5, 0)
    add(B[2], B[3], 16, 0.88, 0)
    add(B[0], B[3], 1, 0.4, 1)
    add(A[0], B[0], 1, 0.6, 2)
    add(A[1], B[1], 1, 0.3, 1)
    add(A[0], D[0], 1, 0.5, 2)
    add(D[0], D[1], 1, 0.5, 1)
    add(A[0], OUTSIDER, 1, 0.5, 1)
    return pd.DataFrame(rows).astype({"mask": "int16", "shared_retweets": "Int32"})


def _traces() -> pd.DataFrame:
    rows = []
    for u in (A[0], A[1]):
        rows += [(u, "co_retweet", p) for p in ("5001", "5002", "5003")]
    rows += [(A[2], "co_retweet", "5001"), (A[1], "co_url", "https://news.example/a"),
             (A[2], "co_url", "https://news.example/a")]
    rows += [(u, "co_retweet", p) for u in (A[2], A[3]) for p in ("5002", "5004")]
    rows += [(B[1], "hashtag_sequence", "water|rent|jobs"), (B[2], "hashtag_sequence", "water|rent|jobs")]
    rows += [(u, "co_retweet", p) for u in (B[0], B[1]) for p in ("5005", "5006", "5007", "5008")]
    return pd.DataFrame(rows, columns=["user_id", "trace", "entity"])


def _posts() -> pd.DataFrame:
    rows = []
    t0 = pd.Timestamp(f"{START}T08:00:00Z")

    def post(pid, author, minutes, text, *, repost_of=None, reply=None, tags=None, urls=None, handle=None):
        rows.append({
            "post_id": pid, "author_id": author, "author_handle": handle or f"h_{author[-3:]}",
            "created_at": t0 + pd.Timedelta(minutes=minutes), "text": text,
            "url": f"https://x.com/h/status/{pid}", "in_reply_to_id": reply,
            "is_repost": repost_of is not None, "repost_of_id": repost_of,
            "like_count": 3, "reply_count": 1, "repost_count": 2, "hashtags": tags or [],
            "urls": urls or [], "collected_at": t0 + pd.Timedelta(days=1)})

    for pid in ("5001", "5002", "5003", "5004"):
        post(pid, OUTSIDER, 1, f"original {pid}", tags=["Water"])
    post("5005", B[0], 5, "b original five", tags=["Rent"])
    post("5006", B[0], 6, "b original six")
    n = 100
    for u in A + B + D:
        post(f"7{n}", u, 600 + n, f"{u[-3:]} says hello", tags=["Water", "rent", "x"],
             urls=["https://news.example/a"])
        n += 1
        post(f"7{n}", u, 700 + n, f"{u[-3:]} replying", reply="7100")
        n += 1
    for u in (A[0], A[1], A[2]):
        post(f"8{n}", u, 2 + n, "RT", repost_of="5001")
        n += 1
    post("8900", A[2], 3, "RT", repost_of="5005")
    post("9999", "9000000000000000555", 1, "old", tags=["gone"])
    frame = pd.DataFrame(rows)
    frame["is_repost"] = frame["is_repost"].astype(bool)
    return frame


def _profiles() -> pd.DataFrame:
    ids = A + B
    return pd.DataFrame({
        "id": ids, "handle": [f"user_{i[-3:]}" for i in ids],
        "display_name": [f"Name {i[-3:]}" for i in ids], "bio": ["a bio"] * len(ids),
        "followers": 10, "following": 5, "statuses": 100,
        "created_at": pd.Timestamp("2019-01-01T00:00:00Z"),
        "profile_as_of": pd.Timestamp("2026-10-01T00:00:00Z"),
    })


def _kenya(ids) -> pd.DataFrame:
    return pd.DataFrame({"posts": 5, "scored": 4, "kenya_model": 0.75}, index=pd.Index(list(ids), name="user_id"))


def make_corpus() -> eb.Corpus:
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    posts = _posts()
    con.register("_posts_df", posts)
    con.execute("CREATE TEMP TABLE _win AS SELECT * FROM _posts_df")
    con.unregister("_posts_df")
    return eb.Corpus(con, START, END)


def make_inputs(**kw) -> ex.Inputs:
    daily = eb.Daily(
        run=_run_row(), communities=_communities(), members=_members(), edges=_edges(),
        traces=_traces(), runs=pd.DataFrame([_run_row()]))
    base = dict(
        daily=daily, profiles_for=_profiles, kenya_for=_kenya, corpus=make_corpus(), post_rows=500,
        trend=pd.DataFrame({"run_id": ["r1"], "tag": ["Water"], "recent_rate": [0.2],
                            "prior_rate": [0.1], "emergence": [2.0], "posts": [30],
                            "censused": [True]}),
        collection={"series": [{"dt": "2026-10-08", "rows": 10, "by_type": {"search": 10}}],
                    "gap_days": []},
        control_arm={"runs": 4, "posts": 40, "truncated_runs": 1, "first": "2026-09-14",
                     "last": "2026-10-09"},
        previous=None, text_evidence=None, denylist=set(), now=NOW)
    base.update(kw)
    return ex.Inputs(**base)


def staged(tmp_path: Path, inp: ex.Inputs, previous_runs=()):
    bundle = ex.assemble(inp)
    manifest = ex.stage(bundle, inp.daily.run, tmp_path, list(previous_runs), NOW)
    return bundle, manifest


def test_the_staged_bundle_passes_the_contract_and_the_guardrails(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs())
    ex.assert_explorer_safe(tmp_path, bundle, set())


def test_the_bundle_holds_exactly_the_kept_communities(tmp_path):
    bundle, manifest = staged(tmp_path, make_inputs())
    index = bundle.files["communities.json"]["communities"]
    assert [c["kept"] for c in index] == [True, True, False]
    assert sorted(f for f in bundle.files if f.startswith("communities/")) == [
        "communities/0.json", "communities/1.json"]
    dropped = index[2]
    assert "top_hashtags" not in dropped and "top_domains" not in dropped
    assert all(D_ not in json.dumps(doc) for doc in bundle.files.values() for D_ in D)
    assert bundle.members == set(A + B)
    assert {u["view"] for u in manifest["unavailable"]} == {"topic_index", "time_coverage"}


def test_drawn_and_thin_edges_split_on_the_default_rule(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs())
    main = bundle.files["graph/c0.json"]["edges"]
    thin = bundle.files["graph/c0.thin.json"]["edges"]
    assert thin["mask"] == [1] and thin["shared"] == [1]
    for mask, shared in zip(main["mask"], main["shared"]):
        assert shared >= 2 or mask != 1
    assert len(main["s"]) == 4
    assert bundle.stats["edges_drawn_all_graph"] == len(bundle.files["graph/all.json"]["edges"]["s"])


def test_the_combined_graph_keeps_cross_community_edges_and_no_outsiders(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs())
    graph = bundle.files["graph/all.json"]
    ids = set(graph["nodes"]["id"])
    assert ids <= set(A + B)
    cids = graph["nodes"]["cid"]
    crossing = [i for i, (s, t) in enumerate(zip(graph["edges"]["s"], graph["edges"]["t"]))
                if cids[s] != cids[t]]
    assert len(crossing) == 1
    meta = bundle.files["graph/meta.json"]
    assert meta["edges"] == [{"a": 0, "b": 1, "edges": 2, "edges_shared_ge_2": 1}]
    assert {n["cid"] for n in meta["nodes"]} == {0, 1}


def test_evidence_names_the_shared_objects_and_caps_the_list(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs())
    ev = bundle.files["evidence/c1.json"]
    graph = bundle.files["graph/c1.json"]
    ids = graph["nodes"]["id"]
    for key, item in ev["edges"].items():
        assert len(item["items"]) <= 5 and item["total"] >= len(item["items"])
    first = next(i for i, (s, t) in enumerate(zip(graph["edges"]["s"], graph["edges"]["t"]))
                 if {ids[s], ids[t]} == {B[0], B[1]})
    entry = ev["edges"][f"e{first}"]
    assert entry["total"] == 4
    assert [i["post"] for i in entry["items"]] == ["5005", "5006", "5007", "5008"]
    assert set(ev["posts"]) == {"5005", "5006"}
    tags = [e for e in ev["edges"].values() if e["items"][0]["trace"] == "hashtag_sequence"]
    assert tags[0]["items"][0]["tags"] == ["water", "rent", "jobs"]


def test_a_fast_retweet_edge_finds_the_original_and_the_gap(tmp_path):
    inp = make_inputs()
    inp.corpus.con.execute(
        """INSERT INTO _win SELECT '8801', ?, 'h', created_at + INTERVAL 20 SECOND, 'RT', url, NULL,
           true, '5002', 1, 1, 1, [], [], collected_at FROM _win WHERE post_id = '5002'""", [A[3]])
    inp.corpus.con.execute(
        """INSERT INTO _win SELECT '8802', ?, 'h', created_at + INTERVAL 30 SECOND, 'RT', url, NULL,
           true, '5002', 1, 1, 1, [], [], collected_at FROM _win WHERE post_id = '5002'""", [A[4]])
    bundle, _ = staged(tmp_path, inp)
    ev = bundle.files["evidence/c0.json"]
    fast = [i for e in ev["edges"].values() for i in e["items"] if i["trace"] == "fast_retweet"]
    assert fast == [{"trace": "fast_retweet", "post": "5002", "gap_seconds": 30}]


def test_text_pairs_come_from_the_injected_reader(tmp_path):
    seen = []

    def reader(pairs):
        seen.extend(pairs)
        a, b = pairs[0]
        return {(a, b): (2, [{"trace": "text_similarity", "post_a": "5005", "post_b": "5006",
                              "similarity": 0.93}])}

    bundle, _ = staged(tmp_path, make_inputs(text_evidence=reader))
    assert seen == [tuple(sorted((B[2], B[3])))]
    item = next(i for e in bundle.files["evidence/c1.json"]["edges"].values() for i in e["items"]
                if i["trace"] == "text_similarity")
    assert item["similarity"] == 0.93


def test_account_cards_cover_every_member_in_the_right_shard(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs())
    cards = {}
    for rel, doc in bundle.files.items():
        if rel.startswith("accounts/"):
            shard = int(rel.split("/")[1].removesuffix(".json"))
            for uid, card in doc["accounts"].items():
                assert eb.account_shard(uid) == shard
                cards[uid] = card
    assert set(cards) == set(A + B)
    card = cards[A[0]]
    assert card["profile"]["display_name"] == "Name 001"
    assert "location" not in json.dumps(card)
    assert card["neighbours_total"] == 5
    assert [n["id"] for n in card["neighbours"]][0] == A[1]
    assert all(n["id"] in set(A + B) for n in card["neighbours"])
    assert card["amplifies"][0]["id"] == OUTSIDER
    assert len(card["recent_posts"]) <= 10
    assert all(p["author_id"] == A[0] for p in card["recent_posts"])


def test_no_per_account_flag_field_exists_anywhere(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs())
    text = json.dumps(bundle.files).lower()
    for word in ("hate", "incit", "toxic", "suspend", "banned", "prevalence"):
        assert f'"{word}' not in text


def test_search_is_sorted_by_lower_cased_handle_and_members_only(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs())
    search = bundle.files["search/accounts.json"]
    assert set(search["id"]) == set(A + B)
    low = [h.lower() for h in search["handle"]]
    assert low == sorted(low)


def test_hashtags_are_lower_cased_sharded_and_examples_are_member_posts(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs())
    index = bundle.files["hashtags/index.json"]
    tags = {t["tag"]: t for t in index["tags"]}
    assert {"water", "rent", "x"} <= set(tags)
    assert tags["water"]["emerging"] is True and index["emerging_run"] == "r1"
    posts = [t["posts"] for t in index["tags"]]
    assert posts == sorted(posts, reverse=True)
    shard = bundle.files[f"hashtags/s{eb.hashtag_shard('water')}.json"]
    entry = shard["tags"]["water"]
    assert any(c["cid"] is None for c in entry["communities"])
    for ex_ in entry["examples"]:
        assert ex_["author_id"] in set(A + B)


def test_links_count_domains_and_urls(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs())
    links = bundle.files["links/index.json"]
    assert links["domains"][0]["domain"] == "news.example"
    assert links["urls"][0]["url"] == "https://news.example/a"
    assert links["urls"][0]["top_cids"][0] in (0, 1)


def test_the_overview_states_edge_basis_and_leaves_unmeasured_coverage_null(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs())
    ov = bundle.files["overview.json"]
    assert ov["surfaced"]["communities"] == 3 and ov["surfaced"]["communities_kept"] == 2
    assert ov["surfaced"]["accounts_in_kept_communities"] == 11
    basis = ov["edge_basis"]
    assert basis["with_shared_ge_2"] <= basis["co_retweet_edges_in_kept"]
    assert ov["window_corpus"]["embedding_coverage"] is None
    assert ov["stamp"]["params"]["leiden_seed"] == 0


def test_community_detail_counts_days_and_kinds(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs())
    detail = bundle.files["communities/0.json"]
    assert len(detail["daily_posts"]) == DAYS
    assert sum(d["originals"] for d in detail["daily_posts"]) == 6
    assert sum(d["replies"] for d in detail["daily_posts"]) == 6
    assert detail["members"]["within_score"] == sorted(detail["members"]["within_score"], reverse=True)
    assert detail["amplified"][0]["id"] == OUTSIDER
    assert detail["amplified_coverage"] == 1.0


def test_the_denylist_removes_an_account_from_every_file(tmp_path):
    victim = A[1]
    bundle, _ = staged(tmp_path, make_inputs(denylist={victim}))
    for rel, doc in bundle.files.items():
        assert victim not in json.dumps(doc), rel
    ex.assert_explorer_safe(tmp_path, bundle, {victim})
    sizes = {c["cid"]: c["size"] for c in bundle.files["communities.json"]["communities"]}
    assert sizes[0] == 5


def test_a_denylisted_authors_posts_are_not_named_as_evidence(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs(denylist={B[0]}))
    text = json.dumps(bundle.files)
    assert '"5005"' not in text and '"5006"' not in text


def test_a_leaked_id_fails_the_safety_check(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs())
    bundle.files["search/accounts.json"]["id"][0] = "9000000000000000777"
    with pytest.raises(ex.ExplorerUnsafe, match="outside the allowed set"):
        ex.assert_explorer_safe(tmp_path, bundle, set())


def test_a_verdict_word_in_written_copy_fails_the_lint(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs())
    path = tmp_path / "explorer/v1/runs" / RUN_ID / "hashtags/index.json"
    doc = json.loads(path.read_text())
    doc["basis"] = "accounts that look coordinated"
    path.write_text(json.dumps(doc))
    with pytest.raises(ex.ExplorerUnsafe):
        ex.assert_explorer_safe(tmp_path, bundle, set())


def test_a_flag_field_fails_the_contract(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs())
    shard = next(r for r in bundle.files if r.startswith("accounts/"))
    path = tmp_path / "explorer/v1/runs" / RUN_ID / shard
    doc = json.loads(path.read_text())
    next(iter(doc["accounts"].values()))["hate_flag"] = True
    path.write_text(json.dumps(doc))
    with pytest.raises(ex.ExplorerUnsafe):
        ex.assert_explorer_safe(tmp_path, bundle, set())


def test_a_tag_that_spells_a_forbidden_word_is_data_not_a_field(tmp_path):
    inp = make_inputs()
    inp.corpus.con.execute(
        "UPDATE _win SET hashtags = ['hatespeech', 'Water'] WHERE post_id = '7100'")
    bundle, _ = staged(tmp_path, inp)
    assert any("hatespeech" in doc.get("tags", {}) for rel, doc in bundle.files.items()
               if rel.startswith("hashtags/s"))
    ex.assert_explorer_safe(tmp_path, bundle, set())


def test_edges_without_trace_rows_are_refused(tmp_path):
    inp = make_inputs()
    inp.daily.edges["shared_retweets"] = pd.array([pd.NA] * len(inp.daily.edges), dtype="Int32")
    with pytest.raises(ValueError, match="shared_retweets"):
        ex.assemble(inp)


def test_the_manifest_lists_every_file_with_its_hash_and_keeps_30_runs(tmp_path):
    older = [{"run_id": f"202610{d:02d}T031500Z", "window_id": f"daily-2026-10-{d:02d}-7d",
              "window_end": f"2026-10-{d:02d}", "computed_at": f"2026-10-{d:02d}T03:41:07Z"}
             for d in range(9, 0, -1)] * 4
    bundle, manifest = staged(tmp_path, make_inputs(), previous_runs=older)
    assert manifest["latest_run"] == RUN_ID and manifest["runs"][0]["run_id"] == RUN_ID
    assert len(manifest["runs"]) == 30
    assert set(manifest["files"]) == set(bundle.files)


class FakeS3:
    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self, existing: dict[str, bytes] | None = None):
        self.objects = dict(existing or {})
        self.puts: list[tuple[str, dict]] = []
        self.fail_on: str | None = None

    def put_object(self, Bucket, Key, Body, **kw):
        if self.fail_on and self.fail_on in Key:
            raise RuntimeError("put failed")
        self.objects[Key] = Body
        self.puts.append((Key, kw))

    def get_paginator(self, _):
        client = self

        class Pager:
            def paginate(self, Bucket, Prefix, Delimiter=None):
                keys = sorted(k for k in client.objects if k.startswith(Prefix))
                if Delimiter:
                    prefixes = sorted({Prefix + k[len(Prefix):].split("/")[0] + "/" for k in keys
                                       if "/" in k[len(Prefix):]})
                    yield {"CommonPrefixes": [{"Prefix": p} for p in prefixes]}
                else:
                    yield {"Contents": [{"Key": k} for k in keys]}

        return Pager()

    def delete_objects(self, Bucket, Delete):
        for o in Delete["Objects"]:
            self.objects.pop(o["Key"], None)


def test_publish_writes_the_manifest_last_and_prunes_only_old_runs(tmp_path):
    bundle, manifest = staged(tmp_path, make_inputs())
    old = "20200101T000000Z"
    client = FakeS3({
        f"explorer/v1/runs/{old}/overview.json": b"{}",
        "explorer/v1/vendor/duckdb.wasm": b"w", "public/summary.json": b"{}",
    })
    ex.publish(tmp_path, manifest, bucket="kenya-monitor-dashboard", client=client, run_id=RUN_ID)
    keys = [k for k, _ in client.puts]
    assert keys[-1] == "explorer/v1/manifest.json"
    assert all(k.startswith(f"explorer/v1/runs/{RUN_ID}/") for k in keys[:-1])
    assert len(keys) == len(bundle.files) + 1
    assert f"explorer/v1/runs/{old}/overview.json" not in client.objects
    assert "explorer/v1/vendor/duckdb.wasm" in client.objects and "public/summary.json" in client.objects
    cache = dict(client.puts)
    assert "immutable" in dict(client.puts[0][1])["CacheControl"]
    assert "max-age=60" in cache["explorer/v1/manifest.json"]["CacheControl"]


def test_a_failed_upload_never_reaches_the_manifest(tmp_path):
    _, manifest = staged(tmp_path, make_inputs())
    client = FakeS3()
    client.fail_on = "overview.json"
    with pytest.raises(RuntimeError):
        ex.publish(tmp_path, manifest, bucket="kenya-monitor-dashboard", client=client, run_id=RUN_ID)
    assert "explorer/v1/manifest.json" not in client.objects


def test_publish_refuses_the_raw_data_bucket(tmp_path):
    _, manifest = staged(tmp_path, make_inputs())
    from kma import db
    with pytest.raises(ValueError, match="raw data bucket"):
        ex.publish(tmp_path, manifest, bucket=db.BUCKET, client=FakeS3(), run_id=RUN_ID)


def test_an_unreadable_posts_source_publishes_nothing(tmp_path, monkeypatch):
    client = FakeS3({"explorer/v1/manifest.json": json.dumps({"runs": []}).encode()})
    before = dict(client.objects)
    monkeypatch.setattr(ex, "read_daily", lambda con, root, run_id: make_inputs().daily)

    def broken(con, window_id):
        raise RuntimeError("posts unreadable")

    monkeypatch.setattr(ex, "load_window", broken)
    monkeypatch.setattr(ex.dashboard, "r2_sources", lambda con: None)
    monkeypatch.setattr(ex, "read_published_manifest", lambda c, b: {"runs": []})
    with pytest.raises(RuntimeError, match="posts unreadable"):
        ex.run(out_dir=None, daily_root=None, run_id=None, denylist=set(), duckdb_memory=None,
               text_evidence=False, client=client, con=duckdb.connect())
    assert client.objects == before and client.puts == []


def test_the_out_dir_refuses_to_overwrite_a_bundle(tmp_path):
    (tmp_path / "explorer" / "v1").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="already exists"):
        ex.run(out_dir=tmp_path, daily_root=None, run_id=None, denylist=set(), duckdb_memory=None,
               text_evidence=False, con=duckdb.connect())


def test_denylist_file_ignores_comments_and_blanks(tmp_path):
    path = tmp_path / "deny.txt"
    path.write_text("# ids only\n\n123\n  456  \n")
    assert ex.read_denylist(path) == {"123", "456"}
    assert ex.read_denylist(tmp_path / "missing.txt") == set()


def test_size_report_counts_gzip_below_raw(tmp_path):
    staged(tmp_path, make_inputs())
    report = ex.size_report(tmp_path)
    assert report["total"]["files"] > 10
    assert 0 < report["total"]["gzip"] < report["total"]["raw"]
    assert "graph" in report["groups"] and "accounts" in report["groups"]


def test_an_emerging_tag_with_no_prior_rate_is_exported_with_null_emergence(tmp_path):
    trend = pd.DataFrame({"run_id": ["r1", "r1", "r1"], "tag": ["Water", "Fresh", "Dead"],
                          "recent_rate": [0.2, 0.3, 0.0], "prior_rate": [0.1, 0.0, 0.0],
                          "emergence": [2.0, float("inf"), float("nan")], "posts": [30, 5, 0],
                          "censused": [True, False, False]})
    bundle, _ = staged(tmp_path, make_inputs(trend=trend))
    ex.assert_explorer_safe(tmp_path, bundle, set())
    emerging = {e["tag"]: e for e in bundle.files["hashtags/index.json"]["emerging"]}
    assert set(emerging) == {"water", "fresh"}
    assert emerging["fresh"]["emergence"] is None
    assert emerging["water"]["emergence"] == 2.0


def test_a_tag_key_is_exempt_but_a_flag_field_inside_a_tag_entry_is_not(tmp_path):
    inp = make_inputs()
    inp.corpus.con.execute(
        "UPDATE _win SET hashtags = ['hatespeech', 'Water'] WHERE post_id = '7100'")
    bundle, _ = staged(tmp_path, inp)
    shard_rel = next(rel for rel, doc in bundle.files.items()
                     if rel.startswith("hashtags/s") and "hatespeech" in doc["tags"])
    path = tmp_path / "explorer/v1/runs" / RUN_ID / shard_rel
    doc = json.loads(path.read_text())
    doc["tags"]["hatespeech"]["hate_share"] = 0.2
    path.write_text(json.dumps(doc))
    with pytest.raises(ex.ExplorerUnsafe, match="hate_share"):
        ex.assert_explorer_safe(tmp_path, bundle, set())


def test_a_kept_community_row_carries_the_template_flag_and_a_dropped_one_does_not(tmp_path):
    inp = make_inputs()
    comms = inp.daily.communities
    for col, vals in {"template_label": ["template_promo", None, None],
                      "template_rate": [0.8, 0.0, 0.5], "template_promo_rate": [0.9, 0.0, 0.0],
                      "template_link_rate": [1.0, 0.0, 0.0], "template_eligible": [40, 30, 9],
                      "template_authors": [6, 5, 4]}.items():
        comms[col] = vals
    bundle, _ = staged(tmp_path, inp)
    ex.assert_explorer_safe(tmp_path, bundle, set())
    rows = {c["cid"]: c for c in bundle.files["communities.json"]["communities"]}
    assert rows[0]["template"] == {"label": "template_promo", "rate": 0.8, "promo_rate": 0.9,
                                   "eligible_posts": 40, "authors": 6}
    assert rows[1]["template"]["label"] is None
    assert "template" not in rows[2]


def test_a_run_without_the_template_columns_exports_null(tmp_path):
    bundle, _ = staged(tmp_path, make_inputs())
    ex.assert_explorer_safe(tmp_path, bundle, set())
    rows = {c["cid"]: c for c in bundle.files["communities.json"]["communities"]}
    assert rows[0]["template"] is None


def test_an_edge_with_many_traces_still_lists_at_most_five_items_across_them(tmp_path):
    inp = make_inputs()
    rows = [(u, "co_url", f"https://a.example/{i}") for u in (A[0], A[1]) for i in range(8)]
    inp.daily.traces = pd.concat(
        [inp.daily.traces, pd.DataFrame(rows, columns=["user_id", "trace", "entity"])],
        ignore_index=True)
    inp.daily.edges.loc[(inp.daily.edges["source"] == A[0]) & (inp.daily.edges["target"] == A[1]),
                        "mask"] = 3
    bundle, _ = staged(tmp_path, inp)
    ev = bundle.files["evidence/c0.json"]["edges"]
    big = max(ev.values(), key=lambda e: e["total"])
    assert big["total"] == 11 and len(big["items"]) == 5
    assert {i["trace"] for i in big["items"]} == {"co_retweet", "co_url"}
    ex.assert_explorer_safe(tmp_path, bundle, set())


def _profile_objects(tmp_path, spec: dict[str, list[str]]) -> list[tuple[str, str]]:
    listing = []
    for dt, ids in spec.items():
        path = tmp_path / f"dt={dt}" / "run=1.parquet"
        path.parent.mkdir(parents=True)
        pd.DataFrame({
            "platform_user_id": ids, "handle": [f"h{i}_{dt}" for i in ids], "display_name": "n",
            "bio": "b", "followers_count": 1, "following_count": 1, "tweet_count": 1,
            "created_at": pd.Timestamp("2020-01-01", tz="UTC"),
            "collected_at": pd.Timestamp(f"{dt}T01:00:00Z"),
        }).to_parquet(path)
        listing.append((dt, str(path)))
    return sorted(listing, reverse=True)


def _members_table(ids: list[str]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("_m", pd.DataFrame({"id": ids, "cid": 0}))
    con.execute("CREATE TEMP TABLE _mem AS SELECT * FROM _m")
    return con


def test_profiles_older_than_the_window_are_found_newest_first(tmp_path):
    listing = _profile_objects(tmp_path, {
        "2026-10-05": ["1", "2"], "2026-09-20": ["2", "3"], "2026-09-01": ["3", "4"]})
    con = _members_table(["1", "2", "3", "4", "5"])
    got = ex.read_profiles(con, date(2026, 10, 1), listing=listing, older_batch=1)
    by = got.set_index("id")["handle"].to_dict()
    assert by == {"1": "h1_2026-10-05", "2": "h2_2026-10-05", "3": "h3_2026-09-20",
                  "4": "h4_2026-09-01"}
    stats = got.attrs["profile_read"]
    assert (stats["found_in_window"], stats["found_in_older"], stats["without_profile"]) == (2, 2, 1)
    assert stats["older_objects_read"] == 2


def test_the_older_scan_stops_when_everyone_is_found_or_the_budget_is_spent(tmp_path):
    listing = _profile_objects(tmp_path, {"2026-10-05": ["1"], "2026-09-20": ["2"], "2026-09-01": ["3"]})
    con = _members_table(["1", "2"])
    got = ex.read_profiles(con, date(2026, 10, 1), listing=listing, older_batch=1)
    assert got.attrs["profile_read"]["older_objects_read"] == 1
    con = _members_table(["1", "2", "3"])
    ticks = iter([0.0, 0.0, 100.0, 200.0, 300.0])
    got = ex.read_profiles(con, date(2026, 10, 1), listing=listing, older_batch=1,
                           older_seconds=50, clock=lambda: next(ticks))
    assert got.attrs["profile_read"]["older_objects_read"] == 1
    assert got.attrs["profile_read"]["without_profile"] == 1

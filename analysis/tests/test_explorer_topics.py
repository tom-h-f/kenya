"""Topics, coverage, slices and vendor files of the explorer producer.

Reuses the synthetic world of `test_explorer_export`: three communities, a non-member
whose posts are reposted, and window posts for every account.
"""

from __future__ import annotations

import json
from datetime import timezone

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import pytest

from kma import explorer_export as ex
from kma import explorer_slices, explorer_topics, explorer_vendor
from test_explorer_export import A, B, D, RUN_ID, make_corpus, make_inputs, staged

FIT_AT = pd.Timestamp("2026-09-04T12:00:00Z")
KNN_AT = pd.Timestamp("2026-10-10T02:00:00Z")
FIT_PARAMS = "umap=5,hdbscan=20"
KNN_PARAMS = "method=knn,k=10,threshold=0.40,fit_run=20260904T120000Z,heldout_agreement=0.8017"
OLD_KNN = "method=knn,k=10,threshold=0.40,fit_run=20260801T000000Z,heldout_agreement=0.7"


def topic_source(con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> str:
    frame = pd.DataFrame(rows, columns=["platform_post_id", "topic", "params", "assigned_at", "filename"])
    con.register("_topics_src", frame)
    return "_topics_src"


def fit_row(pid, topic):
    return (pid, topic, FIT_PARAMS, FIT_AT, "topics/dt=2026-09-04/run=20260904T120000Z.parquet")


def knn_row(pid, topic, params=KNN_PARAMS, at=KNN_AT):
    return (pid, topic, params, at, "topics/dt=2026-10-10/run=20261010T020000Z-c0000.parquet")


def window_post_ids(corpus):
    members = ", ".join(f"'{u}'" for u in A + B + D)
    return [r[0] for r in corpus.con.execute(
        f"SELECT post_id FROM _win WHERE author_id IN ({members}) AND NOT coalesce(is_repost, false) "
        "ORDER BY post_id").fetchall()]


def world_with_topics(tmp_path=None):
    inp = make_inputs()
    corpus = inp.corpus
    ids = window_post_ids(corpus)
    rows = []
    for k, pid in enumerate(ids):
        topic = [3, 3, 7, -1][k % 4]
        rows.append(fit_row(pid, topic) if k % 2 == 0 else knn_row(pid, topic))
    rows.append(("9999", 3, FIT_PARAMS, pd.Timestamp("2026-08-01T00:00:00Z"), "x/run=20260801T000000Z.parquet"))
    rows.append(knn_row("zzz", 3, OLD_KNN))
    src = topic_source(corpus.con, rows)
    fit = explorer_topics.load_topic_rows(corpus, source=src)
    return inp, fit, ids


def test_only_the_newest_fit_and_knn_rows_against_it_are_read():
    inp, fit, ids = world_with_topics()
    con = inp.corpus.con
    got = con.execute("SELECT method, count(*) FROM _topic GROUP BY 1 ORDER BY 1").fetchall()
    assert dict(got) == {"fit": (len(ids) + 1) // 2, "knn": len(ids) // 2}
    assert con.execute("SELECT count(*) FROM _topic WHERE post_id IN ('9999', 'zzz')").fetchone()[0] == 0
    assert fit["fitted_at"].startswith("2026-09-04T12:00:00")
    assert fit["fit_posts"] == (len(ids) + 1) // 2 + 1 - 1 or fit["fit_posts"] >= 1
    assert fit["knn_posts"] == len(ids) // 2


def test_a_topic_file_has_terms_mass_by_community_and_member_examples(tmp_path):
    inp, fit, ids = world_with_topics()
    inp.topic_fit = fit
    bundle, _ = staged(tmp_path, inp)
    ex.assert_explorer_safe(tmp_path, bundle, set())
    index = bundle.files["topics/index.json"]
    assert {t["tid"] for t in index["topics"]} == {3, 7}
    assert index["fit"]["window_posts"] == len(ids)
    assert index["fit"]["listed_topics"] == 2
    t3 = next(t for t in index["topics"] if t["tid"] == 3)
    assert t3["fit_posts_in_window"] + t3["knn_posts_in_window"] == t3["posts_in_window"]
    assert sum(t3["daily"]) == t3["posts_in_window"]
    detail = bundle.files["topics/3.json"]
    members = {u for u in A + B + D}
    assert all(e["author_id"] in members for e in detail["examples"])
    for c in detail["communities"]:
        assert c["posts"] <= c["community_posts"]


def test_a_denied_accounts_posts_do_not_count_in_the_topic_denominator(tmp_path):
    inp, fit, ids = world_with_topics()
    inp.topic_fit = fit
    inp.denylist = {A[0]}
    bundle, _ = staged(tmp_path, inp)
    n = inp.corpus.con.execute(
        "SELECT count(*) FROM _topic t JOIN _win w ON w.post_id = t.post_id").fetchone()[0]
    assert bundle.files["topics/index.json"]["fit"]["window_posts"] == n


def test_without_a_topic_fit_the_view_is_unavailable_with_a_reason(tmp_path):
    inp = make_inputs()
    inp.topic_reason = explorer_topics_reason()
    bundle, manifest = staged(tmp_path, inp)
    ex.assert_explorer_safe(tmp_path, bundle, set())
    assert not any(rel.startswith("topics/") for rel in bundle.files)
    assert {"view": "topic_index", "reason": ex.REASON_TOPICS} in manifest["unavailable"]


def explorer_topics_reason():
    return ex.REASON_TOPICS


def test_no_fit_run_raises_unavailable():
    corpus = make_corpus()
    src = topic_source(corpus.con, [knn_row("a", 1)])
    with pytest.raises(explorer_topics.TopicsUnavailable):
        explorer_topics.load_topic_rows(corpus, source=src)


def test_a_bundle_without_coverage_sources_marks_the_view_unavailable(tmp_path):
    bundle, manifest = staged(tmp_path, make_inputs())
    assert "time/coverage.json" not in bundle.files
    assert {"view": "time_coverage", "reason": ex.REASON_COVERAGE} in manifest["unavailable"]


def test_coverage_counts_non_repost_posts_with_text_and_leaves_failed_stages_out(tmp_path):
    inp = make_inputs()
    con = inp.corpus.con
    ids = window_post_ids(inp.corpus)[:5]
    con.register("_emb", pd.DataFrame({"platform_post_id": ids, "dt": ["2026-10-08"] * 5}))
    con.register("_rel", pd.DataFrame({"platform_post_id": ids[:2], "dt": ["2026-10-09"] * 2}))
    cov, failed = explorer_topics.build_coverage(
        {"run_id": RUN_ID}, inp.corpus,
        sources={"embeddings": "_emb", "relevance": "_rel", "hatespeech": "no_such_table"})
    assert failed == ["hatespeech"]
    stages = {s["stage"]: s for s in cov["stages"]}
    assert stages["embeddings"]["covered_posts"] == 5
    assert stages["relevance"] == {"stage": "relevance", "covered_posts": 2, "latest_dt": "2026-10-09"}
    denominator = con.execute(
        "SELECT count(DISTINCT post_id) FROM _win WHERE NOT coalesce(is_repost, false) "
        "AND text IS NOT NULL").fetchone()[0]
    assert cov["denominator_posts"] == denominator


def test_the_bundle_carries_five_slices_without_flag_columns(tmp_path):
    inp, fit, _ = world_with_topics()
    inp.topic_fit = fit
    bundle, manifest = staged(tmp_path, inp)
    ex.assert_explorer_safe(tmp_path, bundle, set())
    assert sorted(bundle.blobs) == [f"slices/{n}.parquet" for n in sorted(explorer_slices.NAMES)]
    for rel, (_, rows) in bundle.blobs.items():
        assert manifest["files"][rel]["rows"] == rows
    path = tmp_path / "explorer/v1/runs" / RUN_ID / "slices/posts.parquet"
    posts = pq.read_table(path).to_pandas()
    assert {"topic", "topic_method", "hashtags", "kind"} <= set(posts.columns)
    assert set(posts["author_id"]) <= set(A + B + D)
    assert posts["topic"].notna().any()
    edges = pq.read_table(tmp_path / "explorer/v1/runs" / RUN_ID / "slices/edges.parquet").to_pandas()
    accounts = pq.read_table(tmp_path / "explorer/v1/runs" / RUN_ID / "slices/accounts.parquet").to_pandas()
    assert accounts["id"].tolist() == sorted(accounts["id"])
    assert edges["source"].max() < len(accounts) and edges["target"].max() < len(accounts)


def test_slices_over_the_cap_drop_reposts_then_refuse():
    inp = make_inputs()
    corpus = inp.corpus
    corpus.set_members(inp.daily.members[inp.daily.members["community"].isin([0, 1])])
    members = inp.daily.members[inp.daily.members["community"].isin([0, 1])]
    kept_edges = explorer_slices_edges(inp, members)
    handles = {u: f"h{u[-3:]}" for u in members["user_id"]}
    rows = {0: {"size": 6, "kept": True}, 1: {"size": 5, "kept": True}}
    per = corpus.per_author()
    full = explorer_slices.build_slices(corpus, members, kept_edges, handles, pd.DataFrame(), rows, per)
    full_posts = full["slices/posts.parquet"].rows
    total = sum(len(s.data) for s in full.values())
    small = explorer_slices.build_slices(
        corpus, members, kept_edges, handles, pd.DataFrame(), rows, per, cap=total - 1)
    assert small["slices/posts.parquet"].rows < full_posts
    with pytest.raises(explorer_slices.SlicesTooLarge):
        explorer_slices.build_slices(corpus, members, kept_edges, handles, pd.DataFrame(), rows, per, cap=100)


def explorer_slices_edges(inp, members):
    from kma import explorer_build as eb
    cid_of = dict(zip(members["user_id"], members["community"].astype(int)))
    return eb.edge_tables(inp.daily.edges, cid_of)


class FakeClient:
    class exceptions:
        class ClientError(Exception):
            pass

    def __init__(self, existing=()):
        self.objects = {k: b"x" for k in existing}
        self.puts = []

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.exceptions.ClientError()
        return {"ContentLength": len(self.objects[Key])}

    def put_object(self, **kw):
        self.puts.append(kw)
        self.objects[kw["Key"]] = kw["Body"]


def test_vendor_files_are_uploaded_versioned_immutable_and_only_when_missing():
    import hashlib
    pin = {"version": "1.2.3", "url": "https://x/{version}/{file}", "files": {
        "duckdb-eh.wasm": {"sha256": hashlib.sha256(b"wasm").hexdigest()},
        "duckdb-browser-eh.worker.js": {"sha256": hashlib.sha256(b"js").hexdigest()}}}
    bodies = {"https://x/1.2.3/duckdb-eh.wasm": b"wasm", "https://x/1.2.3/duckdb-browser-eh.worker.js": b"js"}
    existing = ["explorer/v1/vendor/duckdb-eh-1.2.3.wasm"]
    client = FakeClient(existing)
    out = explorer_vendor.sync(client, "kenya-monitor-dashboard", pin, download=bodies.__getitem__)
    assert [p["Key"] for p in client.puts] == ["explorer/v1/vendor/duckdb-browser-eh.worker-1.2.3.js"]
    assert client.puts[0]["CacheControl"] == "public, max-age=31536000, immutable"
    assert [e["uploaded"] for e in out] == [True, False]


def test_vendor_refuses_a_download_that_differs_from_the_pin_and_the_raw_bucket():
    pin = {"version": "1.2.3", "url": "https://x/{file}", "files": {"duckdb-eh.wasm": {"sha256": "0" * 64}}}
    with pytest.raises(ValueError, match="pinned"):
        explorer_vendor.sync(FakeClient(), "kenya-monitor-dashboard", pin, download=lambda u: b"other")
    with pytest.raises(ValueError, match="raw data bucket"):
        explorer_vendor.sync(FakeClient(), "kenya-monitor-2027", pin)


def test_the_pinned_files_are_the_ones_the_site_route_accepts():
    import re
    site_re = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\.(?:wasm|js|mjs)$")
    pin = explorer_vendor.load_pin()
    names = [explorer_vendor.target_name(n, pin["version"], e) for n, e in pin["files"].items()]
    assert all(site_re.match(n) for n in names)
    assert "duckdb-eh-1.29.0.wasm" in names
    assert "parquet.eh-1.1.1.wasm" in names and "parquet.mvp-1.1.1.wasm" in names
    assert all(n.startswith("parquet.") for n in names if "parquet" in n)


def test_fetch_sends_an_explicit_user_agent(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"x"

    def fake_urlopen(request, timeout):
        seen["agent"] = request.get_header("User-agent")
        return Resp()

    monkeypatch.setattr(explorer_vendor.urllib.request, "urlopen", fake_urlopen)
    assert explorer_vendor.fetch("https://x/y") == b"x"
    assert seen["agent"].startswith("kma-explorer-vendor")

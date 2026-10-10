"""kma.topics_assign: knn assignment against a fit, on synthetic embeddings.

No R2 and no GPU: a directory-backed store stands in for the bucket and the
search runs on CPU in float32.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kma import topics_assign as ta
from kma.semantic import MODEL as SEMANTIC_MODEL
from kma.semantic import _load_topics_sql

EMB = ta.embeddings_prefix()
TOP = ta.topics_prefix()
FIT_AT = datetime(2026, 9, 4, 8, 48, tzinfo=timezone.utc)


class DirStore:
    """The bucket as a directory, with the same surface as ManifestStore."""

    def __init__(self, root: Path, allowed_puts: set[str] | None = None):
        self.root = root
        self._puts = allowed_puts
        self.put_log: list[str] = []

    def list(self, prefix):
        base = self.root / prefix
        if not base.exists():
            return []
        return sorted(
            (ta.Obj(p.relative_to(self.root).as_posix(), p.stat().st_size) for p in base.rglob("*") if p.is_file()),
            key=lambda o: o.key,
        )

    def get(self, key, dest):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes((self.root / key).read_bytes())

    def put(self, key, src):
        if self._puts is not None and key not in self._puts:
            raise KeyError(key)
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(src).read_bytes())
        self.put_log.append(key)

    def put_keys(self):
        return sorted(self._puts) if self._puts is not None else []


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def centers(n, seed=0):
    return unit(np.random.default_rng(seed).normal(size=(n, ta.DIM)))


def cluster_vecs(center, n, rng, spread=0.12):
    return unit(center + spread * unit(rng.normal(size=(n, ta.DIM))))


def write_embeddings(root: Path, name: str, ids, vecs, day="2026-07-16"):
    path = root / EMB / f"dt={day}" / f"run={name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "platform_post_id": ids,
            "model": ["m"] * len(ids),
            "dim": [ta.DIM] * len(ids),
            "embedding": pa.array([v.tolist() for v in vecs], pa.list_(pa.float64())),
            "embedded_at": [FIT_AT] * len(ids),
        }
    )
    pq.write_table(table, path)
    return path


def write_topics(root: Path, name: str, ids, topics, params, at=FIT_AT, day="2026-09-04"):
    path = root / TOP / f"dt={day}" / f"run={name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    table = pa.table(
        {
            "platform_post_id": ids,
            "topic": pa.array(topics, pa.int64()),
            "model": ["paraphrase-multilingual-mpnet-base-v2"] * n,
            "params": [params] * n,
            "assigned_at": pa.array([at] * n, pa.timestamp("us", tz="UTC")),
        }
    )
    pq.write_table(table, path)
    return path


FIT_PARAMS = "min_cluster_size=60,n_neighbors=15,n_components=5,random_state=42,backend=gpu"
SETTINGS = ta.Settings(
    k=5, power=1, feature="support", threshold=0.6, include_noise=False,
    heldout_agreement=0.9712, noise_rejected=0.8, balanced=0.88,
)


@pytest.fixture
def bucket(tmp_path):
    """3 topics of 40 posts in the fit, 20 fit noise posts, 3 unassigned posts
    per topic, and 5 unassigned posts far from every topic."""
    rng = np.random.default_rng(1)
    cs = centers(3)
    far = unit(rng.normal(size=(5, ta.DIM)))
    fit_ids, fit_vecs, fit_topics = [], [], []
    for t in range(3):
        fit_ids += [f"f{t}-{i}" for i in range(40)]
        fit_vecs.append(cluster_vecs(cs[t], 40, rng))
        fit_topics += [t] * 40
    fit_ids += [f"n{i}" for i in range(20)]
    fit_vecs.append(unit(rng.normal(size=(20, ta.DIM))))
    fit_topics += [-1] * 20
    root = tmp_path / "bucket"
    write_embeddings(root, "fit", fit_ids, np.vstack(fit_vecs))
    write_topics(root, "20260904T084845Z", fit_ids, fit_topics, FIT_PARAMS)
    new_ids, new_vecs = [], []
    for t in range(3):
        new_ids += [f"new{t}-{i}" for i in range(3)]
        new_vecs.append(cluster_vecs(cs[t], 3, rng))
    new_ids += [f"far{i}" for i in range(5)]
    new_vecs.append(far)
    write_embeddings(root, "20261010T000000Z", new_ids, np.vstack(new_vecs), day="2026-10-10")
    return root, new_ids


def run(root, tmp_path, **kw):
    kw.setdefault("run_id", "20261012T053000Z")
    kw.setdefault("device", "cpu")
    kw.setdefault("settings", SETTINGS)
    store = kw.pop("store", None) or DirStore(root)
    return ta.assign(store, tmp_path / "cache", **kw), store


def written(root, run_id="20261012T053000Z"):
    files = sorted((root / TOP).rglob(f"run={run_id}-c*.parquet"))
    return pa.concat_tables([pq.ParquetFile(f).read() for f in files]) if files else None


def test_model_constant_matches_semantic():
    assert ta.MODEL == SEMANTIC_MODEL


def test_keys():
    assert ta.chunk_key("20261012T053000Z", 3) == (
        "topics/platform=x/model=paraphrase-multilingual-mpnet-base-v2"
        "/dt=2026-10-12/run=20261012T053000Z-c0003.parquet"
    )
    assert ta.marker_key("20261012T053000Z").startswith("topics/_assign/platform=x/")
    assert ta.run_id_of("topics/x/dt=2026-09-04/run=20260904T084845Z.parquet") == "20260904T084845Z"


def test_params_roundtrip():
    s = ta.params_string(SETTINGS, "20260904T084845Z", 289000)
    parsed = ta.parse_params(s)
    assert s.startswith("method=knn,k=5,threshold=0.60,fit_run=20260904T084845Z,heldout_agreement=0.9712")
    assert parsed["method"] == "knn" and parsed["fit_run"] == "20260904T084845Z"
    assert parsed["n_reference"] == "289000"


def test_vote_weights_and_features():
    sims = np.array([[0.9, 0.8, 0.79, 0.78]])
    topics = np.array([[1, 2, 2, 2]])
    majority = ta.vote(sims, topics, 4, 0)
    assert majority["label"][0] == 2
    assert majority["share"][0] == pytest.approx(0.75)
    assert majority["top1"][0] == pytest.approx(0.9)
    assert majority["support"][0] == pytest.approx((0.8 + 0.79 + 0.78) / 3)
    sharp = ta.vote(sims, topics, 1, 1)
    assert sharp["label"][0] == 1
    # a sharp weight lets one very close neighbour outvote two distant ones
    s2 = np.array([[0.99, 0.30, 0.29]])
    t2 = np.array([[1, 2, 2]])
    assert ta.vote(s2, t2, 3, 0)["label"][0] == 2
    assert ta.vote(s2, t2, 3, 16)["label"][0] == 1


def test_decide_threshold_sends_weak_votes_to_noise():
    v = ta.vote(np.array([[0.9, 0.2], [0.5, 0.4]]), np.array([[1, 1], [2, 2]]), 2, 1)
    assert ta.decide(v, "top1", 0.6).tolist() == [1, -1]
    assert ta.decide(v, "top1", 0.5).tolist() == [1, 2]


def test_knn_search_matches_brute_force_with_tiles():
    import torch

    rng = np.random.default_rng(3)
    ref = unit(rng.normal(size=(300, 32))).astype(np.float32)
    q = unit(rng.normal(size=(20, 32))).astype(np.float32)
    sims, idx = ta.knn_search(q, torch.from_numpy(ref), 7, block=6, device="cpu", tile=64)
    full = q @ ref.T
    want = np.argsort(-full, axis=1)[:, :7]
    assert (idx == want).all()
    assert np.allclose(sims, np.take_along_axis(full, want, 1), atol=1e-5)


def test_fit_state_picks_newest_fit_and_ignores_stale_knn(tmp_path):
    root = tmp_path
    old = write_topics(root, "20260904T084845Z", ["a"], [0], FIT_PARAMS)
    new_fit = write_topics(
        root, "20261101T000000Z", ["a", "b"], [0, 1], FIT_PARAMS,
        at=FIT_AT + timedelta(days=58), day="2026-11-01",
    )
    stale = write_topics(
        root, "20261010T000000Z-c0000", ["c"], [0], "method=knn,k=5,fit_run=20260904T084845Z",
        at=FIT_AT + timedelta(days=36), day="2026-10-10",
    )
    fresh = write_topics(
        root, "20261105T000000Z-c0000", ["d"], [0], "method=knn,k=5,fit_run=20261101T000000Z",
        at=FIT_AT + timedelta(days=62), day="2026-11-05",
    )
    st = ta.fit_state([old, new_fit, stale, fresh])
    assert st.fit_run == "20261101T000000Z"
    assert st.fit_files == [new_fit]
    assert st.knn_files == [fresh]
    assert st.stale_knn_files == [stale]


def test_fit_state_without_a_fit_is_an_error(tmp_path):
    p = write_topics(tmp_path, "r-c0000", ["a"], [0], "method=knn,fit_run=x")
    with pytest.raises(RuntimeError, match="no fit run"):
        ta.fit_state([p])


def test_plan_rows_takes_newest_copy_and_respects_exclude(tmp_path):
    rng = np.random.default_rng(0)
    v = unit(rng.normal(size=(3, ta.DIM)))
    older = write_embeddings(tmp_path, "20260716T000000Z", ["a", "b"], v[:2], day="2026-07-16")
    newer = write_embeddings(tmp_path, "20261010T000000Z", ["b", "c"], v[1:], day="2026-10-10")
    plan = ta.plan_rows([older, newer], wanted=None, exclude=pa.array(["a"]))
    by_path = {p: r.tolist() for p, r in plan}
    assert by_path == {newer: [0, 1]}  # b and c come from the newer file, a is excluded
    got = {i: v for ids, vs in ta.read_planned(plan) for i, v in zip(ids, vs)}
    assert set(got) == {"b", "c"}
    assert np.allclose(got["b"], v[1], atol=1e-6)


def test_assign_end_to_end(bucket, tmp_path):
    root, new_ids = bucket
    marker, store = run(root, tmp_path)
    assert marker["status"] == "complete"
    assert marker["pending_at_start"] == len(new_ids) == marker["assigned"]
    assert marker["readback_rows"] == marker["readback_distinct"] == len(new_ids)
    out = written(root)
    assert out.schema.equals(ta._TOPICS_SCHEMA)
    got = dict(zip(out.column("platform_post_id").to_pylist(), out.column("topic").to_pylist()))
    for t in range(3):
        assert all(got[f"new{t}-{i}"] == t for i in range(3))
    assert all(got[f"far{i}"] == -1 for i in range(5))
    params = set(out.column("params").to_pylist())
    assert len(params) == 1 and next(iter(params)).startswith("method=knn,k=5,threshold=0.60,fit_run=20260904T084845Z")
    assert set(out.column("model").to_pylist()) == {"paraphrase-multilingual-mpnet-base-v2"}
    assert ta.marker_key("20261012T053000Z") in store.put_log


def test_fit_posts_are_never_reassigned_and_rerun_is_empty(bucket, tmp_path):
    root, new_ids = bucket
    run(root, tmp_path)
    marker, store = run(root, tmp_path, run_id="20261012T060000Z")
    assert marker["status"] == "complete" and marker["assigned"] == 0 and marker["pending_at_start"] == 0
    assert written(root, "20261012T060000Z") is None
    first = written(root)
    assert sorted(first.column("platform_post_id").to_pylist()) == sorted(new_ids)


def test_chunks_resume_after_partial_progress(bucket, tmp_path):
    root, new_ids = bucket
    # Only two chunk urls: the first attempt gets 2 chunks of 6, then reports
    # partial; the next attempt does the rest and nothing is duplicated.
    allowed = {ta.chunk_key("20261012T053000Z", i) for i in range(2)} | {ta.marker_key("20261012T053000Z")}
    marker, _ = run(root, tmp_path, chunk_rows=6, store=DirStore(root, allowed))
    assert marker["status"] == "partial" and marker["assigned"] == 12
    marker2, _ = run(root, tmp_path, run_id="20261012T060000Z", chunk_rows=6)
    assert marker2["status"] == "complete" and marker2["assigned"] == len(new_ids) - 12
    ids = written(root).column(0).to_pylist() + written(root, "20261012T060000Z").column(0).to_pylist()
    assert sorted(ids) == sorted(new_ids)


def test_limit_reports_partial(bucket, tmp_path):
    root, new_ids = bucket
    marker, _ = run(root, tmp_path, limit=4)
    assert marker["status"] == "partial" and marker["assigned"] == 4
    assert marker["pending_at_start"] == len(new_ids)


def test_dry_run_writes_nothing_remote(bucket, tmp_path):
    root, _ = bucket
    marker, store = run(root, tmp_path, dry_run=True)
    assert store.put_log == [] and written(root) is None
    assert marker["assigned"] > 0 and marker["dry_run"] is True


def test_knn_rows_against_an_older_fit_are_pending_again(bucket, tmp_path):
    root, new_ids = bucket
    write_topics(
        root, "20261010T000000Z-c0000", new_ids, [0] * len(new_ids),
        "method=knn,k=5,fit_run=19990101T000000Z", at=FIT_AT + timedelta(days=36), day="2026-10-10",
    )
    marker, _ = run(root, tmp_path)
    assert marker["pending_at_start"] == len(new_ids)
    assert marker["stale_knn_files_ignored"] == 1


def test_readers_see_knn_rows_and_fit_rows_together(bucket, tmp_path):
    """`latest_topics`-style newest-per-post over fit + knn, and the derived
    `assignment_method` column that `load_topics` adds."""
    root, new_ids = bucket
    run(root, tmp_path)
    posts = pa.table(
        {
            "platform": ["x"] * 3,
            "platform_post_id": ["f0-0", "new0-0", "far0"],
            "author_handle": ["a", "b", "c"],
            "text": ["t1", "t2", "t3"],
            "created_at": [FIT_AT] * 3,
            "lang": ["en"] * 3,
            "collected_at": [FIT_AT] * 3,
        }
    )
    pq.write_table(posts, tmp_path / "posts.parquet")
    glob = f"{root}/topics/platform=x/model=*/dt=*/run=*.parquet"
    sql = _load_topics_sql(
        f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)",
        f"read_parquet('{tmp_path}/posts.parquet')",
    )
    rows = duckdb.sql(sql).fetchall()
    by_id = {r[0]: r for r in rows}
    assert by_id["f0-0"][1] == 0 and by_id["f0-0"][6] == "fit"
    assert by_id["new0-0"][1] == 0 and by_id["new0-0"][6] == "knn"
    assert by_id["far0"][1] == -1 and by_id["far0"][6] == "knn"
    # the marker lives outside the readers' glob
    assert not list((root / "topics").glob("platform=*/**/*.json"))


def test_calibration_on_separable_clusters(tmp_path):
    rng = np.random.default_rng(5)
    cs = centers(4, seed=9)
    ids, vecs, topics = [], [], []
    for t in range(4):
        n = 60
        ids += [f"t{t}-{i}" for i in range(n)]
        vecs.append(cluster_vecs(cs[t], n, rng))
        topics += [t] * n
    ids += [f"n{i}" for i in range(40)]
    vecs.append(unit(rng.normal(size=(40, ta.DIM))))
    topics += [-1] * 40
    root = tmp_path / "b"
    write_embeddings(root, "all", ids, np.vstack(vecs))
    files = [root / EMB / "dt=2026-07-16" / "run=all.parquet"]
    fit = pa.table({"platform_post_id": ids, "topic": pa.array(topics, pa.int64())})
    res = ta.run_calibration(files, fit, device="cpu", fraction=0.2, seed=1)
    best = res["topics"]["best"]
    assert best["topic_recall"] > 0.95 and best["noise_recall"] > 0.9
    assert res["topics"]["n_heldout"] == 4 * 12 + 8
    assert res["fp16_vs_fp32_label_match"] == 1.0
    assert set(res["topics"]["settings"]) >= {"k", "power", "feature", "threshold", "heldout_agreement"}


def test_holdout_keeps_every_topic_represented(tmp_path):
    topic = np.array([0] * 3 + [1] * 100 + [-1] * 50)
    held = ta.holdout_split(pa.table({"topic": topic}), 0.05, seed=0)
    assert (~held[:3]).sum() >= 1
    assert held[3:103].sum() == 5 and held[103:].sum() == round(50 * 0.05)


def test_mirror_fetches_only_missing_and_drops_stale(tmp_path):
    root = tmp_path / "b"
    write_embeddings(root, "a", ["1"], unit(np.ones((1, ta.DIM))))
    store = DirStore(root)
    cache = tmp_path / "c"
    files = ta.mirror(store, EMB, cache)
    assert len(files) == 1
    marker = files[0].stat().st_mtime_ns
    ta.mirror(store, EMB, cache)
    assert files[0].stat().st_mtime_ns == marker
    stale = cache / EMB / "dt=2026-01-01" / "run=gone.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"x")
    ta.mirror(store, EMB, cache)
    assert not stale.exists()


class _Server:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.puts: dict[str, bytes] = {}
        self.fail_first = 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if outer.fail_first > 0:
                    outer.fail_first -= 1
                    self.send_response(503)
                    self.end_headers()
                    return
                body = outer.files.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_PUT(self):
                n = int(self.headers["Content-Length"])
                outer.puts[self.path] = self.rfile.read(n)
                self.send_response(200)
                self.end_headers()

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(ta.time, "sleep", lambda s: None)
    s = _Server()
    yield s
    s.close()


def test_manifest_store_get_put_and_retry(server, tmp_path):
    server.files["/obj"] = b"hello"
    server.fail_first = 2
    store = ta.ManifestStore(
        {
            "version": 1,
            "objects": {"embeddings/a": {"size": 5, "url": f"{server.url}/obj"}},
            "puts": {"topics/new": {"put": f"{server.url}/up", "get": f"{server.url}/up"}},
        }
    )
    assert [o.key for o in store.list("embeddings/")] == ["embeddings/a"]
    store.get("embeddings/a", tmp_path / "x" / "a")
    assert (tmp_path / "x" / "a").read_bytes() == b"hello"
    src = tmp_path / "up.bin"
    src.write_bytes(b"payload")
    store.put("topics/new", src)
    assert server.puts["/up"] == b"payload"
    with pytest.raises(KeyError):
        store.put("topics/other", src)


def test_manifest_store_rejects_wrong_size(server, tmp_path):
    server.files["/obj"] = b"hello"
    store = ta.ManifestStore(
        {"version": 1, "objects": {"k": {"size": 9, "url": f"{server.url}/obj"}}}, attempts=2
    )
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        store.get("k", tmp_path / "a")
    assert not (tmp_path / "a").exists()


def test_manifest_version_checked():
    with pytest.raises(ValueError):
        ta.ManifestStore({"version": 2, "objects": {}})


def test_cli_run_exit_codes(bucket, tmp_path):
    root, _ = bucket
    manifest = {"version": 1, "objects": {}, "puts": {}}
    for o in DirStore(root).list(""):
        manifest["objects"][o.key] = {"size": o.size, "url": (root / o.key).as_uri()}
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(manifest))
    cache = tmp_path / "cache"
    cache.mkdir()
    cal = {"fit_run": "20260904T084845Z", "chosen": SETTINGS.to_json()}
    (cache / "calibration-20260904T084845Z.json").write_text(json.dumps(cal))
    rc = ta.main(
        ["run", "--manifest", str(mpath), "--cache", str(cache), "--device", "cpu",
         "--run-id", "20261012T053000Z", "--dry-run", "--status-file", str(tmp_path / "s.json")]
    )
    assert rc == 0
    assert json.loads((tmp_path / "s.json").read_text())["stage"] == "done"


def test_assign_calibrates_once_when_no_settings(bucket, tmp_path):
    root, new_ids = bucket
    store = DirStore(root)
    marker = ta.assign(store, tmp_path / "cache", run_id="20261012T053000Z", device="cpu")
    assert marker["status"] == "complete" and marker["assigned"] == len(new_ids)
    cal = json.loads((tmp_path / "cache" / "calibration-20260904T084845Z.json").read_text())
    assert cal["reference"] in ("topics", "topics+noise")
    assert ta.parse_params(marker["params"])["fit_run"] == "20260904T084845Z"
    assert "heldout_agreement" in ta.parse_params(marker["params"])


def test_cli_calibrate_writes_settings(bucket, tmp_path):
    root, _ = bucket
    manifest = {"version": 1, "objects": {}, "puts": {}}
    for o in DirStore(root).list(""):
        manifest["objects"][o.key] = {"size": o.size, "url": (root / o.key).as_uri()}
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(manifest))
    cache = tmp_path / "cache"
    rc = ta.main(["calibrate", "--manifest", str(mpath), "--cache", str(cache), "--device", "cpu"])
    assert rc == 0
    cal = json.loads((cache / "calibration-20260904T084845Z.json").read_text())
    assert cal["chosen"]["k"] >= 1

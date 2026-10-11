"""The text trace stage and the cloud runner's `--text-trace` modes.

No R2, no Modal, no model. The window's posts and vectors are local parquet,
R2 locations are mapped to a temp directory through `uri`, and the corpus-
reading seams of the rank are replaced the way `test_v2_daily_run` does.
"""

from __future__ import annotations

import itertools
import logging
import subprocess
import sys
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest

from kma import (
    bench, coord2, coord2_communities, coord2_run, embed_window, text_trace, textsim_run,
    v2_daily, v2_daily_check, v2_daily_run, window,
)

DIM = 16
W = window.Window(date(2026, 9, 21), 3)
SENTENCE = "electoral commission rigged tallying centres nationwide again overnight"


def _vec(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=DIM)
    return v / np.linalg.norm(v)


@pytest.fixture
def corpus(tmp_path):
    """Three posts that say the same thing from three accounts, plus unrelated
    ones. Returns the manifest that pins them."""
    shared = _vec(1)
    rows = [
        ("a", "p1", "2026-09-20 10:00:00", SENTENCE, shared),
        ("b", "p2", "2026-09-20 10:05:00", SENTENCE, shared),
        ("c", "p3", "2026-09-20 10:09:00", SENTENCE, shared),
        ("d", "p4", "2026-09-20 11:00:00", "completely different words about weather today", _vec(2)),
        ("a", "p5", "2026-09-21 09:00:00", "another unrelated sentence about farming and rainfall", _vec(3)),
        ("b", "p6", "2026-09-21 09:30:00", "yet another sentence concerning roads and bridges built", _vec(4)),
    ]
    posts = pd.DataFrame({
        "author_id": [r[0] for r in rows],
        "platform_post_id": [r[1] for r in rows],
        "created_at": pd.to_datetime([r[2] for r in rows]),
        "is_repost": False,
        "repost_of_id": None,
        "text": [r[3] for r in rows],
        "collected_at": pd.to_datetime([r[2] for r in rows]),
        "hashtags": None,
        "urls": None,
    })
    emb = pd.DataFrame({
        "platform_post_id": [r[1] for r in rows],
        "embedding": [r[4].tolist() for r in rows],
        "embedded_at": pd.Timestamp("2026-09-22", tz="UTC"),
    })
    posts.to_parquet(tmp_path / "posts.parquet")
    emb.to_parquet(tmp_path / "emb.parquet")
    manifest = pd.DataFrame([
        {"prefix": "posts", "path": str(tmp_path / "posts.parquet"), "rows": len(posts)},
        {"prefix": "embeddings", "path": str(tmp_path / "emb.parquet"), "rows": len(emb)},
    ])
    return manifest


@pytest.fixture
def where(tmp_path):
    root = tmp_path / "r2"

    def uri(key: str) -> str:
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    return uri


def _edges(path) -> pd.DataFrame:
    return pd.read_parquet(path).sort_values(["source", "target"]).reset_index(drop=True)


def test_the_trace_lands_under_the_key_the_cloud_daily_looks_for(corpus, where):
    con = duckdb.connect()
    assert not v2_daily.text_trace_exists(con, W.id, uri=where)
    text_trace.build(con, W.id, manifest=corpus, uri=where)
    assert v2_daily.text_trace_exists(con, W.id, uri=where)


def test_the_modal_stage_and_the_cloud_runner_agree_on_the_trace_key(corpus, where):
    """The overlap is in the R2 key, so a stage that built at another floor than
    the cloud runner looks for would leave the daily ranking without text."""
    con = duckdb.connect()
    _, info = text_trace.build(con, W.id, manifest=corpus, uri=where)
    cloud_key = coord2_run.textsim_key(
        W.id, v2_daily.TEXT_THRESHOLD, overlap=coord2.TEXT_MIN_OVERLAP)
    assert "/overlap=0.45/" in cloud_key
    assert info["r2_key"] == coord2_run.textsim_key(
        W.id, v2_daily.TEXT_THRESHOLD, overlap=coord2.TEXT_MIN_OVERLAP, bucket="day")
    assert v2_daily.text_trace_exists(con, W.id, uri=where)


def test_the_pair_level_object_is_the_day_trace_aggregated(corpus, where):
    con = duckdb.connect()
    bucketed, info = text_trace.build(con, W.id, manifest=corpus, uri=where)
    plain = _edges(where(coord2_run.textsim_key(W.id, 0.85, overlap=coord2.TEXT_MIN_OVERLAP)))
    day = _edges(where(coord2_run.textsim_key(W.id, 0.85, overlap=coord2.TEXT_MIN_OVERLAP, bucket="day")))
    assert info["threshold"] == v2_daily.TEXT_THRESHOLD
    assert info["overlap"] == coord2.TEXT_MIN_OVERLAP
    assert set(zip(plain["source"], plain["target"])) == {("a", "b"), ("a", "c"), ("b", "c")}
    assert "bucket" in day.columns and len(day) == len(bucketed)
    pd.testing.assert_frame_equal(
        plain,
        day.groupby(["source", "target"], as_index=False).agg(weight=("weight", "mean")),
    )


def test_the_stage_and_the_local_build_are_one_code_path(corpus, where, monkeypatch):
    """What `v2_daily_run --text-trace local` builds and what the Modal stage
    builds come from the same call, so the cut cannot drift between them."""
    seen = []
    real = textsim_run.build
    monkeypatch.setattr(textsim_run, "build", lambda *a, **kw: seen.append(kw) or real(*a, **kw))
    con = duckdb.connect()
    text_trace.build(con, W.id, manifest=corpus, uri=where)
    (kw,) = seen
    assert kw["threshold"] == v2_daily.TEXT_THRESHOLD
    assert kw["overlap"] == -1.0
    assert kw["time_bucket"] == "day"
    assert kw["similarity_for"] is coord2.gpu_cosine_pairs


@pytest.fixture
def stage(corpus, where, monkeypatch):
    """`text_trace.run` over the local corpus, with scratch mapped to a temp dir."""
    monkeypatch.setattr(text_trace, "scratch_uri", lambda prefix: where)
    monkeypatch.setattr(window, "build", lambda w, **kw: corpus)
    log = {"embed": []}

    def embed(con, w, **kw):
        log["embed"].append(kw)
        return {"window_id": w.id, "embedded": log.get("embedded", 0), "runs": []}

    monkeypatch.setattr(embed_window, "embed_missing", embed)
    return log


def test_a_scratch_run_builds_once_and_then_reuses(stage, corpus, where):
    con = duckdb.connect()
    first = text_trace.run("2026-09-21", 3, con=con, scratch="s")
    assert first["textsim"] == text_trace.BUILT
    assert first["trace"]["edges"] > 0
    second = text_trace.run("2026-09-21", 3, con=con, scratch="s")
    assert second["textsim"] == text_trace.REUSED


def test_force_rebuilds_a_present_trace(stage):
    con = duckdb.connect()
    text_trace.run("2026-09-21", 3, con=con, scratch="s")
    assert text_trace.run("2026-09-21", 3, con=con, scratch="s", force=True)["textsim"] == text_trace.BUILT


def test_a_scratch_run_writes_the_manifest_it_read_under_the_prefix(stage, corpus, where):
    con = duckdb.connect()
    text_trace.run("2026-09-21", 3, con=con, scratch="s")
    pinned = bench.load(W.id, con=con, uri=where)
    assert list(pinned["path"]) == list(corpus["path"])


def test_scratch_embeds_go_to_the_prefix_not_to_production(stage):
    text_trace.run("2026-09-21", 3, con=duckdb.connect(), scratch="s")
    (kw,) = stage["embed"]
    assert kw["uri"] is not None


def test_an_empty_scratch_prefix_is_refused():
    with pytest.raises(ValueError, match="production"):
        text_trace.scratch_uri("/")


def test_embed_false_loads_nothing(stage):
    text_trace.run("2026-09-21", 3, con=duckdb.connect(), scratch="s", embed=False)
    assert stage["embed"] == []


def test_a_window_with_no_posts_is_refused(stage, monkeypatch):
    monkeypatch.setattr(window, "build", lambda w, **kw: pd.DataFrame([{"prefix": "embeddings"}]))
    with pytest.raises(RuntimeError, match="no posts"):
        text_trace.run("2026-09-21", 3, con=duckdb.connect(), scratch="s")


@pytest.fixture
def production(corpus, monkeypatch):
    """Production mode: the manifest and trace probes are in-memory fakes."""
    state = {"manifest": None, "built": [], "trace": False, "embedded": 0}

    def load(name, **kw):
        if state["manifest"] is None:
            raise duckdb.IOException("no manifest")
        return state["manifest"]

    def build(w, **kw):
        state["built"].append(kw.get("write"))
        state["manifest"] = corpus
        return corpus

    def ensure(w, **kw):
        return state["manifest"] if state["manifest"] is not None else build(w, write=True)

    def trace(con, wid, **kw):
        state["trace"] = True
        return pd.DataFrame(), {"edges": 0}

    monkeypatch.setattr(bench, "load", load)
    monkeypatch.setattr(window, "build", build)
    monkeypatch.setattr(window, "ensure", ensure)
    monkeypatch.setattr(v2_daily, "text_trace_exists", lambda con, sid, **kw: state["trace"])
    monkeypatch.setattr(text_trace, "build", trace)
    monkeypatch.setattr(
        embed_window, "embed_missing",
        lambda con, w, **kw: {"window_id": w.id, "embedded": state["embedded"], "runs": []},
    )
    return state


def test_a_covered_window_with_a_trace_does_no_work(production):
    production["manifest"] = pd.DataFrame([{"prefix": "posts", "rows": 1}])
    production["trace"] = True
    out = text_trace.run("2026-09-21", 3, con=duckdb.connect())
    assert out["textsim"] == text_trace.REUSED
    assert production["built"] == []
    assert out["manifest_rebuilt"] is False


def test_vectors_written_after_the_manifest_rebuild_it_and_the_trace(production):
    """A manifest pinned before the embed stage cannot see its vectors; reusing
    it, or a trace built from it, would be a quiet low-coverage result."""
    production["manifest"] = pd.DataFrame([{"prefix": "posts", "rows": 1}])
    production["trace"] = True
    production["embedded"] = 40
    out = text_trace.run("2026-09-21", 3, con=duckdb.connect())
    assert out["manifest_rebuilt"] is True
    assert production["built"] == [True]
    assert out["textsim"] == text_trace.BUILT


def test_a_first_run_pins_the_manifest_once(production):
    production["embedded"] = 40
    out = text_trace.run("2026-09-21", 3, con=duckdb.connect())
    assert production["built"] == [True]
    assert out["manifest_rebuilt"] is False


# ---- the cloud runner's modes ----------------------------------------------------


def _clique(prefix: str, n: int) -> list[tuple[str, str]]:
    return list(itertools.combinations([f"{prefix}{i}" for i in range(n)], 2))


@pytest.fixture
def runner(monkeypatch):
    calls = {"build": 0, "embed": 0, "snapshot": [], "persist": []}

    def build(w, **kw):
        return pd.DataFrame([{"prefix": "posts", "rows": 5}])

    def trace(con, wid, **kw):
        calls["build"] += 1
        edges = pd.DataFrame({"source": ["u0"], "target": ["u1"], "weight": [0.9],
                              "bucket": ["2026-09-20"]})
        return edges, {"snapshot": wid, "threshold": 0.85, "overlap": 0.1, "edges": 1}

    def networks(con, snapshot, **kw):
        calls["snapshot"].append(kw)
        nets = {"co_retweet": pd.DataFrame(_clique("k", 8) + _clique("o", 8),
                                           columns=["source", "target"])}
        if kw.get("text_threshold") is not None:
            nets["text_similarity"] = pd.DataFrame(_clique("u", 6), columns=["source", "target"])
        sizes = pd.DataFrame([{"trace": t, "edges": len(e)} for t, e in nets.items()])
        return nets, sizes, "SELECT 1"

    def embed(con, w, **kw):
        calls["embed"] += 1
        return {"window_id": w.id, "embedded": 0, "runs": []}

    monkeypatch.setattr(window, "build", build)
    monkeypatch.setattr(window, "ensure", lambda w, **kw: build(w))
    monkeypatch.setattr(textsim_run, "build", trace)
    monkeypatch.setattr(embed_window, "embed_missing", embed)
    monkeypatch.setattr(coord2_run, "snapshot_networks", networks)
    monkeypatch.setattr(
        coord2_communities, "model_kenya_share",
        lambda con, view, ids, **kw: pd.DataFrame(
            {"user_id": ids, "posts": 10, "scored": 10, "kenya_model": 1.0}).set_index("user_id"),
    )
    monkeypatch.setattr(coord2_run, "attach_relevance",
                        lambda con, scores, view, **kw: scores.assign(n_posts=10, kenya_share=1.0))
    return calls


def _go(tmp_path, mode, *, present, **kw):
    import unittest.mock as mock

    with mock.patch.object(v2_daily, "text_trace_exists", lambda con, sid, **k: present):
        return v2_daily_run.run(
            "2026-09-21", 7, con=duckdb.connect(), out_dir=tmp_path / "out",
            textfile_dir=tmp_path / "textfile", text_trace_mode=mode, **kw,
        )


def _row(tmp_path, kind: str) -> pd.DataFrame:
    (path,) = (tmp_path / "out/coord2/platform=x" / f"kind={kind}").glob("dt=*/run=*.parquet")
    return pd.read_parquet(path)


def _prom(tmp_path) -> dict[str, float]:
    text = (tmp_path / "textfile/kma_v2_daily.prom").read_text()
    return {line.split()[0]: float(line.split()[1]) for line in text.splitlines()
            if line and not line.startswith("#")}


def test_modal_mode_reuses_the_persisted_trace(tmp_path, runner):
    out = _go(tmp_path, "modal", present=True)
    assert out["textsim"] == "reused"
    assert out["text_trace"] == text_trace.REUSED
    assert runner["build"] == 0 and runner["embed"] == 0
    assert runner["snapshot"][-1]["text_threshold"] == v2_daily.TEXT_THRESHOLD
    assert _prom(tmp_path)["kma_v2_daily_text_trace_present"] == 1
    assert _row(tmp_path, "daily_runs")["text_trace"].iloc[0] == "reused"


def test_modal_mode_never_builds_locally_when_the_trace_is_missing(tmp_path, runner, caplog):
    with caplog.at_level(logging.WARNING):
        out = _go(tmp_path, "modal", present=False, embed=True)
    assert runner["build"] == 0 and runner["embed"] == 0
    assert out["text_trace"] == text_trace.MISSING
    assert out["textsim"]["status"] == "missing"
    assert out["textsim"]["expected_key"] == coord2_run.textsim_key(
        "daily-2026-09-21-7d", 0.85, overlap=coord2.TEXT_MIN_OVERLAP
    )
    assert any("TEXT TRACE MISSING" in r.message and r.levelno == logging.WARNING
               for r in caplog.records)


def test_a_missing_trace_still_ranks_and_says_so_everywhere(tmp_path, runner):
    out = _go(tmp_path, "modal", present=False)
    assert runner["snapshot"][-1]["text_threshold"] is None
    assert out["rank"]["listed_accounts"] > 0
    prom = _prom(tmp_path)
    assert prom["kma_v2_daily_text_trace_present"] == 0
    assert prom["kma_v2_daily_last_success_timestamp_seconds"] > 0
    row = _row(tmp_path, "daily_runs").iloc[0]
    assert row["text_trace"] == "missing" and row["text_trace_mode"] == "modal"
    assert pd.isna(row["text_threshold"])
    assert "edges_text_similarity" not in row.index
    scores = _row(tmp_path, "scores")
    assert scores["text_threshold"].isna().all()


def test_a_missing_trace_run_reads_back_beside_a_normal_one(tmp_path, runner):
    """`kind=daily_runs` and `kind=scores` are read with union_by_name across
    days; a null threshold column must not break the read."""
    _go(tmp_path / "a", "modal", present=True)
    _go(tmp_path / "b", "modal", present=False)
    frames = list((tmp_path).glob("*/out/coord2/platform=x/kind=scores/dt=*/run=*.parquet"))
    assert len(frames) == 2
    glob = str(tmp_path / "*/out/coord2/platform=x/kind=scores/dt=*/run=*.parquet")
    got = duckdb.connect().sql(
        f"SELECT text_threshold FROM read_parquet('{glob}', union_by_name=true)"
    ).df()
    assert got["text_threshold"].notna().any() and got["text_threshold"].isna().any()


def test_none_mode_skips_text_even_when_a_trace_exists(tmp_path, runner):
    out = _go(tmp_path, "none", present=True)
    assert out["text_trace"] == text_trace.SKIPPED
    assert runner["build"] == 0 and runner["embed"] == 0
    assert runner["snapshot"][-1]["text_threshold"] is None
    assert _row(tmp_path, "daily_runs")["text_trace"].iloc[0] == "skipped"
    assert _prom(tmp_path)["kma_v2_daily_text_trace_present"] == 0


def test_local_mode_builds_on_this_host(tmp_path, runner):
    out = _go(tmp_path, "local", present=True)
    assert out["text_trace"] == text_trace.BUILT
    assert runner["build"] == 1
    assert _prom(tmp_path)["kma_v2_daily_text_trace_present"] == 1


def test_local_mode_reuses_when_not_in_scratch(tmp_path, runner, monkeypatch):
    monkeypatch.setattr(coord2_run, "persist", lambda con, scores, **kw: "k")
    monkeypatch.setattr(v2_daily, "_write", lambda con, frame, key, uri: key)
    import unittest.mock as mock

    with mock.patch.object(v2_daily, "text_trace_exists", lambda con, sid, **k: True):
        out = v2_daily_run.run("2026-09-21", 7, con=duckdb.connect(), textfile_dir=tmp_path,
                               text_trace_mode="local", embed=False)
    assert out["text_trace"] == text_trace.REUSED and runner["build"] == 0


def test_an_unknown_mode_is_refused(tmp_path, runner):
    with pytest.raises(ValueError, match="text_trace_mode"):
        _go(tmp_path, "gpu", present=True)


def test_the_cli_default_is_modal(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        v2_daily_run, "run", lambda *a, **kw: seen.update(kw) or {"ok": True},
    )
    import kma.db

    monkeypatch.setattr(kma.db, "connect", lambda: duckdb.connect())
    v2_daily_run.main(["--textfile-dir", str(tmp_path)])
    assert seen["text_trace_mode"] == "modal"
    v2_daily_run.main(["--textfile-dir", str(tmp_path), "--text-trace", "local"])
    assert seen["text_trace_mode"] == "local"


def test_the_gauge_keeps_its_first_lines_and_adds_one_sample(tmp_path):
    path = v2_daily_run.write_gauge(tmp_path, now=1_000, text_trace_present=False)
    lines = path.read_text().splitlines()
    assert lines[:2] == [
        "# TYPE kma_v2_daily_last_success_timestamp_seconds gauge",
        "kma_v2_daily_last_success_timestamp_seconds 1000",
    ]
    assert lines[2:] == [
        "# TYPE kma_v2_daily_text_trace_present gauge",
        "kma_v2_daily_text_trace_present 0",
    ]
    assert v2_daily_check.read_gauge(tmp_path)[0] == 1000.0


def test_the_modal_free_import_path_holds():
    code = (
        "import sys, kma.text_trace, kma.v2_daily_run, kma.v2_daily_check;"
        "sys.exit(1 if 'modal' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0

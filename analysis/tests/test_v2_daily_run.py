"""The CPU runner: stage order, what `kind=scores` carries, floors, canary, gauge.

No R2 and no Modal. The corpus-reading seams (`window.build`, `textsim_run.build`,
`coord2_run.snapshot_networks`, the relevance reads) are replaced; ranking,
community reporting, persistence and the gauge are the real code.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys

import duckdb
import pandas as pd
import pytest

from kma import canary, coord2, coord2_communities, coord2_run, textsim_run, v2_daily, v2_daily_run
from kma import window, windowed

GAUGE = "kma_v2_daily_last_success_timestamp_seconds"


def _clique(prefix: str, n: int) -> list[tuple[str, str]]:
    return list(itertools.combinations([f"{prefix}{i}" for i in range(n)], 2))


def _networks() -> dict[str, pd.DataFrame]:
    return {
        "co_retweet": pd.DataFrame(_clique("k", 8) + _clique("o", 8), columns=["source", "target"]),
        "text_similarity": pd.DataFrame(_clique("u", 6), columns=["source", "target"]),
    }


@pytest.fixture
def stages(monkeypatch):
    """Replace everything that would read the corpus. Returns the call log."""
    calls: dict[str, list] = {"build": [], "snapshot": [], "order": []}

    def build(w, **kw):
        calls["order"].append("manifest")
        return pd.DataFrame([{"prefix": "posts", "rows": 5}])

    def trace(con, snapshot, **kw):
        calls["order"].append("textsim")
        calls["build"].append(kw)
        edges = pd.DataFrame({
            "source": ["u0", "u1"], "target": ["u1", "u2"], "weight": [0.9, 0.9],
            "bucket": ["2026-09-20", "2026-09-21"],
        })
        return edges, {"snapshot": snapshot, "threshold": 0.85, "overlap": 0.1, "edges": 2}

    def networks(con, snapshot, **kw):
        calls["order"].append("rank")
        calls["snapshot"].append(kw)
        nets = _networks()
        sizes = pd.DataFrame([{"trace": t, "edges": len(e)} for t, e in nets.items()])
        return nets, sizes, "SELECT 1"

    def kenya(con, view, ids, **kw):
        return pd.DataFrame({"user_id": ids, "posts": 10, "scored": 10, "kenya_model": 1.0}
                            ).set_index("user_id")

    def relevance(con, scores, view, **kw):
        return scores.assign(n_posts=10, kenya_share=1.0)

    monkeypatch.setattr(window, "build", build)
    monkeypatch.setattr(textsim_run, "build", trace)
    monkeypatch.setattr(coord2_run, "snapshot_networks", networks)
    monkeypatch.setattr(coord2_communities, "model_kenya_share", kenya)
    monkeypatch.setattr(coord2_run, "attach_relevance", relevance)
    return calls


def _run(tmp_path, **kw):
    return v2_daily_run.run(
        "2026-09-21", 7, con=duckdb.connect(), out_dir=tmp_path / "out",
        textfile_dir=kw.pop("textfile_dir", tmp_path / "textfile"), **kw,
    )


def _scores(tmp_path) -> pd.DataFrame:
    (path,) = (tmp_path / "out/coord2/platform=x/kind=scores").glob("dt=*/run=*.parquet")
    return pd.read_parquet(path)


def test_stages_run_in_order_and_the_gauge_comes_last(tmp_path, stages):
    out = _run(tmp_path)
    assert stages["order"] == ["manifest", "textsim", "rank"]
    assert out["window_id"] == "daily-2026-09-21-7d"
    assert (tmp_path / "textfile/kma_v2_daily.prom").exists()


def test_the_text_trace_is_the_bucketed_day_trace_at_the_production_cut(tmp_path, stages):
    _run(tmp_path)
    (kw,) = stages["build"]
    assert kw["time_bucket"] == "day"
    assert kw["threshold"] == v2_daily.TEXT_THRESHOLD
    assert kw["similarity_for"] is coord2.gpu_cosine_pairs
    assert kw["persist"] is False


def test_scores_carry_community_and_what_the_collector_reads(tmp_path, stages):
    _run(tmp_path)
    scores = _scores(tmp_path)
    assert {"user_id", "centrality", "kenya_share", "community", "computed_at"} <= set(scores)
    assert scores["community"].notna().all()
    assert scores["community"].nunique() == 3
    assert set(scores["snapshot"]) == {"daily-2026-09-21-7d"}
    assert set(scores["window_since"]) == {"2026-09-15"}
    assert set(scores["window_until"]) == {"2026-09-22"}
    assert set(scores["min_entities"]) == {coord2.MIN_ENTITIES_PER_USER}
    assert scores["edges_co_retweet"].iloc[0] == len(_networks()["co_retweet"])


def test_scores_are_the_listed_accounts(tmp_path, stages):
    _run(tmp_path)
    accounts = pd.read_parquet(next((tmp_path / "out").glob("coord2/platform=x/kind=daily_accounts/*/*.parquet")))
    assert set(_scores(tmp_path)["user_id"]) == set(accounts["user_id"].astype(str))


def test_the_rolling_rank_floor_defaults_to_the_coord2_value(tmp_path, stages):
    _run(tmp_path)
    assert stages["snapshot"][-1]["min_entities"] == coord2.MIN_ENTITIES_PER_USER


def test_the_rolling_rank_floor_is_a_flag(tmp_path, stages):
    _run(tmp_path, min_entities=3)
    assert stages["snapshot"][-1]["min_entities"] == 3
    assert set(_scores(tmp_path)["min_entities"]) == {3}


@pytest.mark.parametrize("spec", [windowed.DETECTION, windowed.SERIES])
def test_the_windowed_pass_uses_the_floor_windowed_decided(monkeypatch, spec):
    seen = {}
    monkeypatch.setattr(windowed, "timed_traces", lambda con, view: {})
    monkeypatch.setattr(
        windowed, "run_rows",
        lambda rows, *, width, min_entities: seen.update(width=width, floor=min_entities)
        or windowed.WindowedResult(pd.DataFrame({"user_id": ["a"]}), pd.DataFrame(), pd.DataFrame()),
    )
    ctx = v2_daily.PassContext(con=None, window=None, view="v", networks={}, graph=None,
                               centrality=None, run_id="r")
    tables = v2_daily_run.windowed_pass(spec)(ctx)
    assert seen == {"width": spec.width, "floor": spec.min_entities}
    assert list(tables) == [f"windowed_{spec.width}"]


def test_the_windowed_flag_lands_beside_the_daily_tables(tmp_path, stages, monkeypatch):
    monkeypatch.setattr(windowed, "timed_traces", lambda con, view: {})
    monkeypatch.setattr(
        windowed, "run_rows",
        lambda rows, *, width, min_entities: windowed.WindowedResult(
            pd.DataFrame({"user_id": ["k0"], "window": ["2026-09-20"]}), pd.DataFrame(), pd.DataFrame()),
    )
    _run(tmp_path, windowed_width="week")
    found = list((tmp_path / "out").glob("coord2/platform=x/kind=daily_windowed_week/*/*.parquet"))
    assert len(found) == 1
    assert set(pd.read_parquet(found[0])["min_entities"]) == {windowed.SERIES.min_entities}


def test_the_canary_reaches_rank_and_stays_out_of_scores(tmp_path, stages, monkeypatch):
    planted = pd.DataFrame({c: ["x"] for c in coord2.VIEW_COLUMNS})
    monkeypatch.setattr(canary, "load_active", lambda con, **kw: planted)
    ids = [canary.account_id("2026-09-21", n) for n in range(6)]
    monkeypatch.setattr(
        canary, "text_edges",
        lambda posts, **kw: pd.DataFrame(
            [(a, b, 0.95) for a, b in itertools.combinations(ids, 2)],
            columns=["source", "target", "weight"]),
    )
    seen = {}
    real = v2_daily.rank
    monkeypatch.setattr(
        v2_daily, "rank", lambda *a, **kw: seen.update(canary=kw["canary_posts"]) or real(*a, **kw)
    )
    _run(tmp_path, with_canary=True)
    assert seen["canary"] is planted
    members = pd.read_parquet(next((tmp_path / "out").glob("coord2/platform=x/kind=daily_members/*/*.parquet")))
    assert members["user_id"].map(canary.is_canary).any()
    assert not _scores(tmp_path)["user_id"].map(canary.is_canary).any()


def test_no_canary_flag_means_the_planted_posts_are_never_read(tmp_path, stages, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("canary.load_active called without --canary")

    monkeypatch.setattr(canary, "load_active", boom)
    _run(tmp_path)


def test_a_failed_stage_leaves_the_previous_gauge_alone(tmp_path, stages, monkeypatch):
    textfile = tmp_path / "textfile"
    previous = v2_daily_run.write_gauge(textfile, now=1_000)
    monkeypatch.setattr(coord2_run, "persist", lambda *a, **kw: (_ for _ in ()).throw(OSError("r2")))
    with pytest.raises(OSError):
        _run(tmp_path)
    assert previous.read_text() == v2_daily_run.gauge_text(1_000)


def test_an_empty_listing_is_a_failure_not_a_healthy_run(tmp_path, stages, monkeypatch):
    monkeypatch.setattr(coord2_run, "snapshot_networks", lambda *a, **kw: (
        {"co_retweet": pd.DataFrame({"source": [], "target": [], "weight": []})},
        pd.DataFrame([{"trace": "co_retweet", "edges": 0}]), "SELECT 1"))
    with pytest.raises(RuntimeError, match="no accounts"):
        _run(tmp_path)
    assert not (tmp_path / "textfile/kma_v2_daily.prom").exists()


def test_an_empty_window_is_refused_before_anything_is_ranked(tmp_path, stages, monkeypatch):
    monkeypatch.setattr(window, "build", lambda w, **kw: pd.DataFrame([{"prefix": "embeddings", "rows": 1}]))
    with pytest.raises(RuntimeError, match="no posts"):
        _run(tmp_path)
    assert not (tmp_path / "textfile").exists()


def test_the_gauge_is_exactly_the_contract(tmp_path):
    path = v2_daily_run.write_gauge(tmp_path / "textfile", now=1_791_603_006.7)
    assert path.name == "kma_v2_daily.prom"
    assert path.read_text() == (
        f"# TYPE {GAUGE} gauge\n{GAUGE} 1791603006\n"
    )
    assert oct(path.stat().st_mode & 0o777) == "0o644"


def test_the_gauge_replaces_atomically_and_leaves_no_temp_file(tmp_path):
    textfile = tmp_path / "textfile"
    v2_daily_run.write_gauge(textfile, now=1)
    v2_daily_run.write_gauge(textfile, now=2)
    assert [p.name for p in textfile.iterdir()] == ["kma_v2_daily.prom"]
    assert (textfile / "kma_v2_daily.prom").read_text().endswith(f"{GAUGE} 2\n")


def test_a_failed_gauge_write_keeps_the_old_file_and_cleans_up(tmp_path, monkeypatch):
    textfile = tmp_path / "textfile"
    v2_daily_run.write_gauge(textfile, now=1)

    def fail(src, dst):
        raise OSError("disk")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        v2_daily_run.write_gauge(textfile, now=2)
    assert [p.name for p in textfile.iterdir()] == ["kma_v2_daily.prom"]
    assert (textfile / "kma_v2_daily.prom").read_text() == v2_daily_run.gauge_text(1)


def test_a_scheduled_run_needs_somewhere_to_report(monkeypatch):
    monkeypatch.delenv(v2_daily_run.TEXTFILE_DIR_ENV, raising=False)
    with pytest.raises(SystemExit):
        v2_daily_run.main(["--days", "7"])


def test_the_runner_imports_without_modal():
    code = "import sys, kma.v2_daily_run; sys.exit('modal' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_scores_frame_stringifies_ids_and_drops_canary_accounts():
    frame = pd.DataFrame({
        "user_id": [101, "canary:2026-09-21:0", "7"], "community": [0, 1, 2], "centrality": [1.0, 2.0, 3.0],
    })
    out = v2_daily_run.scores_frame(frame)
    assert list(out["user_id"]) == ["101", "7"]
    assert list(out["community"]) == [0, 2]



def test_r2_mode_uses_the_production_writers_and_reuses_an_existing_trace(tmp_path, stages, monkeypatch):
    written = []
    monkeypatch.setattr(window, "ensure", lambda w, **kw: pd.DataFrame([{"prefix": "posts", "rows": 5}]))
    monkeypatch.setattr(window, "build", lambda *a, **kw: pytest.fail("manifest built, not ensured"))
    monkeypatch.setattr(v2_daily, "text_trace_exists", lambda con, snapshot, **kw: True)
    monkeypatch.setattr(v2_daily, "_write", lambda con, frame, key, uri: written.append(key) or key)
    monkeypatch.setattr(coord2_run, "persist", lambda con, scores, **kw: written.append("scores") or "k")

    out = v2_daily_run.run("2026-09-21", 7, con=duckdb.connect(), textfile_dir=tmp_path)
    assert out["textsim"] == "reused"
    assert stages["build"] == []
    assert stages["snapshot"][-1].get("text_edges") is None
    assert "scores" in written and any("kind=daily_accounts" in k for k in written)

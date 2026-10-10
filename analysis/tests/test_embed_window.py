"""The embedding stage: which posts it picks, the rows it writes, when it loads a model.

No R2 and no model. The posts and the existing vectors are tables on one
DuckDB connection, the encoder is a fake, and runs land under a temp directory.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import duckdb
import numpy as np
import pandas as pd
import pytest

from kma import embed_window, semantic, v2_daily, window

WORDS = "electoral commission rigged tallying centres nationwide again"


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute(
        "CREATE TABLE _posts (author_id VARCHAR, platform_post_id VARCHAR, created_at TIMESTAMP,"
        " is_repost BOOLEAN, repost_of_id VARCHAR, text VARCHAR, collected_at TIMESTAMP,"
        " hashtags VARCHAR, urls VARCHAR)"
    )
    rows = [
        ("u1", "fresh", "2026-09-20 10:00:00", False, None, f"fresh {WORDS}"),
        ("u2", "embedded", "2026-09-20 11:00:00", False, None, f"embedded {WORDS}"),
        ("u3", "retweet", "2026-09-20 12:00:00", True, "fresh", f"RT {WORDS}"),
        ("u4", "short", "2026-09-20 13:00:00", False, None, "too short"),
        ("u5", "second", "2026-09-21 09:00:00", False, None, f"second {WORDS}"),
    ]
    c.executemany(
        "INSERT INTO _posts VALUES (?, ?, ?::TIMESTAMP, ?, ?, ?, ?::TIMESTAMP, NULL, NULL)",
        [(a, p, t, rt, of, text, t) for a, p, t, rt, of, text in rows],
    )
    c.execute("CREATE TABLE _emb (platform_post_id VARCHAR)")
    c.execute("INSERT INTO _emb VALUES ('embedded')")
    return c


class FakeEncoder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        out = np.zeros((len(texts), semantic.DIM), dtype=np.float32)
        out[np.arange(len(texts)), np.arange(len(texts)) % semantic.DIM] = 1.0
        return out


@pytest.fixture
def encoder(monkeypatch):
    enc = FakeEncoder()
    monkeypatch.setattr(semantic, "_model", lambda model: enc)
    return enc


@pytest.fixture
def clock(monkeypatch):
    """One second per call, and no real sleeping."""
    state = {"now": datetime(2026, 10, 11, 3, 30, tzinfo=timezone.utc)}

    def tick():
        state["now"] += timedelta(seconds=1)
        return state["now"]

    monkeypatch.setattr(semantic, "_utcnow", tick)
    monkeypatch.setattr(semantic.time, "sleep", lambda s: None)


W = window.Window(date(2026, 9, 21), 7)


def _missing(con):
    return embed_window.missing(con, W, posts="_posts", embeddings="_emb")


def test_only_eligible_posts_without_a_vector_are_missing(con):
    assert sorted(_missing(con)["post_id"]) == ["fresh", "second"]


def test_the_definition_is_text_rows_not_a_copy_of_it(con):
    from kma import coord2

    view = coord2.posts_view("_posts")
    eligible = set(coord2.text_rows(con, view)["post_id"])
    assert eligible == {"fresh", "embedded", "second"}
    assert set(_missing(con)["post_id"]) == eligible - {"embedded"}


def test_rows_are_written_in_the_production_format(con, encoder, clock, tmp_path):
    keys = semantic.embed_texts(
        con, ["a", "b"], ["x y z w", "p q r s"], uri=lambda key: str(_local(tmp_path, key)),
    )
    (key,) = keys
    assert key.startswith("embeddings/platform=x/model=paraphrase-multilingual-mpnet-base-v2/dt=2026-10-11/run=")
    frame = pd.read_parquet(_local(tmp_path, key))
    assert list(frame.columns) == ["platform_post_id", "model", "dim", "embedding", "embedded_at"]
    assert frame["platform_post_id"].tolist() == ["a", "b"]
    assert set(frame["model"]) == {"paraphrase-multilingual-mpnet-base-v2"}
    assert set(frame["dim"]) == {semantic.DIM}
    assert len(frame["embedding"].iloc[0]) == semantic.DIM
    schema = con.sql(f"DESCRIBE SELECT embedding FROM read_parquet('{_local(tmp_path, key)}')").fetchone()
    assert schema[1] == "DOUBLE[]"


def test_each_chunk_is_its_own_run_with_its_own_key(con, encoder, clock, tmp_path):
    ids = [str(i) for i in range(5)]
    keys = semantic.embed_texts(
        con, ids, [f"t{i} a b c" for i in ids], chunk=2, uri=lambda key: str(_local(tmp_path, key)),
    )
    assert len(keys) == 3 and len(set(keys)) == 3
    assert [len(c) for c in encoder.calls] == [2, 2, 1]
    got = pd.concat(pd.read_parquet(_local(tmp_path, k)) for k in keys)
    assert got["platform_post_id"].tolist() == ids


def test_a_chunk_in_the_same_second_waits_rather_than_overwriting(con, encoder, monkeypatch, tmp_path):
    base = datetime(2026, 10, 11, 3, 30, tzinfo=timezone.utc)
    stamps = iter([base, base, base + timedelta(seconds=1), base + timedelta(seconds=1),
                   base + timedelta(seconds=2)])
    slept = []
    monkeypatch.setattr(semantic, "_utcnow", lambda: next(stamps))
    monkeypatch.setattr(semantic.time, "sleep", lambda s: slept.append(s))
    keys = semantic.embed_texts(
        con, ["a", "b"], ["x y z w", "p q r s"], chunk=1, uri=lambda key: str(_local(tmp_path, key)),
    )
    assert len(set(keys)) == 2 and slept


def test_a_covered_window_loads_no_model(con, monkeypatch, tmp_path):
    con.execute("INSERT INTO _emb VALUES ('fresh'), ('second')")
    monkeypatch.setattr(semantic, "_model", lambda m: pytest.fail("model loaded"))
    out = embed_window.embed_missing(con, W, posts="_posts", embeddings="_emb")
    assert out["embedded"] == 0 and out["runs"] == []


def test_embed_missing_writes_the_missing_posts_and_drops_the_model(con, encoder, clock, tmp_path):
    out = embed_window.embed_missing(
        con, W, posts="_posts", embeddings="_emb", uri=lambda key: str(_local(tmp_path, key)),
    )
    assert out["embedded"] == 2
    (key,) = out["runs"]
    assert sorted(pd.read_parquet(_local(tmp_path, key))["platform_post_id"]) == ["fresh", "second"]
    assert semantic._MODELS == {}


def test_the_stage_runs_before_the_window_is_pinned():
    order: list[str] = []
    manifest = pd.DataFrame([{"prefix": "posts", "rows": 3}])

    out = v2_daily.orchestrate(
        "2026-09-21", 7, con=None,
        embed=lambda w: order.append("embed") or {"embedded": 4},
        ensure=lambda w: order.append("manifest") or manifest,
        trace_exists=lambda sid: True,
        textsim=lambda sid: {},
        rank_fn=lambda w: order.append("rank") or {},
    )
    assert order == ["embed", "manifest", "rank"]
    assert out["embed"] == {"embedded": 4}


def test_no_embed_stage_leaves_the_summary_as_it_was():
    out = v2_daily.orchestrate(
        "2026-09-21", 7, con=None,
        ensure=lambda w: pd.DataFrame([{"prefix": "posts", "rows": 3}]),
        trace_exists=lambda sid: True, textsim=lambda sid: {}, rank_fn=lambda w: {},
    )
    assert "embed" not in out


def _local(root, key):
    path = root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

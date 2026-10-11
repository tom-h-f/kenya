"""Two writers can score the same post (the GPU backlog run and a CPU pass).

Readers take the latest row per post, so a duplicate never double counts. The
writer re-checks just before it writes so the overlap stays small."""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pandas as pd
import pytest

from kma import db, hatespeech, incitement


def _rows(*items):
    return pd.DataFrame(
        [{"platform": "x", "platform_post_id": pid, "scored_at": at,
          "dehumanisation_score": d, "violence_call_score": 0.0,
          "othering_score": 0.0, "political_criticism_score": 0.0}
         for pid, at, d in items]
    )


@pytest.fixture
def con_with_dups(monkeypatch):
    con = duckdb.connect()
    frame = _rows(
        ("a", datetime(2026, 10, 10, 23, 3, tzinfo=timezone.utc), 0.9),
        ("a", datetime(2026, 10, 10, 23, 16, tzinfo=timezone.utc), 0.8),
        ("b", datetime(2026, 10, 10, 23, 3, tzinfo=timezone.utc), 0.1),
    )
    con.register("_inc", frame)
    src = "(SELECT * FROM _inc)"
    monkeypatch.setattr(db, "incitement_source", lambda platform="*": src)
    monkeypatch.setattr(hatespeech, "incitement_source", lambda platform="*": src)
    return con


def test_latest_incitement_one_row_per_post(con_with_dups):
    got = db.latest_incitement(con_with_dups).df()
    assert sorted(got["platform_post_id"]) == ["a", "b"]
    assert got.set_index("platform_post_id").loc["a", "dehumanisation_score"] == 0.8


def test_hatespeech_nli_reader_dedupes(con_with_dups):
    got = hatespeech._nli_for(con_with_dups, "x")
    assert got["platform_post_id"].is_unique and len(got) == 2
    assert got.set_index("platform_post_id").loc["a", "dehumanisation_score"] == 0.8


def test_drop_scored_removes_posts_scored_meanwhile():
    con = duckdb.connect()
    con.execute("CREATE TABLE _s (platform_post_id VARCHAR)")
    con.execute("INSERT INTO _s VALUES ('a')")
    frame = pd.DataFrame({"platform_post_id": ["a", "b"], "text": ["x", "y"]})
    assert db.drop_scored(con, frame, "_s")["platform_post_id"].tolist() == ["b"]


def test_drop_scored_missing_source_keeps_everything():
    frame = pd.DataFrame({"platform_post_id": ["a"], "text": ["x"]})
    assert len(db.drop_scored(duckdb.connect(), frame, "_nope")) == 1


def test_pending_posts_raises_on_a_failed_read(monkeypatch):
    """A failed read of the scored prefix is not "nothing scored": that would
    drop the anti-join and rescore the newest rows (two tf1 passes scored the
    same 500 posts an hour apart on 2026-09-01)."""
    con = duckdb.connect()
    con.execute("CREATE TABLE _posts (platform VARCHAR, platform_post_id VARCHAR, "
                "text VARCHAR, collected_at TIMESTAMP)")
    monkeypatch.setattr(db, "posts_source", lambda platform="*", type="*": "_posts")
    broken = "read_parquet('/nonexistent/dir/*.parquet')"
    monkeypatch.setattr(db, "_prefix_absent", lambda exc: False)
    with pytest.raises(duckdb.Error):
        db.pending_posts(con, broken)


def test_score_new_skips_posts_another_writer_scored(monkeypatch):
    pending = pd.DataFrame({
        "platform_post_id": ["a", "b"], "text": ["x", "y"],
        "lexicon_hits": [[], []], "lexicon_categories": [[], []],
    })
    monkeypatch.setattr(incitement, "_pending", lambda *a, **k: pending.copy())
    monkeypatch.setattr(incitement, "_pipe", lambda *a, **k: None)
    scores = {"dehumanisation": 0.1, "violence_call": 0.2, "othering": 0.3,
              "political_criticism": 0.4}
    monkeypatch.setattr(incitement, "nli_scores", lambda *a, **k: [dict(scores), dict(scores)])
    monkeypatch.setattr(incitement, "drop_scored",
                        lambda con, df, src: df[df["platform_post_id"] != "a"].reset_index(drop=True))
    written = {}

    class Con:
        def register(self, name, table): written["table"] = table
        def unregister(self, name): pass
        def execute(self, sql): written["sql"] = sql

    assert incitement.score_new(Con(), limit=2) == 1
    assert written["table"].column("platform_post_id").to_pylist() == ["b"]

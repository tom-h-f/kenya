"""A rolling window has to name one corpus, forever, from its id alone.

Run against a local directory standing in for R2, like `test_bench.py`.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import duckdb
import pytest

from kma import bench, window
from test_bench import FakeR2, _uri, _write


@pytest.fixture
def corpus(tmp_path):
    con = duckdb.connect()
    for dt, rows in (("2026-09-01", 2), ("2026-09-02", 3), ("2026-09-03", 5), ("2026-09-04", 7)):
        _write(con, tmp_path / f"posts/platform=x/type=search/dt={dt}/run=r.parquet", rows)
    for dt in ("2026-08-20", "2026-09-03", "2026-09-04"):
        _write(con, tmp_path / f"embeddings/platform=x/dt={dt}/run=e.parquet", 1)
    _write(con, tmp_path / "authors/platform=x/dt=2026-09-03/run=a.parquet", 11)
    return con, tmp_path, FakeR2(tmp_path)


def test_the_id_carries_end_and_length():
    assert window.Window(date(2026, 9, 21), 90).id == "daily-2026-09-21-90d"


def test_start_is_inclusive_and_counts_days():
    w = window.Window(date(2026, 9, 21), 1)
    assert w.start == w.end
    assert window.Window(date(2026, 9, 21), 90).start == date(2026, 6, 24)


def test_default_end_is_yesterday_in_utc():
    """A `dt` partition is only closed once its day is over."""
    late = datetime(2026, 9, 22, 23, 59, tzinfo=timezone.utc)
    assert window.default_end(late) == date(2026, 9, 21)
    assert window.Window.ending("", now=late).end == date(2026, 9, 21)


def test_a_window_needs_a_day():
    with pytest.raises(ValueError):
        window.Window(date(2026, 9, 21), 0)


def test_posts_are_windowed_by_collection_day(corpus):
    con, root, client = corpus
    w = window.Window(date(2026, 9, 3), 2)
    m = window.build(w, con=con, client=client, uri=_uri(root), write=False)
    posts = m[m["prefix"] == "posts"]
    assert sorted(posts["dt"]) == ["2026-09-02", "2026-09-03"]
    assert posts["rows"].sum() == 8


def test_embeddings_are_kept_from_before_the_window_but_not_after(corpus):
    """A post re-collected inside the window may have been embedded weeks
    earlier; windowing the start of embeddings would drop its vector."""
    con, root, client = corpus
    w = window.Window(date(2026, 9, 3), 2)
    m = window.build(w, con=con, client=client, uri=_uri(root), write=False)
    assert sorted(m.loc[m["prefix"] == "embeddings", "dt"]) == ["2026-08-20", "2026-09-03"]


def test_prefixes_the_detector_does_not_read_are_left_out(corpus):
    con, root, client = corpus
    m = window.build(window.Window(date(2026, 9, 4), 30), con=con, client=client,
                     uri=_uri(root), write=False)
    assert set(m["prefix"]) == {"posts", "embeddings"}


def test_ensure_reads_back_rather_than_rebuilding(corpus):
    """The same id must read the same objects after a backfill lands in an
    old partition - that is what makes a daily run reproducible."""
    con, root, client = corpus
    w = window.Window(date(2026, 9, 3), 2)
    # R2 has no directories; the local stand-in needs the parent to exist.
    (root / f"bench/snapshot={w.id}").mkdir(parents=True)
    first = window.ensure(w, con=con, client=client, uri=_uri(root))
    _write(con, root / "posts/platform=x/type=replies/dt=2026-09-03/run=late.parquet", 100)
    again = window.ensure(w, con=con, client=client, uri=_uri(root))
    assert len(again) == len(first)
    src = bench.pinned_source(again, "posts")
    assert con.sql(f"SELECT count(*) FROM {src}").fetchone()[0] == 8

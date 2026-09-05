"""Snapshots have one job: the same name reads the same corpus a week later.

These tests run against a local directory standing in for R2, so the
path-listing and pinning logic is exercised without the network.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from kma import bench


class FakeR2:
    """list_objects_v2 over a local directory, paginated like the real thing so
    the pagination path is actually covered."""

    def __init__(self, root: Path, page_size: int = 2) -> None:
        self.root = root
        self.page_size = page_size

    def list_objects_v2(self, Bucket: str, Prefix: str, ContinuationToken: str | None = None):
        keys = sorted(
            str(p.relative_to(self.root))
            for p in self.root.rglob("*")
            if p.is_file() and str(p.relative_to(self.root)).startswith(Prefix)
        )
        start = int(ContinuationToken or 0)
        page = keys[start : start + self.page_size]
        contents = [
            {"Key": k, "Size": (self.root / k).stat().st_size, "ETag": f'"etag-{k}"'}
            for k in page
        ]
        nxt = start + self.page_size
        return {
            "Contents": contents,
            "IsTruncated": nxt < len(keys),
            "NextContinuationToken": str(nxt),
        }


def _write(con: duckdb.DuckDBPyConnection, path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"platform_post_id": [f"p{i}" for i in range(rows)]})
    con.register("_t", df)
    con.execute(f"COPY _t TO '{path}' (FORMAT parquet)")
    con.unregister("_t")


@pytest.fixture
def corpus(tmp_path):
    con = duckdb.connect()
    _write(con, tmp_path / "posts/platform=x/type=search/dt=2026-09-01/run=a.parquet", 3)
    _write(con, tmp_path / "posts/platform=x/type=search/dt=2026-09-02/run=b.parquet", 5)
    _write(con, tmp_path / "posts/platform=x/type=replies/dt=2026-09-02/run=c.parquet", 7)
    _write(con, tmp_path / "authors/platform=x/dt=2026-09-02/run=d.parquet", 11)
    return con, tmp_path, FakeR2(tmp_path)


def _uri(root: Path):
    return lambda key: f"{root}/{key}"


def test_partition_values_reads_hive_segments():
    got = bench.partition_values("posts/platform=x/type=search/dt=2026-09-01/run=a.parquet")
    assert got == {"platform": "x", "type": "search", "dt": "2026-09-01"}


def test_partition_values_ignores_unkeyed_segments():
    assert bench.partition_values("census_runs/plain/run=a.parquet") == {}


def test_snapshot_counts_rows_from_footers(corpus):
    con, root, client = corpus
    m = bench.snapshot(
        "s1", prefixes=("posts", "authors"), con=con, client=client,
        uri=_uri(root), write=False,
    )
    assert len(m) == 4
    assert m.groupby("prefix")["rows"].sum().to_dict() == {"posts": 15, "authors": 11}
    # Pagination is exercised: the fake pages two at a time.
    assert set(m["type"].dropna()) == {"search", "replies"}


def test_snapshot_without_rows_still_lists(corpus):
    con, root, client = corpus
    m = bench.snapshot(
        "s1", prefixes=("posts",), con=con, client=client, uri=_uri(root),
        rows=False, write=False,
    )
    assert len(m) == 3
    assert m["rows"].isna().all()


def test_snapshot_skips_prefixes_with_no_objects(corpus):
    con, root, client = corpus
    m = bench.snapshot(
        "s1", prefixes=("posts", "follows"), con=con, client=client,
        uri=_uri(root), write=False,
    )
    assert set(m["prefix"]) == {"posts"}


def test_snapshot_refuses_to_write_an_empty_manifest(corpus):
    con, root, client = corpus
    with pytest.raises(RuntimeError, match="no objects"):
        bench.snapshot("s1", prefixes=("follows",), con=con, client=client, uri=_uri(root))


def test_pinned_source_is_a_path_list_not_a_glob(corpus):
    con, root, client = corpus
    m = bench.snapshot(
        "s1", prefixes=("posts",), con=con, client=client, uri=_uri(root), write=False,
    )
    src = bench.pinned_source(m, "posts")
    assert "*" not in src
    assert src.count("run=") == 3
    assert con.sql(f"SELECT count(*) FROM {src}").fetchone()[0] == 15


def test_pinned_source_filters_on_partition(corpus):
    con, root, client = corpus
    m = bench.snapshot(
        "s1", prefixes=("posts",), con=con, client=client, uri=_uri(root), write=False,
    )
    src = bench.pinned_source(m, "posts", type="search")
    assert con.sql(f"SELECT count(*) FROM {src}").fetchone()[0] == 8


def test_pinned_read_ignores_objects_written_after_the_snapshot(corpus):
    """The acceptance criterion: a pinned read does not grow when the corpus does."""
    con, root, client = corpus
    m = bench.snapshot(
        "s1", prefixes=("posts",), con=con, client=client, uri=_uri(root), write=False,
    )
    before = con.sql(f"SELECT count(*) FROM {bench.pinned_source(m, 'posts')}").fetchone()[0]

    _write(con, root / "posts/platform=x/type=search/dt=2026-09-06/run=z.parquet", 99)

    after = con.sql(f"SELECT count(*) FROM {bench.pinned_source(m, 'posts')}").fetchone()[0]
    assert before == after == 15

    grown = bench.snapshot(
        "s2", prefixes=("posts",), con=con, client=client, uri=_uri(root), write=False,
    )
    assert con.sql(f"SELECT count(*) FROM {bench.pinned_source(grown, 'posts')}").fetchone()[0] == 114


def test_pinned_source_rejects_an_unknown_partition_column(corpus):
    con, root, client = corpus
    m = bench.snapshot(
        "s1", prefixes=("posts",), con=con, client=client, uri=_uri(root), write=False,
    )
    with pytest.raises(ValueError, match="unknown partition column"):
        bench.pinned_source(m, "posts", nonsense="x")


def test_pinned_source_rejects_an_empty_selection(corpus):
    con, root, client = corpus
    m = bench.snapshot(
        "s1", prefixes=("posts",), con=con, client=client, uri=_uri(root), write=False,
    )
    with pytest.raises(ValueError, match="no objects"):
        bench.pinned_source(m, "posts", type="hate_search")


def test_diff_reports_additions_and_disappearances(corpus):
    con, root, client = corpus
    before = bench.snapshot(
        "s1", prefixes=("posts",), con=con, client=client, uri=_uri(root), write=False,
    )
    _write(con, root / "posts/platform=x/type=search/dt=2026-09-06/run=z.parquet", 99)
    after = bench.snapshot(
        "s2", prefixes=("posts",), con=con, client=client, uri=_uri(root), write=False,
    )

    d = bench.diff(before, after).set_index("prefix")
    assert d.loc["posts", "added"] == 1
    assert d.loc["posts", "added_rows"] == 99
    assert d.loc["posts", "removed"] == 0

    # A per-run object vanishing means something deleted history, so it is
    # reported rather than silently treated as no change.
    (root / "posts/platform=x/type=search/dt=2026-09-01/run=a.parquet").unlink()
    shrunk = bench.snapshot(
        "s3", prefixes=("posts",), con=con, client=client, uri=_uri(root), write=False,
    )
    assert bench.diff(after, shrunk).set_index("prefix").loc["posts", "removed"] == 1


def test_snapshot_roundtrips_through_a_written_manifest(corpus):
    con, root, client = corpus
    # R2 creates keys on write; a local filesystem stand-in needs the directory.
    (root / "bench/snapshot=s1").mkdir(parents=True)
    written = bench.snapshot(
        "s1", prefixes=("posts",), con=con, client=client, uri=_uri(root), write=True,
    )
    back = bench.load("s1", con=con, uri=_uri(root))
    assert sorted(back["path"]) == sorted(written["path"])
    assert back["snapshot"].unique().tolist() == ["s1"]

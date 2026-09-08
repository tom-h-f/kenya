"""Census-pass counter writes (local parquet, no R2)."""

from __future__ import annotations

import logging

import duckdb
import pytest

from kenya_monitor.storage import CENSUS_RUN_SCHEMA, Storage


class _LocalStorage(Storage):
    """Storage with the R2 secret and the r2:// COPY target swapped for a local
    file, so the real write path (schema, dtypes, parquet round-trip) runs."""

    def __init__(self, out_path):
        self.cfg = None
        self.con = duckdb.connect()
        self.out = out_path
        self.keys: list[str] = []

    def _uri(self, key: str) -> str:
        self.keys.append(key)
        return str(self.out)


def _one(path) -> dict:
    """The single written row as a dict.

    fetchall rather than .df(), and the timestamp read back as text: neither
    numpy nor pytz is a collector dependency, and a test must not quietly make
    one required."""
    con = duckdb.connect()
    rel = con.sql(
        f"SELECT * REPLACE (collected_at::VARCHAR AS collected_at) FROM '{path}'"
    )
    rows = rel.fetchall()
    assert len(rows) == 1
    return dict(zip(rel.columns, rows[0]))


@pytest.fixture
def storage(tmp_path):
    return _LocalStorage(tmp_path / "census_run.parquet")


def test_write_census_run_writes_one_row_under_the_census_runs_prefix(storage):
    key = storage.write_census_run(
        {
            "pass_kind": "merged",
            "candidates_in_band": 3176,
            "candidates_uncensused": 2874,
            "selected_retweeted": 250,
            "due_retweeted": 245,
            "fetched_retweeted": 245,
            "skipped_ttl_retweeted": 5,
            "degree_p50": 73,
            "n_over_band_max": 16,
            "band_max": 100,
        },
        platform="x",
    )

    assert key.startswith("census_runs/platform=x/dt=")
    assert key.endswith(".parquet")
    row = _one(storage.out)
    assert row["candidates_uncensused"] == 2874
    assert row["pass_kind"] == "merged"
    assert row["platform"] == "x"
    assert row["run_id"]
    assert row["collected_at"] is not None


def test_write_census_run_persists_a_pass_that_fetched_nothing(storage):
    """A pass with no due work is the saturation observation Q3 is watching
    for, not an outage. Skipping it would make the two indistinguishable."""
    key = storage.write_census_run(
        {
            "pass_kind": "baseline",
            "candidates_in_band": 412,
            "candidates_uncensused": 0,
            "selected_retweeted": 0,
            "fetched_retweeted": 0,
        }
    )

    assert key is not None
    row = _one(storage.out)
    assert row["fetched_retweeted"] == 0
    assert row["candidates_in_band"] == 412


def test_write_census_run_ignores_unknown_counters(storage, caplog):
    """`collect_snowball` threads one stats dict through several stages; a key
    the schema does not carry must not fail the write - but it must be reported.

    An undeclared column reads back as NULL, which is indistinguishable from a
    genuine zero, so a rename on either writer would silently retire a counter."""
    with caplog.at_level(logging.WARNING, logger="kenya_monitor"):
        key = storage.write_census_run(
            {"fetched_retweeted": 1, "selected_deg_min": 4, "not_a_column": "x"}
        )

    assert key is not None
    row = _one(storage.out)
    assert "not_a_column" not in row
    assert row["selected_deg_min"] == 4
    assert "not_a_column" in caplog.text


def test_write_census_run_skips_an_empty_stats_dict(storage):
    assert storage.write_census_run({}) is None




def test_census_run_schema_keeps_the_supply_denominator():
    """Degree is recoverable from engagements/ by grouping on the object; the
    denominator - how many candidates existed and how many were skipped - is
    not recoverable after the fact, which is the whole reason this prefix
    exists."""
    names = {f.name for f in CENSUS_RUN_SCHEMA}
    assert {"candidates_in_band", "candidates_uncensused",
            "skipped_ttl_retweeted", "due_retweeted"} <= names


def test_write_census_ttl_writes_one_row_per_object(tmp_path):
    """The TTL ledger is the only per-object record of census SELECTION.

    Without it, an object selected that returned no retweeters writes no
    `engagements/` row and is indistinguishable from an object never selected -
    the ambiguity that held Track B per-id reproduction to 0.404 recall.
    """
    out = tmp_path / "ttl.parquet"
    store = _LocalStorage(out)
    key = store.write_census_ttl(
        {"1111": "2026-09-08T10:00:00+00:00", "2222": "2026-09-08T11:00:00+00:00"},
        platform="x",
    )
    assert key is not None and "census_ttl/platform=x" in key

    con = duckdb.connect()
    rel = con.sql(f"SELECT * FROM '{out}'")
    rows = [dict(zip(rel.columns, r)) for r in rel.fetchall()]
    assert len(rows) == 2
    assert {r["object_id"] for r in rows} == {"1111", "2222"}
    assert all(r["captured_at"] for r in rows)


def test_write_census_ttl_skips_an_empty_ledger(tmp_path):
    store = _LocalStorage(tmp_path / "ttl.parquet")
    assert store.write_census_ttl({}, platform="x") is None

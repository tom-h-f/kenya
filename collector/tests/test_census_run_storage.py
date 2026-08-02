"""Census-pass counter writes (local parquet, no R2)."""

from __future__ import annotations

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


def test_write_census_run_ignores_unknown_counters(storage):
    """`collect_snowball` threads one stats dict through several stages; a key
    the schema does not carry must not fail the write."""
    key = storage.write_census_run(
        {"fetched_retweeted": 1, "selected_deg_min": 4, "not_a_column": "x"}
    )

    assert key is not None
    row = _one(storage.out)
    assert "not_a_column" not in row
    assert row["selected_deg_min"] == 4


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

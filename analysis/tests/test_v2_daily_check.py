"""Host checks for the v2 daily job, and the deploy files that must keep them true.

No R2 and no network: the R2 client is a fake and the persisted tables are local
parquet. The unit-file tests pin the properties that make the job keep working
unattended (retry on failure, catch-up after downtime, bounded memory), so a
well-meant edit to `deploy/cloud/` fails a test instead of failing at 03:30.
"""

from __future__ import annotations

import configparser
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from kma import canary, db, v2_daily_check, v2_daily_run

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deploy" / "cloud"
NOW = 1_800_000_000.0
FULL_ENV = {v: "x" for v in db.R2_CREDENTIAL_VARS}


class FakeClient:
    def __init__(self, fail_on: str | None = None, corrupt: bool = False):
        self.objects: dict[str, bytes] = {}
        self.fail_on = fail_on
        self.corrupt = corrupt

    def put_object(self, Bucket, Key, Body):
        if self.fail_on and Key.startswith(self.fail_on):
            raise PermissionError("AccessDenied")
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        body = b"other" if self.corrupt else self.objects[Key]
        return {"Body": type("B", (), {"read": lambda self, b=body: b})()}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)


def _by_name(checks):
    return {c.name: c for c in checks}


def test_preflight_passes_on_a_healthy_host(tmp_path):
    client = FakeClient()
    checks = v2_daily_check.preflight(
        env=FULL_ENV, textfile_dir=tmp_path, client=client, available_gb=16.0
    )
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]
    assert {"r2 credentials", "textfile dir writable", "memory available", "r2 write"} <= set(
        _by_name(checks)
    )


def test_preflight_leaves_no_probe_object_or_temp_file_behind(tmp_path):
    client = FakeClient()
    v2_daily_check.preflight(env=FULL_ENV, textfile_dir=tmp_path, client=client, available_gb=16.0)
    assert client.objects == {}
    assert list(tmp_path.iterdir()) == []


def test_missing_credentials_fail_and_skip_the_r2_probes(tmp_path):
    client = FakeClient()
    checks = _by_name(v2_daily_check.preflight(
        env={}, textfile_dir=tmp_path, client=client, available_gb=16.0
    ))
    assert not checks["r2 credentials"].ok
    assert "r2 write" not in checks
    assert client.objects == {}


def test_a_textfile_dir_the_user_cannot_write_fails(tmp_path):
    checks = _by_name(v2_daily_check.preflight(
        env=FULL_ENV, textfile_dir=tmp_path / "missing" / "dir", available_gb=16.0
    ))
    assert not checks["textfile dir writable"].ok


def test_low_or_unreadable_memory_fails():
    assert not v2_daily_check.check_memory(2.0).ok
    assert not v2_daily_check.check_memory(None).ok
    assert v2_daily_check.check_memory(v2_daily_check.MEASURED_PEAK_GB).ok


def test_a_write_denied_under_one_prefix_fails_and_cleans_up():
    client = FakeClient(fail_on="coord2/")
    check = v2_daily_check.check_r2_write(client, "bucket")
    assert not check.ok and "AccessDenied" in check.detail
    assert client.objects == {}


def test_a_read_back_mismatch_fails_and_cleans_up():
    client = FakeClient(corrupt=True)
    check = v2_daily_check.check_r2_write(client, "bucket")
    assert not check.ok
    assert client.objects == {}


def test_the_write_probe_touches_both_prefixes_the_job_writes():
    seen: list[str] = []

    class Spy(FakeClient):
        def put_object(self, Bucket, Key, Body):
            seen.append(Key.split("/")[0])
            super().put_object(Bucket, Key, Body)

    v2_daily_check.check_r2_write(Spy(), "bucket")
    assert seen == ["bench", "coord2"]


def test_the_runner_path_imports_without_modal():
    code = (
        "import sys, kma.v2_daily_check, kma.v2_daily_run;"
        "sys.exit(1 if 'modal' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO / "analysis").returncode == 0


def _scores(tmp_path, *, computed_at=NOW, ids=("1", "2", "3"), drop=()):
    frame = pd.DataFrame({
        "user_id": list(ids), "community": 0, "centrality": 1.0,
        "snapshot": "daily-2026-10-09-30d",
        "computed_at": pd.Timestamp(computed_at, unit="s", tz="UTC"),
    }).drop(columns=list(drop))
    path = tmp_path / "scores.parquet"
    frame.to_parquet(path)
    return f"(SELECT *, 'r1' AS run FROM read_parquet('{path}'))"


def _members(tmp_path, rows=5):
    path = tmp_path / "members.parquet"
    pd.DataFrame({"user_id": [str(i) for i in range(rows)]}).to_parquet(path)
    return f"read_parquet('{path}', filename=true)"


def _runs(tmp_path, state="reused", *, field=True):
    frame = pd.DataFrame({"window_id": ["daily-2026-10-09-30d"], "text_trace_mode": ["modal"]})
    if field:
        frame["text_trace"] = state
    path = tmp_path / "runs.parquet"
    frame.to_parquet(path)
    return f"read_parquet('{path}', filename=true)"


def _post(tmp_path, *, gauge_at=NOW + 60, text_trace="reused", trace_gauge=None,
          runs_field=True, **scores_kwargs):
    textfile = tmp_path / "textfile"
    present = text_trace in ("built", "reused") if trace_gauge is None else trace_gauge
    v2_daily_run.write_gauge(textfile, now=gauge_at, text_trace_present=present)
    return _by_name(v2_daily_check.postrun(
        con=duckdb.connect(), textfile_dir=textfile, now=lambda: NOW + 3600,
        scores_source=_scores(tmp_path, **scores_kwargs),
        members_source=_members(tmp_path),
        runs_source=_runs(tmp_path, text_trace, field=runs_field),
    ))


def test_postrun_passes_after_a_good_run(tmp_path):
    checks = _post(tmp_path)
    assert all(c.ok for c in checks.values()), [c for c in checks.values() if not c.ok]


def test_a_stale_gauge_fails_at_the_alert_threshold(tmp_path):
    textfile = tmp_path / "textfile"
    v2_daily_run.write_gauge(textfile, now=NOW)
    old = _by_name(v2_daily_check.postrun(
        con=duckdb.connect(), textfile_dir=textfile,
        now=lambda: NOW + 31 * 3600,
        scores_source=_scores(tmp_path), members_source=_members(tmp_path),
        runs_source=_runs(tmp_path),
    ))
    assert not old["gauge fresh"].ok


def test_a_missing_gauge_fails(tmp_path):
    checks = _by_name(v2_daily_check.postrun(
        con=duckdb.connect(), textfile_dir=tmp_path / "nothing", now=lambda: NOW,
        scores_source=_scores(tmp_path), members_source=_members(tmp_path),
        runs_source=_runs(tmp_path),
    ))
    assert not checks["gauge present"].ok


def test_scores_without_a_community_column_fail(tmp_path):
    assert not _post(tmp_path, drop=("community",))["scores columns"].ok


def test_a_planted_canary_id_in_scores_fails(tmp_path):
    planted = f"{canary.ID_PREFIX}0001"
    assert not _post(tmp_path, ids=("1", planted))["no canary id in scores"].ok


def test_an_empty_listing_fails(tmp_path):
    assert not _post(tmp_path, ids=())["scores listed accounts"].ok


def test_a_gauge_older_than_its_data_fails(tmp_path):
    assert not _post(tmp_path, gauge_at=NOW - 600)["gauge written after its data"].ok


def test_empty_daily_members_fail(tmp_path):
    checks = _by_name(v2_daily_check.postrun(
        con=duckdb.connect(), textfile_dir=tmp_path / "nothing", now=lambda: NOW,
        scores_source=_scores(tmp_path), members_source=_members(tmp_path, rows=0),
        runs_source=_runs(tmp_path),
    ))
    assert not checks["daily members written"].ok


@pytest.mark.parametrize("state", ["built", "reused", "skipped"])
def test_a_run_with_a_trace_or_a_deliberate_skip_passes(tmp_path, state):
    checks = _post(tmp_path, text_trace=state)
    assert checks["text trace in the latest run"].ok
    assert checks["text trace gauge"].ok
    assert state in checks["text trace in the latest run"].detail


def test_a_run_that_ranked_without_its_trace_fails_and_names_the_window(tmp_path):
    checks = _post(tmp_path, text_trace="missing")
    failed = checks["text trace in the latest run"]
    assert not failed.ok
    assert "missing" in failed.detail and "daily-2026-10-09-30d" in failed.detail
    assert checks["text trace gauge"].ok  # the gauge says 0 and so does the row
    assert checks["gauge fresh"].ok and checks["scores listed accounts"].ok


def test_a_gauge_that_disagrees_with_the_run_row_fails(tmp_path):
    assert not _post(tmp_path, text_trace="reused", trace_gauge=False)["text trace gauge"].ok


def test_a_run_row_from_before_the_flag_cannot_be_vouched_for(tmp_path):
    checks = _post(tmp_path, runs_field=False)
    assert not checks["text trace in the latest run"].ok
    assert "no text_trace field" in checks["text trace in the latest run"].detail


def test_a_gauge_file_without_the_text_sample_fails_that_check(tmp_path):
    textfile = tmp_path / "textfile"
    v2_daily_run.write_gauge(textfile, now=NOW + 60)
    checks = _by_name(v2_daily_check.postrun(
        con=duckdb.connect(), textfile_dir=textfile, now=lambda: NOW + 3600,
        scores_source=_scores(tmp_path), members_source=_members(tmp_path),
        runs_source=_runs(tmp_path),
    ))
    assert not checks["text trace gauge"].ok


def _unit(name: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str
    parser.read(DEPLOY / name)
    return parser


def test_the_service_runs_the_runner_without_modal_and_with_a_frozen_lock():
    exec_start = _unit("kma-v2-daily.service")["Service"]["ExecStart"]
    assert "kma.v2_daily_run" in exec_start
    assert "--frozen" in exec_start and "--no-dev" in exec_start
    assert "modal" not in exec_start


def test_the_service_is_bounded_and_retries_after_a_transient_failure():
    service = _unit("kma-v2-daily.service")["Service"]
    assert service["MemoryMax"].endswith("G")
    assert service["Restart"] == "on-failure"
    assert int(service["RestartSec"]) >= 60
    assert service["TimeoutStartSec"]
    assert service["User"] == "tom"
    assert service["SupplementaryGroups"] == "debian"
    assert service["EnvironmentFile"] == "/etc/kma/v2-daily.env"


def test_the_retry_budget_is_declared_in_the_unit_section():
    unit = _unit("kma-v2-daily.service")["Unit"]
    assert int(unit["StartLimitBurst"]) >= 2
    assert unit["StartLimitIntervalSec"]


def test_the_unit_points_the_gauge_at_the_node_exporter_textfile_dir():
    env = _unit("kma-v2-daily.service")["Service"]["Environment"]
    assert env == "KMA_TEXTFILE_DIR=/opt/iot-cloud/state/textfile"


def test_the_timer_catches_up_after_downtime():
    timer = _unit("kma-v2-daily.timer")["Timer"]
    assert timer["Persistent"] == "true"
    assert timer["OnCalendar"].endswith("UTC")


def test_the_gauge_name_matches_the_alert_contract():
    assert v2_daily_run.GAUGE_METRIC == "kma_v2_daily_last_success_timestamp_seconds"
    assert v2_daily_run.GAUGE_FILE == "kma_v2_daily.prom"


def test_the_verify_script_is_executable_and_runs_both_stages():
    script = DEPLOY / "verify.sh"
    assert script.stat().st_mode & 0o111
    text = script.read_text()
    assert "kma.v2_daily_check" in text
    assert "preflight)" in text and "postrun)" in text

"""Checks that the v2 daily job on its host works, and still works after it runs.

    uv run python -m kma.v2_daily_check preflight --textfile-dir /opt/iot-cloud/state/textfile
    uv run python -m kma.v2_daily_check postrun   --textfile-dir /opt/iot-cloud/state/textfile

`preflight` changes nothing a run depends on: it reads R2, writes and deletes a
probe object under `bench/` and `coord2/` (the two prefixes the job writes),
and creates and removes a temp file in the textfile directory. `postrun` reads
what the last run left behind. Each prints one line per check and exits 1 if
any failed, so a deploy script, a timer's `ExecStartPost`, or a human can gate
on it.

The gauge is the alert signal (`CloudKenyaV2DailyStale`); these checks are what
the gauge cannot say: that the run listed accounts, that `kind=scores` carries
the `community` column the collector reads, that no planted canary id leaked
into it, that the gauge was written after its data and not before, and whether
the run had a text trace. A run whose trace never arrived from the Modal job
still lists accounts and still writes the gauge, so only these checks say that
its ranking has no text similarity in it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import duckdb

from kma import canary, db
from kma.text_trace import BUILT, REUSED, SKIPPED
from kma.v2_daily_run import GAUGE_FILE, GAUGE_METRIC, TEXT_TRACE_METRIC

# Peak RSS of the 7 day window measured 2026-10-10 (deploy/cloud/README.md). It
# grows with embedded vectors, so a run at fuller coverage needs more: this is a
# floor a host must clear, not a guarantee.
MEASURED_PEAK_GB = 6.6

# The stale-gauge alert fires at 30 h; a check that disagrees with the alert
# would pass a state that pages.
DEFAULT_MAX_AGE_HOURS = 30.0

REQUIRED_SCORE_COLUMNS = ("user_id", "community", "centrality", "snapshot", "computed_at")
WRITE_PREFIXES = ("bench", "coord2")


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


def _fmt(check: Check) -> str:
    return f"{'ok  ' if check.ok else 'FAIL'} {check.name}: {check.detail}".rstrip(": ")


def mem_available_gb(meminfo: Path = Path("/proc/meminfo")) -> float | None:
    """`MemAvailable` in GiB, or None where /proc/meminfo does not exist."""
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        return None
    return None


def check_credentials(env: dict[str, str]) -> Check:
    missing = [v for v in db.R2_CREDENTIAL_VARS if not env.get(v)]
    return Check("r2 credentials", not missing, f"missing {', '.join(missing)}" if missing else "present")


def check_no_modal() -> Check:
    """The scheduled path must not depend on Modal: importing the runner here
    already happened, so a modal import anywhere on it would be in sys.modules."""
    return Check("no modal on the runner path", "modal" not in sys.modules, "modal not imported")


def check_textfile_dir(directory: Path) -> Check:
    """Create and remove a temp file the way `write_gauge` does: node-exporter
    only reads `*.prom`, so the probe's suffix keeps it from being scraped."""
    probe = directory / f".preflight.{os.getpid()}.tmp"
    try:
        probe.write_text("probe\n")
        probe.unlink()
    except OSError as exc:
        return Check("textfile dir writable", False, f"{directory}: {exc}")
    return Check("textfile dir writable", True, str(directory))


def check_memory(available_gb: float | None, floor_gb: float = MEASURED_PEAK_GB) -> Check:
    if available_gb is None:
        return Check("memory available", False, "cannot read MemAvailable")
    return Check(
        "memory available", available_gb >= floor_gb,
        f"{available_gb:.1f} GiB available, measured peak {floor_gb:.1f} GiB",
    )


def check_r2_read(con: duckdb.DuckDBPyConnection) -> Check:
    source = db.posts_source("x")
    try:
        n = con.sql(f"SELECT count(*) FROM (SELECT 1 FROM {source} LIMIT 1)").fetchone()[0]
    except duckdb.Error as exc:
        return Check("r2 read", False, str(exc).splitlines()[0][:160])
    return Check("r2 read", n > 0, "posts readable" if n else "posts prefix is empty")


def check_r2_write(client, bucket: str, prefixes=WRITE_PREFIXES) -> Check:
    """Put, read back and delete one object under each prefix the job writes. The
    key is unique and outside any hive glob the readers use."""
    made: list[str] = []
    try:
        for prefix in prefixes:
            key = f"{prefix}/_preflight/{uuid.uuid4().hex}.txt"
            client.put_object(Bucket=bucket, Key=key, Body=b"preflight")
            made.append(key)
            if client.get_object(Bucket=bucket, Key=key)["Body"].read() != b"preflight":
                return Check("r2 write", False, f"read-back mismatch under {prefix}/")
        for key in made:
            client.delete_object(Bucket=bucket, Key=key)
        made.clear()
    except Exception as exc:  # noqa: BLE001 - any client error is a failed check
        return Check("r2 write", False, f"{type(exc).__name__}: {str(exc)[:160]}")
    finally:
        for key in made:
            try:
                client.delete_object(Bucket=bucket, Key=key)
            except Exception:  # noqa: BLE001 - best effort cleanup of a probe
                pass
    return Check("r2 write", True, f"put, read and delete under {', '.join(p + '/' for p in prefixes)}")


def preflight(
    *,
    env: dict[str, str],
    textfile_dir: Path,
    con: duckdb.DuckDBPyConnection | None = None,
    client=None,
    bucket: str | None = None,
    available_gb: float | None = None,
    floor_gb: float = MEASURED_PEAK_GB,
) -> list[Check]:
    checks = [check_credentials(env), check_no_modal(), check_textfile_dir(textfile_dir),
              check_memory(available_gb, floor_gb)]
    if checks[0].ok and con is not None:
        checks.append(check_r2_read(con))
    if checks[0].ok and client is not None:
        checks.append(check_r2_write(client, bucket or db.BUCKET))
    return checks


def read_gauge(directory: Path) -> tuple[float | None, str]:
    path = directory / GAUGE_FILE
    try:
        text = path.read_text()
    except OSError as exc:
        return None, f"{path}: {exc}"
    for line in text.splitlines():
        if line.startswith(GAUGE_METRIC + " "):
            try:
                return float(line.split()[1]), str(path)
            except ValueError:
                break
    return None, f"{path}: no valid {GAUGE_METRIC} sample"


def read_text_trace_gauge(directory: Path) -> float | None:
    """The text-trace sample beside the freshness gauge, or None when the file
    has none (it was written by a run that predates the sample)."""
    try:
        text = (directory / GAUGE_FILE).read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(TEXT_TRACE_METRIC + " "):
            try:
                return float(line.split()[1])
            except ValueError:
                return None
    return None


def check_text_trace(con: duckdb.DuckDBPyConnection, runs_source: str,
                     gauge: float | None) -> list[Check]:
    """What the latest `daily_runs` row and the gauge say about the text trace.

    `missing` fails: the run ranked without text similarity, which is over half
    the fused edges, and nothing else about it looks wrong. `skipped` passes,
    since `--text-trace none` is a decision. A row with no `text_trace` field
    was written by code that predates it and cannot be vouched for."""
    try:
        row = con.sql(
            f"SELECT * FROM {runs_source} ORDER BY filename DESC LIMIT 1"
        ).df()
    except duckdb.Error as exc:
        return [Check("text trace in the latest run", False, str(exc).splitlines()[0][:160])]
    if row.empty:
        return [Check("text trace in the latest run", False, "no daily_runs row")]
    if "text_trace" not in row.columns or row["text_trace"].isna().all():
        return [Check("text trace in the latest run", False,
                      "latest daily_runs row has no text_trace field")]
    state = str(row["text_trace"].iloc[0])
    mode = str(row["text_trace_mode"].iloc[0]) if "text_trace_mode" in row.columns else "?"
    window_id = str(row["window_id"].iloc[0]) if "window_id" in row.columns else "?"
    checks = [Check("text trace in the latest run", state in (BUILT, REUSED, SKIPPED),
                    f"{state} (mode {mode}) for {window_id}")]
    present = state in (BUILT, REUSED)
    if gauge is None:
        checks.append(Check("text trace gauge", False, f"no {TEXT_TRACE_METRIC} sample"))
    else:
        checks.append(Check("text trace gauge", int(gauge) == int(present),
                            f"{TEXT_TRACE_METRIC} {int(gauge)}, run says {state}"))
    return checks


def postrun(
    *,
    con: duckdb.DuckDBPyConnection,
    textfile_dir: Path,
    scores_source: str | None = None,
    members_source: str | None = None,
    runs_source: str | None = None,
    now: Callable[[], float] = time.time,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> list[Check]:
    scores_source = scores_source or db.coord2_scores_source("x")
    members_source = members_source or (
        f"read_parquet('r2://{db.BUCKET}/coord2/platform=x/kind=daily_members/dt=*/run=*.parquet', "
        "union_by_name=true, hive_partitioning=true, filename=true)"
    )
    runs_source = runs_source or (
        f"read_parquet('r2://{db.BUCKET}/coord2/platform=x/kind=daily_runs/dt=*/run=*.parquet', "
        "union_by_name=true, hive_partitioning=true, filename=true)"
    )
    checks: list[Check] = []

    gauge, where = read_gauge(textfile_dir)
    if gauge is None:
        checks.append(Check("gauge present", False, where))
    else:
        age_h = (now() - gauge) / 3600
        checks.append(Check("gauge fresh", 0 <= age_h <= max_age_hours,
                            f"{age_h:.1f} h old, alert at {max_age_hours:.0f} h"))

    try:
        frame = con.sql(
            f"SELECT * FROM {scores_source} WHERE run = (SELECT max(run) FROM {scores_source})"
        ).df()
    except duckdb.Error as exc:
        checks.append(Check("scores readable", False, str(exc).splitlines()[0][:160]))
        frame = None
    if frame is not None:
        missing = [c for c in REQUIRED_SCORE_COLUMNS if c not in frame.columns]
        checks.append(Check("scores columns", not missing,
                            f"missing {', '.join(missing)}" if missing else "community and ranking present"))
        checks.append(Check("scores listed accounts", len(frame) > 0, f"{len(frame)} accounts"))
        leaked = [u for u in frame.get("user_id", []) if canary.is_canary(str(u))]
        checks.append(Check("no canary id in scores", not leaked, f"{len(leaked)} planted ids"))
        if "computed_at" in frame.columns and len(frame) and gauge is not None:
            computed = frame["computed_at"].max().timestamp()
            checks.append(Check("gauge written after its data", gauge >= computed - 1,
                                f"gauge {gauge - computed:+.0f} s from scores"))

    checks.extend(check_text_trace(con, runs_source, read_text_trace_gauge(textfile_dir)))

    try:
        latest = con.sql(
            f"SELECT filename, count(*) AS n FROM {members_source} "
            "GROUP BY filename ORDER BY filename DESC LIMIT 1"
        ).fetchone()
        n = latest[1] if latest else 0
        checks.append(Check("daily members written", n > 0, f"{n} rows in the latest run"))
    except duckdb.Error as exc:
        checks.append(Check("daily members written", False, str(exc).splitlines()[0][:160]))
    return checks


def _r2_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the v2 daily job on this host.")
    ap.add_argument("stage", choices=("preflight", "postrun"))
    ap.add_argument("--textfile-dir", type=Path, required=True)
    ap.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS)
    ap.add_argument("--min-available-gb", type=float, default=MEASURED_PEAK_GB)
    args = ap.parse_args(argv)

    env = dict(os.environ)
    if args.stage == "preflight":
        creds_ok = check_credentials(env).ok
        checks = preflight(
            env=env, textfile_dir=args.textfile_dir,
            con=db.connect() if creds_ok else None,
            client=_r2_client() if creds_ok else None,
            available_gb=mem_available_gb(), floor_gb=args.min_available_gb,
        )
    else:
        con = db.connect()
        con.execute("SET enable_progress_bar = false")
        checks = postrun(con=con, textfile_dir=args.textfile_dir, max_age_hours=args.max_age_hours)

    for check in checks:
        print(_fmt(check))
    failed = [c for c in checks if not c.ok]
    print(f"{args.stage}: {len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

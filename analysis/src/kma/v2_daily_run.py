"""The v2 daily pass as a CPU job: manifest, text trace, rank, scores, gauge.

    cd analysis
    KMA_TEXTFILE_DIR=/path/to/textfile uv run python -m kma.v2_daily_run
    uv run python -m kma.v2_daily_run --days 7 --out-dir /scratch/v2   # nothing written to R2

Runs `v2_daily.orchestrate` with its stages on this host instead of Modal, so a
scheduled run and a hand run are the same code as `modal_v2_daily.py`. Nothing
on this path imports `modal`, and the relevance scoring that app calls first
stays where it is: this reads whatever scores `relevance/` already holds.

## What it writes (R2 mode)

- `bench/snapshot=<window id>/manifest.parquet` and the text trace under
  `coord2/kind=textsim`, keyed by window id and reused on a re-run.
- `coord2/kind=daily_*`, from `v2_daily.rank`.
- `coord2/kind=scores`, with the `community` column. The collector's targeting
  (`adaptive.v2_accounts`, `deep_timelines._score_targets`) reads this kind and
  only a hand run wrote it before. One row per LISTED account: the report's
  output, ranked within community, not a second top-N. Canary accounts are
  left out - they are not real ids and the collector would try to fetch them.
- the freshness gauge, last, only after everything above succeeded.

`--out-dir` keeps the manifest and the text trace in memory and writes every
other table under that directory, so a measurement or a test touches no R2
prefix. Reads still come from R2.

## Floors

The rolling rank keeps `coord2.MIN_ENTITIES_PER_USER`; `--min-entities`
overrides it and the value is recorded on every run row. The decided windowed
floors (`windowed.DETECTION`, `windowed.SERIES`) are for per-day and per-week
windows and are applied by `--windowed`, not by the rolling rank. The floor
does not reach the text trace, which keeps `apply_text_floor`'s own default.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd

from kma import canary, coord2, coord2_run, textsim_run, v2_daily, window, windowed

log = logging.getLogger(__name__)

TEXTFILE_DIR_ENV = "KMA_TEXTFILE_DIR"
GAUGE_FILE = "kma_v2_daily.prom"
GAUGE_METRIC = "kma_v2_daily_last_success_timestamp_seconds"

# Measured 2026-10-10, see docs/plans/2026-10-10-cloud-v2-daily.md.
DEFAULT_DAYS = 30


def gauge_text(epoch_seconds: int) -> str:
    return f"# TYPE {GAUGE_METRIC} gauge\n{GAUGE_METRIC} {epoch_seconds}\n"


def write_gauge(directory: Path, now: float | None = None) -> Path:
    """Write the freshness gauge atomically.

    The temp file sits in the same directory (a rename across filesystems is
    not atomic) and does not end in `.prom`, which is the only suffix
    node-exporter's textfile collector reads, so a half-written file is never
    scraped. Mode 0644 because `mkstemp` makes 0600 and the exporter often runs
    as another user inside a container."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / GAUGE_FILE
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{GAUGE_FILE}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(gauge_text(int(time.time() if now is None else now)))
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def local_uri(root: Path) -> Callable[[str], str]:
    def uri(key: str) -> str:
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    return uri


def scores_frame(accounts: pd.DataFrame) -> pd.DataFrame:
    """The `kind=scores` rows for a daily run: the listed accounts, ids as
    strings, planted canary accounts removed."""
    frame = accounts.assign(user_id=accounts["user_id"].astype(str))
    frame = frame[~frame["user_id"].map(canary.is_canary)]
    return frame.reset_index(drop=True)


def windowed_pass(spec: windowed.WindowSpec) -> v2_daily.ExtraPass:
    """The windowed detector as an extra pass over the daily's own posts view,
    canary included, at the floor `windowed` decided for that width."""

    def windowed_scores(ctx: v2_daily.PassContext) -> dict[str, pd.DataFrame]:
        rows = windowed.timed_traces(ctx.con, ctx.view)
        result = windowed.run_rows(rows, width=spec.width, min_entities=spec.min_entities)
        return {f"windowed_{spec.width}": result.per_window.assign(min_entities=spec.min_entities)}

    windowed_scores.__name__ = f"windowed_{spec.width}"
    return windowed_scores


def peak_rss_mb() -> float:
    """Peak resident set of this process; `ru_maxrss` is bytes on macOS and
    kilobytes on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def run(
    end: str | None,
    days: int,
    *,
    con: duckdb.DuckDBPyConnection,
    out_dir: Path | None = None,
    min_entities: int = coord2.MIN_ENTITIES_PER_USER,
    with_canary: bool = False,
    windowed_width: str | None = None,
    textfile_dir: Path | None = None,
) -> dict:
    uri = local_uri(out_dir) if out_dir else None
    held: dict = {}

    def ensure(w: window.Window) -> pd.DataFrame:
        if out_dir:
            held["manifest"] = window.build(w, con=con, write=False)
        else:
            held["manifest"] = window.ensure(w, con=con)
        return held["manifest"]

    def build_trace(window_id: str) -> dict:
        edges, info = textsim_run.build(
            con, window_id,
            threshold=v2_daily.TEXT_THRESHOLD, overlap=-1.0,
            similarity_for=coord2.gpu_cosine_pairs, time_bucket="day",
            persist=out_dir is None, manifest=held["manifest"],
        )
        if out_dir:
            plain = edges.groupby(["source", "target"], as_index=False).agg(
                weight=("weight", "mean")
            ).astype({"source": str, "target": str})
            held["text_edges"] = plain
            for frame, bucket in ((edges, "day"), (plain, None)):
                key = coord2_run.textsim_key(
                    window_id, info["threshold"], overlap=info["overlap"] or None, bucket=bucket
                )
                frame.to_parquet(uri(key), compression="zstd")
        return info

    def rank(w: window.Window) -> dict:
        planted = canary.load_active(con) if with_canary else None
        if with_canary:
            log.info("canary: %d planted post(s) active", 0 if planted is None else len(planted))
        extra = []
        if windowed_width:
            spec = windowed.DETECTION if windowed_width == windowed.DETECTION.width else windowed.SERIES
            extra.append(windowed_pass(spec))
        result = v2_daily.rank(
            con, w, min_entities=min_entities, extra_passes=extra, canary_posts=planted,
            persist=True, uri=uri, manifest=held["manifest"], text_edges=held.get("text_edges"),
        )
        scores = scores_frame(result.tables["accounts"])
        if scores.empty:
            raise RuntimeError(
                f"window {w.id} listed no accounts; refusing to report a healthy run"
            )
        until = (w.end + timedelta(days=1)).isoformat()
        scores_key = coord2_run.persist(
            con, scores, snapshot=w.id, trace_sizes=result.trace_sizes,
            text_threshold=v2_daily.TEXT_THRESHOLD, text_overlap=coord2.TEXT_MIN_OVERLAP,
            min_entities=min_entities, since=w.start.isoformat(), until=until, uri=uri,
        )
        return {**result.summary, "keys": {**result.keys, "scores": scores_key},
                "scores_rows": len(scores)}

    out = v2_daily.orchestrate(
        end or None, days, textsim=build_trace, rank_fn=rank, con=con, ensure=ensure,
        trace_exists=(lambda _: False) if out_dir else None,
    )
    out["peak_rss_mb"] = peak_rss_mb()
    if textfile_dir is not None:
        out["gauge"] = str(write_gauge(textfile_dir))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="The v2 daily pass on this host.")
    ap.add_argument("--end", default="", help="last collection day, ISO; default yesterday UTC")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS, help="window length in days")
    ap.add_argument("--min-entities", type=int, default=coord2.MIN_ENTITIES_PER_USER,
                    help="activity floor of the rolling rank's four bipartite traces")
    ap.add_argument("--canary", action="store_true",
                    help="union the active planted canary into the window")
    ap.add_argument("--windowed", choices=windowed.WIDTHS,
                    help="also run the windowed detector at its decided floor for that width")
    ap.add_argument("--out-dir", type=Path,
                    help="write results here and nothing to R2 (measurement, tests)")
    ap.add_argument("--textfile-dir", type=Path,
                    default=Path(os.environ[TEXTFILE_DIR_ENV]) if os.getenv(TEXTFILE_DIR_ENV) else None,
                    help=f"directory for {GAUGE_FILE}; default ${TEXTFILE_DIR_ENV}")
    args = ap.parse_args(argv)

    if args.textfile_dir is None and args.out_dir is None:
        ap.error(f"set ${TEXTFILE_DIR_ENV} or --textfile-dir: a run that cannot report "
                 "its freshness should not start")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from kma.db import connect

    con = connect()
    # DuckDB's progress bar writes to stdout and corrupts the JSON summary.
    con.execute("SET enable_progress_bar = false")
    out = run(
        args.end, args.days, con=con, out_dir=args.out_dir,
        min_entities=args.min_entities, with_canary=args.canary,
        windowed_width=args.windowed, textfile_dir=args.textfile_dir,
    )
    print(json.dumps(out, default=str, indent=2))


if __name__ == "__main__":
    main()

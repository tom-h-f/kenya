"""The v2 daily pass as a CPU job: manifest, text trace, rank, scores, gauge.

    cd analysis
    KMA_TEXTFILE_DIR=/path/to/textfile uv run python -m kma.v2_daily_run
    uv run python -m kma.v2_daily_run --days 7 --out-dir /scratch/v2   # nothing written to R2

Runs `v2_daily.orchestrate` with its stages on this host instead of Modal, so a
scheduled run and a hand run are the same code as `modal_v2_daily.py`. Nothing
on this path imports `modal`, and the relevance scoring that app calls first
stays where it is: this reads whatever scores `relevance/` already holds.

## What it writes (R2 mode)

- `embeddings/` runs for the window's text-eligible posts that had no vector
  (`embed_window`), BEFORE the manifest is built so the trace sees them. A
  covered window writes nothing and loads no model. `--no-embed` skips it.
- `bench/snapshot=<window id>/manifest.parquet` and the text trace under
  `coord2/kind=textsim`, keyed by window id and reused on a re-run.
- `coord2/kind=daily_*`, from `v2_daily.rank`.
- `coord2/kind=scores`, with the `community` column. The collector's targeting
  (`adaptive.v2_accounts`, `deep_timelines._score_targets`) reads this kind and
  only a hand run wrote it before. One row per LISTED account: the report's
  output, ranked within community, not a second top-N. Canary accounts are
  left out - they are not real ids and the collector would try to fetch them.
- the freshness gauge, last, only after everything above succeeded, with a
  second sample `kma_v2_daily_text_trace_present` (1 or 0) in the same file.

## Where the text trace comes from (`--text-trace`)

- `modal` (the CLI default): the trace the `kma-text-trace` Modal job persisted
  for this window id. Nothing is embedded or searched here. If it is absent the
  rank runs WITHOUT text similarity, never with a locally built one - at full
  coverage that is the out-of-memory this mode exists to avoid - and the run
  says so: `text_trace=missing` on the `daily_runs` row, a WARNING in the
  journal, and a 0 on the text-trace gauge. This host needs no Modal
  credentials: the only coupling is R2.
- `local`: embed the window's missing posts and build the trace on this host.
- `none`: no text similarity, on purpose (`text_trace=skipped`).

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

from kma import canary, coord2, coord2_run, embed_window, text_trace, v2_daily, window, windowed

log = logging.getLogger(__name__)

TEXTFILE_DIR_ENV = "KMA_TEXTFILE_DIR"
GAUGE_FILE = "kma_v2_daily.prom"
GAUGE_METRIC = "kma_v2_daily_last_success_timestamp_seconds"
TEXT_TRACE_METRIC = "kma_v2_daily_text_trace_present"

# Measured 2026-10-10, see docs/plans/2026-10-10-cloud-v2-daily.md.
DEFAULT_DAYS = 30


def gauge_text(epoch_seconds: int, text_trace_present: bool | None = None) -> str:
    text = f"# TYPE {GAUGE_METRIC} gauge\n{GAUGE_METRIC} {epoch_seconds}\n"
    if text_trace_present is not None:
        text += f"# TYPE {TEXT_TRACE_METRIC} gauge\n{TEXT_TRACE_METRIC} {int(text_trace_present)}\n"
    return text


def write_gauge(directory: Path, now: float | None = None,
                text_trace_present: bool | None = None) -> Path:
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
            handle.write(gauge_text(int(time.time() if now is None else now), text_trace_present))
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
    embed: bool = True,
    text_trace_mode: str = "local",
) -> dict:
    """`text_trace_mode` defaults to `local`, what this function did before
    the mode existed; the CLI's default is `modal`."""
    if text_trace_mode not in text_trace.MODES:
        raise ValueError(f"text_trace_mode must be one of {text_trace.MODES}, got {text_trace_mode!r}")
    uri = local_uri(out_dir) if out_dir else None
    held: dict = {}
    # `ru_maxrss` only rises, so the value after each stage is the job's peak
    # so far: the stage where it jumps is the stage that sets the peak.
    rss_after: dict[str, float] = {}

    def ensure(w: window.Window) -> pd.DataFrame:
        if out_dir:
            held["manifest"] = window.build(w, con=con, write=False)
        else:
            held["manifest"] = window.ensure(w, con=con)
        rss_after["window"] = peak_rss_mb()
        return held["manifest"]

    def trace_reused(window_id: str) -> bool:
        present = v2_daily.text_trace_exists(con, window_id)
        if present:
            held["text_trace"] = text_trace.REUSED
        return present

    def trace_missing(window_id: str) -> dict:
        held["text_trace"] = text_trace.MISSING
        key = coord2_run.textsim_key(
            window_id, v2_daily.TEXT_THRESHOLD, overlap=coord2.TEXT_MIN_OVERLAP
        )
        log.warning(
            "TEXT TRACE MISSING for window %s: ranking WITHOUT text similarity. Expected %s, "
            "written by the kma-text-trace Modal job; check its logs and `modal app list`.",
            window_id, key,
        )
        return {"status": text_trace.MISSING, "expected_key": key}

    def trace_skipped(window_id: str) -> dict:
        held["text_trace"] = text_trace.SKIPPED
        log.info("text trace skipped on purpose (--text-trace none)")
        return {"status": text_trace.SKIPPED}

    def build_trace(window_id: str) -> dict:
        edges, info = text_trace.build(
            con, window_id, manifest=held["manifest"], persist=out_dir is None,
        )
        held["text_trace"] = text_trace.BUILT
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
        rss_after["textsim"] = peak_rss_mb()
        return info

    def rank(w: window.Window) -> dict:
        planted = canary.load_active(con) if with_canary else None
        if with_canary:
            log.info("canary: %d planted post(s) active", 0 if planted is None else len(planted))
        # A run without a trace must not claim a threshold it never applied:
        # `text_threshold` null on the row and on `kind=scores` is how a reader
        # tells a no-text rank from a 0.85 one.
        has_text = held["text_trace"] in (text_trace.BUILT, text_trace.REUSED)
        threshold = v2_daily.TEXT_THRESHOLD if has_text else None
        overlap = coord2.TEXT_MIN_OVERLAP if has_text else None
        extra = []
        if windowed_width:
            spec = windowed.DETECTION if windowed_width == windowed.DETECTION.width else windowed.SERIES
            extra.append(windowed_pass(spec))
        result = v2_daily.rank(
            con, w, min_entities=min_entities, extra_passes=extra, canary_posts=planted,
            persist=True, uri=uri, manifest=held["manifest"],
            text_edges=held.get("text_edges") if has_text else None,
            text_threshold=threshold, text_overlap=overlap,
            summary_extra={"text_trace": held["text_trace"], "text_trace_mode": text_trace_mode},
        )
        scores = scores_frame(result.tables["accounts"])
        if scores.empty:
            raise RuntimeError(
                f"window {w.id} listed no accounts; refusing to report a healthy run"
            )
        until = (w.end + timedelta(days=1)).isoformat()
        scores_key = coord2_run.persist(
            con, scores, snapshot=w.id, trace_sizes=result.trace_sizes,
            text_threshold=threshold, text_overlap=overlap,
            min_entities=min_entities, since=w.start.isoformat(), until=until, uri=uri,
        )
        rss_after["rank"] = peak_rss_mb()
        return {**result.summary, "keys": {**result.keys, "scores": scores_key},
                "scores_rows": len(scores)}

    local = text_trace_mode == "local"
    if embed and local and out_dir:
        log.info("embedding stage skipped: --out-dir writes nothing to R2")
    embed_stage = (
        (lambda w: embed_window.embed_missing(con, w)) if embed and local and not out_dir else None
    )
    textsim, exists = {
        "local": (build_trace, (lambda _: False) if out_dir else trace_reused),
        "modal": (trace_missing, trace_reused),
        "none": (trace_skipped, lambda _: False),
    }[text_trace_mode]

    out = v2_daily.orchestrate(
        end or None, days, textsim=textsim, rank_fn=rank, con=con, ensure=ensure,
        trace_exists=exists, embed=embed_stage,
    )
    out["text_trace"] = held["text_trace"]
    out["peak_rss_mb"] = peak_rss_mb()
    out["peak_rss_mb_after"] = rss_after
    if textfile_dir is not None:
        present = held["text_trace"] in (text_trace.BUILT, text_trace.REUSED)
        out["gauge"] = str(write_gauge(textfile_dir, text_trace_present=present))
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
    ap.add_argument("--text-trace", choices=text_trace.MODES, default="modal",
                    dest="text_trace_mode",
                    help="modal: reuse the trace the Modal job persisted, and rank without text "
                         "if it is absent; local: embed and build it on this host; none: no text")
    ap.add_argument("--no-embed", action="store_true",
                    help="with --text-trace local, skip the stage that embeds the window's "
                         "eligible posts with no vector")
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
        embed=not args.no_embed, text_trace_mode=args.text_trace_mode,
    )
    print(json.dumps(out, default=str, indent=2))


if __name__ == "__main__":
    main()

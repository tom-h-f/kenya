"""Recurring enrichment worker: keep R2 embeddings + labels current with the
always-on collector.

`embed_new` and `classify_new` are both incremental - they only touch posts not
yet embedded / labelled, then write one Parquet run to R2. So keeping Phase 2 up
to date is just re-running them on a loop: chew through any backlog in bounded
batches, then idle when nothing is pending. Text-similarity coordination and the
narrative scorecards read these prefixes, so this is what stops them going stale.

    python -m kma.enrich --once            # one bounded pass
    python -m kma.enrich --loop            # forever (deploy target)
    python -m kma.enrich --loop --coord-hours 0   # skip coordination refresh
    python -m kma.enrich --loop --limit 500

Runs wherever torch runs (MPS on the Mac, CPU on a server); the models pick the
device automatically.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import subprocess
import sys
import time

from kma.classify import classify_new
from kma.db import connect
from kma.hatespeech import score_new as score_hate_new
from kma.semantic import embed_new

log = logging.getLogger("kma.enrich")

# Per-pass cap keeps memory bounded on CPU-only / low-RAM hosts: a big first
# backlog is processed over several fast passes rather than one giant batch.
BATCH_LIMIT = int(os.getenv("ENRICH_LIMIT", "500"))
BATCH_SIZE = int(os.getenv("ENRICH_BATCH_SIZE", "64"))
# Short cooldown while catching up (there is more to do), long idle when caught
# up (wait for the collector to accumulate new posts).
BUSY_COOLDOWN_S = int(os.getenv("ENRICH_BUSY_COOLDOWN_S", "45"))
IDLE_MIN_S = int(os.getenv("ENRICH_IDLE_MIN_S", "600"))
IDLE_MAX_S = int(os.getenv("ENRICH_IDLE_MAX_S", "1200"))

# Coordination refresh. The collector's `adaptive.cluster_accounts` reads the
# LATEST persisted cluster run, so without a schedule the cluster half of the
# targeting loop freezes at whenever someone last ran it by hand. Much slower
# than the enrichment passes (a full projection + significance test over the
# corpus), hence its own timer rather than every cycle. 0 disables.
COORD_REFRESH_HOURS = float(os.getenv("COORD_REFRESH_HOURS", "6"))
# A failed refresh should not wait out the whole period. Seen in practice: pi0's
# first attempt died on a transient R2 timeout and then sat idle for hours,
# leaving the collector's cluster targeting frozen on a stale run.
COORD_RETRY_MINUTES = float(os.getenv("COORD_RETRY_MINUTES", "30"))


def run_once(
    limit: int | None = BATCH_LIMIT,
    batch_size: int = BATCH_SIZE,
    embed: bool = True,
    classify: bool = True,
    hate: bool = True,
) -> dict[str, int]:
    """One bounded enrichment pass. Returns {embedded, labelled, hate_scored}
    counts (only the passes that ran)."""
    con = connect()
    counts: dict[str, int] = {}
    if embed:
        counts["embedded"] = embed_new(con, limit=limit, batch_size=batch_size)
        log.info("embedded %d new post(s)", counts["embedded"])
    if classify:
        counts["labelled"] = classify_new(con, limit=limit, batch_size=batch_size)
        log.info("labelled %d new post(s)", counts["labelled"])
    if hate:
        counts["hate_scored"] = score_hate_new(con, limit=limit, batch_size=batch_size)
        log.info("hate-scored %d new post(s)", counts["hate_scored"])
    return counts


def _subprocess_pass(flags: list[str], limit: int | None, batch_size: int) -> int:
    """Run one `--once` pass in a fresh subprocess and return its processed
    count. Isolating each model set per process means only one is resident at a
    time - the peak-memory guard for RAM-tight hosts (the embedding mpnet, the
    two classifier transformers, and the ~2GB afro-xlmr hate model never
    coexist). `flags` disables the other passes so this one runs alone."""
    cmd = [sys.executable, "-m", "kma.enrich", "--once", *flags, "--batch-size", str(batch_size)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        log.error("pass %s failed (rc=%d): %s", flags, proc.returncode, proc.stderr[-500:])
        return 0
    # last line is "enriched: {'embedded': N}" / "{'labelled': N}"
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("enriched:")]
    if not line:
        return 0
    import ast

    try:
        return sum(ast.literal_eval(line[-1].split("enriched:", 1)[1].strip()).values())
    except Exception:
        return 0


def _coordination_pass() -> bool:
    """Rebuild and persist coordination clusters in a fresh subprocess.

    Subprocess for the same reason as the model passes: igraph plus the full
    embedding matrix is the heaviest thing this worker does, and it must not
    stay resident between cycles. Failure is logged, never fatal - a stale
    cluster run degrades targeting, a dead worker stops enrichment entirely."""
    cmd = [sys.executable, "-m", "kma.coordination_run", "--persist"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("coordination refresh failed (rc=%d): %s", proc.returncode, proc.stderr[-500:])
        return False
    log.info("coordination refreshed and persisted")
    return True


def run_loop(
    limit: int | None = BATCH_LIMIT,
    batch_size: int = BATCH_SIZE,
    embed: bool = True,
    classify: bool = True,
    hate: bool = True,
    isolate: bool = True,
    coord_hours: float = COORD_REFRESH_HOURS,
) -> None:
    """Forever: bounded passes back to back while there is a backlog, then a
    jittered idle once caught up. A per-pass failure is logged and retried, so a
    transient R2 / model hiccup never kills the worker.

    `isolate` (default) runs embed, classify and hate in separate subprocesses so
    their models never coexist in memory - keep it on for < ~6GB hosts.

    Every `coord_hours` the coordination clusters are rebuilt and persisted, so
    the collector's cluster-based targeting stays live instead of frozen at the
    last manual run."""
    cycle = 0
    next_coord = time.monotonic() if coord_hours > 0 else float("inf")
    while True:
        cycle += 1
        done = 0
        if time.monotonic() >= next_coord:
            # Scheduled before the enrichment passes so a long backlog cannot
            # keep postponing it indefinitely.
            ok = _coordination_pass()
            next_coord = time.monotonic() + (
                coord_hours * 3600 if ok else COORD_RETRY_MINUTES * 60
            )
        try:
            if isolate:
                if embed:
                    done += _subprocess_pass(["--no-classify", "--no-hate"], limit, batch_size)
                if classify:
                    done += _subprocess_pass(["--no-embed", "--no-hate"], limit, batch_size)
                if hate:
                    done += _subprocess_pass(["--no-embed", "--no-classify"], limit, batch_size)
            else:
                done = sum(run_once(limit, batch_size, embed, classify, hate).values())
        except Exception:
            log.exception("enrich cycle %d failed", cycle)
        wait = BUSY_COOLDOWN_S if done > 0 else random.uniform(IDLE_MIN_S, IDLE_MAX_S)
        log.info("cycle %d done (%d processed); next in %.0fs", cycle, done, wait)
        time.sleep(wait)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Keep R2 embeddings + labels current.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--loop", action="store_true", help="run forever (default)")
    mode.add_argument("--once", action="store_true", help="single bounded pass")
    ap.add_argument("--limit", type=int, default=BATCH_LIMIT, help="max posts per pass")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--no-embed", action="store_true", help="skip embeddings")
    ap.add_argument("--no-classify", action="store_true", help="skip sentiment/emotion")
    ap.add_argument("--no-hate", action="store_true", help="skip hate-speech scoring")
    ap.add_argument(
        "--no-isolate", action="store_true",
        help="load all models in one process (faster, needs ~6GB RAM)",
    )
    ap.add_argument(
        "--coord-hours", type=float, default=COORD_REFRESH_HOURS,
        help="hours between coordination cluster refreshes (0 disables)",
    )
    args = ap.parse_args()
    embed, classify, hate = not args.no_embed, not args.no_classify, not args.no_hate
    if args.once:
        counts = run_once(args.limit, args.batch_size, embed, classify, hate)
        print(f"enriched: {counts}")
    else:
        run_loop(
            args.limit, args.batch_size, embed, classify, hate,
            isolate=not args.no_isolate, coord_hours=args.coord_hours,
        )


if __name__ == "__main__":
    main()

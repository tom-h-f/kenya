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
from kma.incitement import score_new as score_incitement_new
from kma.semantic import embed_new

log = logging.getLogger("kma.enrich")

#: The enrichment passes, in the order they must run.
#:
#: `incitement` sits BEFORE `hate` deliberately. `hatespeech._measure_frame`
#: resolves `coded_suspect` at scoring time by reading the `incitement/` prefix,
#: and `measure.coded_suspect` returns a non-null False when no NLI score is
#: present. A post scored before its NLI exists is therefore written
#: `coded_suspect=False, flagged=hate_flag` permanently - and the collector's
#: hate-seeking reads exactly those persisted flags. Running the NLI first is
#: what makes the coded lens work on newly collected posts at all.
PASSES: tuple[str, ...] = ("embed", "classify", "incitement", "hate")

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
# Cluster scoring, on its own much slower timer. Measured 2026-08-14: 2,276s to
# score one run on an idle laptop, dominated by whole-prefix reads of `posts/`
# and `authors/` that no per-cluster filter avoids. Nothing in the collector
# loop consumes scorecards, so unlike the coordination refresh above, lag here
# costs a stale dashboard rather than stale targeting. 0 disables.
SCORECARD_REFRESH_HOURS = float(os.getenv("SCORECARD_REFRESH_HOURS", "24"))
# Adjudication - layer three, and the step that turns a cluster count into a
# finding. Slower again than scoring: it reads dossiers, which cost three fixed
# whole-prefix scans, and then a model call per candidate. Defaults to OFF,
# because without ANTHROPIC_API_KEY the pass can only fail; set the hours once a
# key is configured.
ADJUDICATE_REFRESH_HOURS = float(os.getenv("ADJUDICATE_REFRESH_HOURS", "0"))


def run_once(
    limit: int | None = BATCH_LIMIT,
    batch_size: int = BATCH_SIZE,
    embed: bool = True,
    classify: bool = True,
    hate: bool = True,
    incitement: bool = True,
) -> dict[str, int]:
    """One bounded enrichment pass. Returns
    {embedded, labelled, incitement_scored, hate_scored} for the passes that
    ran, in `PASSES` order."""
    con = connect()
    counts: dict[str, int] = {}
    if embed:
        counts["embedded"] = embed_new(con, limit=limit, batch_size=batch_size)
        log.info("embedded %d new post(s)", counts["embedded"])
    if classify:
        counts["labelled"] = classify_new(con, limit=limit, batch_size=batch_size)
        log.info("labelled %d new post(s)", counts["labelled"])
    if incitement:
        counts["incitement_scored"] = score_incitement_new(
            con, limit=limit, batch_size=batch_size
        )
        log.info("incitement-scored %d new post(s)", counts["incitement_scored"])
    if hate:
        counts["hate_scored"] = score_hate_new(con, limit=limit, batch_size=batch_size)
        log.info("hate-scored %d new post(s)", counts["hate_scored"])
    return counts


def _only(pass_name: str) -> list[str]:
    """Flags that disable every pass but this one.

    Derived rather than written out: with four passes the hand-maintained
    `--no-a --no-b --no-c` lists are combinatorial, and a missed flag silently
    runs two models in one process - which is the exact thing isolation exists
    to prevent."""
    return [f"--no-{p}" for p in PASSES if p != pass_name]


def _subprocess_pass(flags: list[str], limit: int | None, batch_size: int) -> int:
    """Run one `--once` pass in a fresh subprocess and return its processed
    count. Isolating each model set per process means only one is resident at a
    time - the peak-memory guard for RAM-tight hosts (the embedding mpnet, the
    two classifier transformers, the NLI model and the ~2GB afro-xlmr hate model
    never coexist). `flags` disables the other passes so this one runs alone."""
    cmd = [sys.executable, "-m", "kma.enrich", "--once", *flags, "--batch-size", str(batch_size)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    # The child logs to stderr (logging.basicConfig), so discarding it on success
    # threw away every per-pass progress line and left `docker compose logs`
    # showing nothing but a summed total.
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        log.error("pass %s failed (rc=%d): %s", flags, proc.returncode, proc.stderr[-500:])
        return 0
    # last line is "enriched: {'embedded': N}" / "{'labelled': N}"
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("enriched:")]
    if not line:
        # A pass that ran but printed nothing recognisable is a broken contract,
        # not an empty backlog. Returning 0 silently sent the loop into its
        # 10-20 minute idle branch with work still outstanding.
        log.error(
            "pass %s produced no 'enriched:' line; treating as a failure", flags
        )
        return 0
    import ast

    try:
        return sum(ast.literal_eval(line[-1].split("enriched:", 1)[1].strip()).values())
    except (ValueError, SyntaxError, AttributeError, TypeError):
        log.exception("pass %s emitted an unparseable count: %r", flags, line[-1])
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
    log.info("coordination refreshed and persisted: %s", _coordination_headline(proc.stdout))
    return True


def _scorecard_pass() -> bool:
    """Score the latest persisted coordination run in a fresh subprocess.

    Reads the run back out of R2 rather than rebuilding it, so this does not
    repeat the projection the coordination pass just did. Subprocess for the
    same reason as everything else here: it loads sklearn and a slice of the
    embedding matrix, and must not stay resident between cycles."""
    cmd = [sys.executable, "-m", "kma.scorecard_run", "--persist"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("scorecard refresh failed (rc=%d): %s", proc.returncode, proc.stderr[-500:])
        return False
    log.info("scorecards refreshed: %s", _scorecard_headline(proc.stdout))
    return True


def _adjudicate_pass() -> bool:
    """Judge the latest run's candidate clusters in a fresh subprocess."""
    cmd = [sys.executable, "-m", "kma.adjudicate_run", "--persist"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("adjudication failed (rc=%d): %s", proc.returncode, proc.stderr[-500:])
        return False
    log.info("adjudicated: %s", _adjudicate_headline(proc.stdout))
    return True


def _adjudicate_headline(stdout: str) -> str:
    import json

    try:
        out = json.loads(stdout)
        by_type = out.get("by_type") or {}
        types = ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())) or "none"
        return f"{out.get('n_verdicts')}/{out.get('n_candidates')} judged ({types})"
    except Exception:
        return "(no summary)"


def _scorecard_headline(stdout: str) -> str:
    import json

    try:
        out = json.loads(stdout)
        top = out.get("top_cluster") or {}
        return (
            f"{out.get('n_clusters')} clusters scored"
            + (f", top index {top['inauthenticity_index']:.2f}"
               f" (cluster {top['cluster_id']}, {top['size']} accounts)" if top else "")
        )
    except Exception:
        return "(no summary)"


def _coordination_headline(stdout: str) -> str:
    """The run's outcome counters as one log line.

    The metrics of record are in R2 (`coordination/kind=run_metrics`); this only
    puts them in container logs too, so `docker compose logs` shows whether a
    pass detected anything without a round trip to the bucket."""
    import json

    try:
        out = json.loads(stdout)
        return (
            f"{out['n_clusters']} clusters ({out['n_accounts']} accounts), "
            f"{out['n_corroborated_clusters']} corroborated "
            f"({out['n_corroborated_accounts']} accounts)"
        )
    except Exception:
        return "(no summary)"


def run_loop(
    limit: int | None = BATCH_LIMIT,
    batch_size: int = BATCH_SIZE,
    embed: bool = True,
    classify: bool = True,
    hate: bool = True,
    incitement: bool = True,
    isolate: bool = True,
    coord_hours: float = COORD_REFRESH_HOURS,
    scorecard_hours: float = SCORECARD_REFRESH_HOURS,
    adjudicate_hours: float = ADJUDICATE_REFRESH_HOURS,
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
    # Deliberately NOT due at startup, unlike the coordination refresh. Scoring
    # needs a persisted run to read, and on a first boot there may be none; more
    # practically, a restart should not spend its first ~40 minutes scoring
    # before any enrichment happens.
    next_scorecard = (
        time.monotonic() + scorecard_hours * 3600 if scorecard_hours > 0 else float("inf")
    )
    # Also not due at startup, and for a stronger reason than scoring: it needs
    # a scorecard run to have happened first, and on a fresh boot there is none.
    next_adjudicate = (
        time.monotonic() + adjudicate_hours * 3600 if adjudicate_hours > 0 else float("inf")
    )
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
        if time.monotonic() >= next_scorecard:
            ok = _scorecard_pass()
            next_scorecard = time.monotonic() + (
                scorecard_hours * 3600 if ok else COORD_RETRY_MINUTES * 60
            )
        if time.monotonic() >= next_adjudicate:
            ok = _adjudicate_pass()
            next_adjudicate = time.monotonic() + (
                adjudicate_hours * 3600 if ok else COORD_RETRY_MINUTES * 60
            )
        enabled = {
            "embed": embed, "classify": classify,
            "incitement": incitement, "hate": hate,
        }
        try:
            if isolate:
                for name in PASSES:
                    if enabled[name]:
                        done += _subprocess_pass(_only(name), limit, batch_size)
            else:
                done = sum(
                    run_once(
                        limit, batch_size, embed, classify, hate, incitement
                    ).values()
                )
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
        "--no-incitement", action="store_true",
        help="skip coded-incitement NLI (leaves coded_suspect unresolvable)",
    )
    ap.add_argument(
        "--no-isolate", action="store_true",
        help="load all models in one process (faster, needs ~6GB RAM)",
    )
    ap.add_argument(
        "--coord-hours", type=float, default=COORD_REFRESH_HOURS,
        help="hours between coordination cluster refreshes (0 disables)",
    )
    ap.add_argument(
        "--scorecard-hours", type=float, default=SCORECARD_REFRESH_HOURS,
        help="hours between cluster scoring passes (0 disables)",
    )
    ap.add_argument(
        "--adjudicate-hours", type=float, default=ADJUDICATE_REFRESH_HOURS,
        help="hours between adjudication passes (0 disables; needs an API key)",
    )
    args = ap.parse_args()
    embed, classify, hate = not args.no_embed, not args.no_classify, not args.no_hate
    incitement = not args.no_incitement
    if args.once:
        counts = run_once(args.limit, args.batch_size, embed, classify, hate, incitement)
        print(f"enriched: {counts}")
    else:
        run_loop(
            args.limit, args.batch_size, embed, classify, hate,
            incitement=incitement,
            isolate=not args.no_isolate, coord_hours=args.coord_hours,
            scorecard_hours=args.scorecard_hours,
            adjudicate_hours=args.adjudicate_hours,
        )


if __name__ == "__main__":
    main()

"""Recurring enrichment worker: keep R2 embeddings + labels current with the
always-on collector.

Every pass is incremental - it only touches posts not yet embedded / labelled /
scored, then writes one Parquet run to R2. So keeping Phase 2 up to date is just
re-running them on a loop: chew through any backlog in bounded batches, then
idle when nothing is pending. The v2 text trace and the collector's targeting
gates read these prefixes, so this is what stops them going stale.

Coordination detection is not run here: v1 was retired 2026-10-10 and v2 runs
on its own daily schedule (docs/plans/2026-10-10-retire-v1.md).

    python -m kma.enrich --once            # one bounded pass
    python -m kma.enrich --loop            # forever (deploy target)
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


def run_loop(
    limit: int | None = BATCH_LIMIT,
    batch_size: int = BATCH_SIZE,
    embed: bool = True,
    classify: bool = True,
    hate: bool = True,
    incitement: bool = True,
    isolate: bool = True,
) -> None:
    """Forever: bounded passes back to back while there is a backlog, then a
    jittered idle once caught up. A per-pass failure is logged and retried, so a
    transient R2 / model hiccup never kills the worker.

    `isolate` (default) runs embed, classify, incitement and hate in separate
    subprocesses so their models never coexist in memory - keep it on for
    < ~6GB hosts."""
    cycle = 0
    while True:
        cycle += 1
        done = 0
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
            isolate=not args.no_isolate,
        )


if __name__ == "__main__":
    main()

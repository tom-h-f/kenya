#!/usr/bin/env bash
# Runs inside a detached tmux session on mike-wsl, started by pi0's
# kenya-incitement-cron.sh (or by hand for the backlog). Takes a lock so it never
# overlaps another run of itself. The last line of the log is `exit N`, which is what
# the scheduler polls for.
#
#   run-incitement.sh RUN_ID
#
# KMA_INCITEMENT_BATCH, KMA_INCITEMENT_LIMIT and KMA_INCITEMENT_CHUNK_ROWS are passed
# through when set.
set -uo pipefail
BASE="$HOME/projects/kenya-incitement"
RUN_ID="${1:?usage: run-incitement.sh RUN_ID}"
LOG="$BASE/logs/$RUN_ID.log"
mkdir -p "$BASE/logs"
cd "$BASE"
exec 9>"$BASE/job.lock"
if ! flock -n 9; then
  echo "another incitement run holds $BASE/job.lock" >>"$LOG"
  echo "exit 5" >>"$LOG"
  exit 5
fi
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$BASE/repo/analysis/src" PYTHONUNBUFFERED=1
args=(run --manifest cache/manifest.json --cache cache --run-id "$RUN_ID"
      --status-file "cache/status-$RUN_ID.json" --batch "${KMA_INCITEMENT_BATCH:-32}")
[ -n "${KMA_INCITEMENT_LIMIT:-}" ] && args+=(--limit "$KMA_INCITEMENT_LIMIT")
[ -n "${KMA_INCITEMENT_CHUNK_ROWS:-}" ] && args+=(--chunk-rows "$KMA_INCITEMENT_CHUNK_ROWS")
uv run python -m kma.incitement_backlog "${args[@]}" >>"$LOG" 2>&1
code=$?
rm -f "$BASE/cache/manifest.json"
echo "exit $code" >>"$LOG"
exit "$code"

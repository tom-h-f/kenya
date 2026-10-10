#!/usr/bin/env bash
# Runs inside a detached tmux session on mike-wsl, started by pi0's
# kenya-topics-cron.sh. Takes a lock so it never overlaps another run of itself.
# The last line of the log is `exit N`, which is what the scheduler polls for.
#
#   run-topics.sh RUN_ID
#
# KMA_TOPICS_QUERY_BLOCK, KMA_TOPICS_LIMIT and KMA_TOPICS_CHUNK_ROWS are passed
# through when set.
set -uo pipefail
BASE="$HOME/projects/kenya-topics"
RUN_ID="${1:?usage: run-topics.sh RUN_ID}"
LOG="$BASE/logs/$RUN_ID.log"
mkdir -p "$BASE/logs"
cd "$BASE"
exec 9>"$BASE/job.lock"
if ! flock -n 9; then
  echo "another topics run holds $BASE/job.lock" >>"$LOG"
  echo "exit 5" >>"$LOG"
  exit 5
fi
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$BASE/repo/analysis/src" PYTHONUNBUFFERED=1
args=(run --manifest cache/manifest.json --cache cache --run-id "$RUN_ID"
      --status-file "cache/status-$RUN_ID.json" --query-block "${KMA_TOPICS_QUERY_BLOCK:-1024}")
[ -n "${KMA_TOPICS_LIMIT:-}" ] && args+=(--limit "$KMA_TOPICS_LIMIT")
[ -n "${KMA_TOPICS_CHUNK_ROWS:-}" ] && args+=(--chunk-rows "$KMA_TOPICS_CHUNK_ROWS")
uv run python -m kma.topics_assign "${args[@]}" >>"$LOG" 2>&1
code=$?
rm -f "$BASE/cache/manifest.json"
echo "exit $code" >>"$LOG"
exit "$code"

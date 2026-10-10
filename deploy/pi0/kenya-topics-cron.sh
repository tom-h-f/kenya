#!/usr/bin/env bash
# Twice-weekly topic assignment: pi0 cron -> mike-pc, retried until it works.
#
#   flock -n /tmp/kenya-topics.lock /home/tom/kenya-topics-cron.sh
#
# Per attempt: wake mike-pc if it is not answering, take an `awake` hold, sync
# the repo clone on mike-wsl, ship a signed-url manifest (mike-wsl holds no R2
# keys), run kma.topics_assign detached in tmux, poll its `exit N` line, then
# check R2 independently. Any failure backs off exponentially (cap 60 min) until
# DEADLINE seconds after the scheduled start, then gives up loudly. mike-pc is
# never shut down or put to sleep from here.
#
# Test hooks: KMA_TOPICS_FAIL_FIRST=n fails the first n attempts before any
# remote work; KMA_TOPICS_DEADLINE_S, KMA_TOPICS_BACKOFF_BASE_S,
# KMA_TOPICS_TEXTFILE_DIR, KMA_TOPICS_LIMIT, KMA_TOPICS_POLL_S and
# KMA_TOPICS_NO_WAKE override defaults.
set -uo pipefail

HOME_DIR=/home/tom
R2="$HOME_DIR/kenya_topics_r2.py"
TEXTFILE_DIR="${KMA_TOPICS_TEXTFILE_DIR:-/var/lib/node_exporter/textfile}"
PROM="$TEXTFILE_DIR/kma_topics.prom"
DEADLINE_S="${KMA_TOPICS_DEADLINE_S:-72000}"
BACKOFF_BASE_S="${KMA_TOPICS_BACKOFF_BASE_S:-60}"
BACKOFF_CAP_S="${KMA_TOPICS_BACKOFF_CAP_S:-3600}"
ATTEMPT_TIMEOUT_S="${KMA_TOPICS_ATTEMPT_TIMEOUT_S:-14400}"
AWAKE_FOR="${KMA_TOPICS_AWAKE_FOR:-4h}"
REF="${KMA_TOPICS_REF:-master}"
FAIL_FIRST="${KMA_TOPICS_FAIL_FIRST:-0}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 mike-wsl)

mkdir -p "$HOME_DIR/cron-logs"
LOG="$HOME_DIR/cron-logs/kenya-topics-$(date -u +%Y%m%d).log"
exec >>"$LOG" 2>&1

now() { date +%s; }
log() { echo "[$(date -u +%FT%TZ)] $*"; }
START=$(now)
SCHEDULED_DEADLINE=$((START + DEADLINE_S))

prev_value() { # metric name
  if [ -r "$PROM" ]; then awk -v m="$1" '$1==m {print $2}' "$PROM"; fi
}

write_prom() { # attempt_ts success_ts assigned ok
  local tmp
  tmp=$(mktemp /tmp/kma_topics.XXXXXX)
  cat >"$tmp" <<PROM_EOF
# HELP kma_topics_last_success_timestamp_seconds Unix time of the last topics run that passed its completion marker and R2 check.
# TYPE kma_topics_last_success_timestamp_seconds gauge
kma_topics_last_success_timestamp_seconds $2
# HELP kma_topics_last_attempt_timestamp_seconds Unix time of the last topics attempt, successful or not.
# TYPE kma_topics_last_attempt_timestamp_seconds gauge
kma_topics_last_attempt_timestamp_seconds $1
# HELP kma_topics_assigned_posts Rows assigned by the last successful topics run.
# TYPE kma_topics_assigned_posts gauge
kma_topics_assigned_posts $3
# HELP kma_topics_last_attempt_ok 1 if the last attempt succeeded, 0 if the run gave up or is still retrying.
# TYPE kma_topics_last_attempt_ok gauge
kma_topics_last_attempt_ok $4
PROM_EOF
  chmod 0644 "$tmp"
  if [ -w "$TEXTFILE_DIR" ]; then
    mv "$tmp" "$PROM"
  else
    # install writes a temp name and renames, so node_exporter never reads a partial file
    sudo -n install -m 0644 -o root -g root "$tmp" "$PROM" && rm -f "$tmp"
  fi
}

LAST_OK_TS=$(prev_value kma_topics_last_success_timestamp_seconds); LAST_OK_TS=${LAST_OK_TS:-0}
LAST_ASSIGNED=$(prev_value kma_topics_assigned_posts); LAST_ASSIGNED=${LAST_ASSIGNED:-0}
TOTAL_ASSIGNED=0
WE_SET_AWAKE=""
QUERY_BLOCK=""

log "=== kenya-topics start (deadline ${DEADLINE_S}s, ref $REF) ==="
write_prom "$START" "$LAST_OK_TS" "$LAST_ASSIGNED" 0

mike_up() { "${SSH[@]}" true 2>/dev/null; }

ensure_up() {
  mike_up && return 0
  [ -n "${KMA_TOPICS_NO_WAKE:-}" ] && return 1
  local w i
  for w in 1 2 3 4; do
    log "mike-wsl not answering: wake $w/4"
    wake-lan mike-pc || true
    for i in $(seq 1 15); do
      sleep 5
      mike_up && { log "mike-wsl up"; return 0; }
    done
  done
  return 1
}

remote() { "${SSH[@]}" bash -s -- "$@"; }

gpu_busy() {
  local used
  used=$(remote <<'REMOTE_EOF'
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
REMOTE_EOF
) || return 2
  [ -n "$used" ] && [ "$used" -gt 2000 ]
}

adopt_active() { # prints the run id of a still-running run of ours, if any
  remote <<'REMOTE_EOF'
B="$HOME/projects/kenya-topics"
[ -f "$B/active" ] || exit 0
rid=$(cat "$B/active")
if tmux has-session -t "kt-$rid" 2>/dev/null; then echo "$rid"; fi
REMOTE_EOF
}

attempt() {
  local n=$1 rid active
  if [ "$FAIL_FIRST" -ge "$n" ]; then log "forced failure (KMA_TOPICS_FAIL_FIRST=$FAIL_FIRST)"; return 1; fi
  ensure_up || { log "could not reach mike-wsl"; return 1; }

  active=$(adopt_active) || { log "adopt check failed"; return 1; }
  if [ -n "$active" ]; then
    rid=$active
    log "adopting still-running run $rid"
  else
    if gpu_busy; then log "GPU busy (another job); not starting"; return 1; fi
    rid=$(date -u +%Y%m%dT%H%M%SZ)
    local flag_before
    flag_before=$(remote <<'REMOTE_EOF'
awake status 2>/dev/null | head -1
REMOTE_EOF
)
    if [ "$flag_before" = "awake: off" ]; then
      remote <<REMOTE_EOF || return 1
awake on $AWAKE_FOR >/dev/null
cat "/mnt/c/Users/Tom French/.wsl-host-agent/awake" > "\$HOME/projects/kenya-topics/our-hold"
REMOTE_EOF
      WE_SET_AWAKE=1
      log "awake on $AWAKE_FOR"
    else
      log "awake already held by someone else ($flag_before); leaving it"
    fi
    log "sync repo to $REF"
    remote <<REMOTE_EOF || { log "sync failed"; return 1; }
set -e
B="\$HOME/projects/kenya-topics"
export PATH="\$HOME/.local/bin:\$PATH"
cd "\$B/repo"
git fetch -q origin
git checkout -q -B "$REF" "origin/$REF"
cp deploy/mike/pyproject.toml "\$B/pyproject.toml"
cd "\$B"
uv sync -q
REMOTE_EOF
    log "ship manifest for $rid"
    # umask on both ends: the file holds signed urls valid for hours
    (umask 077; python3 "$R2" manifest "$rid" --chunks 200 --hours 10) \
      | "${SSH[@]}" 'umask 077; mkdir -p "$HOME/projects/kenya-topics/cache"; cat > "$HOME/projects/kenya-topics/cache/manifest.json"' \
      || { log "manifest failed"; return 1; }
    local envs=""
    [ -n "$QUERY_BLOCK" ] && envs="KMA_TOPICS_QUERY_BLOCK=$QUERY_BLOCK"
    [ -n "${KMA_TOPICS_LIMIT:-}" ] && envs="$envs KMA_TOPICS_LIMIT=$KMA_TOPICS_LIMIT"
    remote <<REMOTE_EOF || { log "tmux start failed"; return 1; }
B="\$HOME/projects/kenya-topics"
echo "$rid" > "\$B/active"
tmux new-session -d -s "kt-$rid" "env $envs \$B/repo/deploy/mike/run-topics.sh $rid"
REMOTE_EOF
    log "started tmux kt-$rid"
  fi

  local t0 code
  t0=$(now)
  while :; do
    sleep "${KMA_TOPICS_POLL_S:-30}"
    code=$(remote <<REMOTE_EOF 2>/dev/null
tail -n 1 "\$HOME/projects/kenya-topics/logs/$rid.log" 2>/dev/null | sed -n 's/^exit \([0-9]*\)\$/\1/p'
REMOTE_EOF
) || code=""
    [ -n "$code" ] && break
    if [ $(( $(now) - t0 )) -gt "$ATTEMPT_TIMEOUT_S" ]; then
      log "run $rid still going after ${ATTEMPT_TIMEOUT_S}s; leaving it, will adopt next attempt"
      return 1
    fi
  done
  log "run $rid exited $code"
  remote <<REMOTE_EOF || true
tail -n 6 "\$HOME/projects/kenya-topics/logs/$rid.log" | cut -c1-300
REMOTE_EOF
  remote <<REMOTE_EOF >/dev/null 2>&1 || true
B="\$HOME/projects/kenya-topics"
[ "\$(cat "\$B/active" 2>/dev/null)" = "$rid" ] && rm -f "\$B/active"
REMOTE_EOF
  case "$code" in
    0) ;;
    4) QUERY_BLOCK=256; log "GPU OOM: next attempt uses query block 256"; return 1 ;;
    *) return 1 ;;
  esac
  local res got
  res=$(python3 "$R2" verify "$rid") || { log "R2 verify failed: $res"; return 1; }
  log "R2 verify: $res"
  got=$(printf '%s' "$res" | python3 -c 'import json,sys;print(json.load(sys.stdin)["assigned"])')
  TOTAL_ASSIGNED=$((TOTAL_ASSIGNED + got))
  return 0
}

release_awake() {
  [ -n "$WE_SET_AWAKE" ] || return 0
  mike_up || return 0
  remote <<'REMOTE_EOF' || true
B="$HOME/projects/kenya-topics"
flag="/mnt/c/Users/Tom French/.wsl-host-agent/awake"
if [ -f "$B/active" ] && tmux has-session -t "kt-$(cat "$B/active")" 2>/dev/null; then
  echo "run still active: leaving the timed hold to expire"; exit 0
fi
if [ -f "$flag" ] && [ "$(cat "$flag")" = "$(cat "$B/our-hold" 2>/dev/null)" ]; then
  awake off
else
  echo "awake flag was changed by someone else: not touching it"
fi
REMOTE_EOF
}

n=0
delay=$BACKOFF_BASE_S
while :; do
  n=$((n + 1))
  write_prom "$(now)" "$LAST_OK_TS" "$LAST_ASSIGNED" 0
  log "--- attempt $n ---"
  if attempt "$n"; then
    release_awake
    write_prom "$(now)" "$(now)" "$TOTAL_ASSIGNED" 1
    log "=== kenya-topics SUCCESS after $n attempt(s), assigned $TOTAL_ASSIGNED ==="
    exit 0
  fi
  left=$((SCHEDULED_DEADLINE - $(now)))
  if [ "$left" -le 0 ] || [ "$delay" -ge "$left" ]; then
    release_awake
    write_prom "$(now)" "$LAST_OK_TS" "$LAST_ASSIGNED" 0
    log "=== kenya-topics GAVE UP after $n attempt(s), ${DEADLINE_S}s deadline; last success ts $LAST_OK_TS ==="
    exit 1
  fi
  log "attempt $n failed; retry in ${delay}s (${left}s left)"
  sleep "$delay"
  delay=$((delay * 2)); [ "$delay" -gt "$BACKOFF_CAP_S" ] && delay=$BACKOFF_CAP_S
done

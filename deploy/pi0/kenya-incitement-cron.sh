#!/usr/bin/env bash
# Twice-weekly incitement scoring: pi0 cron -> mike-pc, retried until it works.
#
#   flock -n /tmp/kenya-incitement.lock /home/tom/kenya-incitement-cron.sh
#
# Per attempt: wake mike-pc if it is not answering, take an `awake` hold, sync
# the repo clone on mike-wsl, ship a signed-url manifest (mike-wsl holds no R2
# keys), run kma.incitement_backlog detached in tmux, poll its `exit N` line, then
# check R2 independently. Any failure backs off exponentially (cap 60 min) until
# DEADLINE seconds after the scheduled start, then gives up loudly. mike-pc is
# never shut down or put to sleep from here.
#
# Test hooks: KMA_INCITEMENT_FAIL_FIRST=n fails the first n attempts before any
# remote work; KMA_INCITEMENT_DEADLINE_S, KMA_INCITEMENT_BACKOFF_BASE_S,
# KMA_INCITEMENT_TEXTFILE_DIR, KMA_INCITEMENT_LIMIT, KMA_INCITEMENT_POLL_S and
# KMA_INCITEMENT_NO_WAKE override defaults.
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin

HOME_DIR=/home/tom
# Signing helpers: kenya_topics_r2.py (already on pi0) provides SigV4 list and presign;
# the manifest and verify logic for this job is the python block below, so this
# wrapper is the only file the incitement job adds to pi0.
TEXTFILE_DIR="${KMA_INCITEMENT_TEXTFILE_DIR:-/var/lib/node_exporter/textfile}"
PROM="$TEXTFILE_DIR/kma_incitement.prom"
DEADLINE_S="${KMA_INCITEMENT_DEADLINE_S:-72000}"
BACKOFF_BASE_S="${KMA_INCITEMENT_BACKOFF_BASE_S:-60}"
BACKOFF_CAP_S="${KMA_INCITEMENT_BACKOFF_CAP_S:-3600}"
ATTEMPT_TIMEOUT_S="${KMA_INCITEMENT_ATTEMPT_TIMEOUT_S:-6600}"
AWAKE_FOR="${KMA_INCITEMENT_AWAKE_FOR:-2h}"
REF="${KMA_INCITEMENT_REF:-master}"
FAIL_FIRST="${KMA_INCITEMENT_FAIL_FIRST:-0}"
# pi0 has a key for the Windows side of mike-pc only, so WSL is reached through
# `wsl -e bash -s` and scripts go over stdin (PowerShell mangles quoted args).
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 mike-pc "wsl -d Debian -u tom -e bash -s")

mkdir -p "$HOME_DIR/cron-logs"
LOG="$HOME_DIR/cron-logs/kenya-incitement-$(date -u +%Y%m%d).log"
exec >>"$LOG" 2>&1

now() { date +%s; }
log() { echo "[$(date -u +%FT%TZ)] $*"; }
START=$(now)
SCHEDULED_DEADLINE=$((START + DEADLINE_S))

prev_value() { # metric name
  if [ -r "$PROM" ]; then awk -v m="$1" '$1==m {print $2}' "$PROM"; fi
}

write_prom() { # attempt_ts success_ts scored ok pending_at_start
  local tmp
  tmp=$(mktemp /tmp/kma_incitement.XXXXXX)
  cat >"$tmp" <<PROM_EOF
# HELP kma_incitement_last_success_timestamp_seconds Unix time of the last incitement run that passed its completion marker and R2 check.
# TYPE kma_incitement_last_success_timestamp_seconds gauge
kma_incitement_last_success_timestamp_seconds $2
# HELP kma_incitement_last_attempt_timestamp_seconds Unix time of the last incitement attempt, successful or not.
# TYPE kma_incitement_last_attempt_timestamp_seconds gauge
kma_incitement_last_attempt_timestamp_seconds $1
# HELP kma_incitement_scored_posts Rows scored by the last successful incitement run.
# TYPE kma_incitement_scored_posts gauge
kma_incitement_scored_posts $3
# HELP kma_incitement_pending_at_start_posts Unscored posts with text when the last successful run began.
# TYPE kma_incitement_pending_at_start_posts gauge
kma_incitement_pending_at_start_posts $5
# HELP kma_incitement_last_attempt_ok 1 if the last attempt succeeded, 0 if the run gave up or is still retrying.
# TYPE kma_incitement_last_attempt_ok gauge
kma_incitement_last_attempt_ok $4
PROM_EOF
  chmod 0644 "$tmp"
  if [ -w "$TEXTFILE_DIR" ]; then
    mv "$tmp" "$PROM"
  else
    # install writes a temp name and renames, so node_exporter never reads a partial file
    sudo -n install -m 0644 -o root -g root "$tmp" "$PROM" && rm -f "$tmp"
  fi
}

LAST_OK_TS=$(prev_value kma_incitement_last_success_timestamp_seconds); LAST_OK_TS=${LAST_OK_TS:-0}
LAST_ASSIGNED=$(prev_value kma_incitement_scored_posts); LAST_ASSIGNED=${LAST_ASSIGNED:-0}
LAST_PENDING=$(prev_value kma_incitement_pending_at_start_posts); LAST_PENDING=${LAST_PENDING:-0}
TOTAL_PENDING=0
TOTAL_ASSIGNED=0
WE_SET_AWAKE=""
BATCH=""

log "=== kenya-incitement start (deadline ${DEADLINE_S}s, ref $REF) ==="
write_prom "$START" "$LAST_OK_TS" "$LAST_ASSIGNED" 0 "$LAST_PENDING" "$LAST_PENDING"

r2py() { # manifest RUN_ID --chunks N --hours H | verify RUN_ID
  python3 - "$@" <<'PY_EOF'
import json, sys, urllib.request, datetime
sys.path.insert(0, "/home/tom")
import kenya_topics_r2 as K

PLATFORM = "x"
RELEVANCE = "kenya-relevance-afroxlmr-2026-09-13"
READ = (f"posts/platform={PLATFORM}/", f"incitement/platform={PLATFORM}/",
        f"relevance/platform={PLATFORM}/model={RELEVANCE}/")


def chunk_keys(run_id, n):
    day = f"{run_id[0:4]}-{run_id[4:6]}-{run_id[6:8]}"
    return [f"incitement/platform={PLATFORM}/dt={day}/run={run_id}-c{i:04d}.parquet" for i in range(n)]


def marker_key(run_id):
    return f"incitement/_backlog/platform={PLATFORM}/run={run_id}.json"


def manifest(run_id, chunks, hours):
    seconds = int(hours * 3600)
    objects = {}
    for prefix in READ:
        for key, size in K.list_prefix(prefix):
            objects[key] = {"size": size, "url": K.presign("GET", key, seconds)}
    keys = chunk_keys(run_id, chunks) + [marker_key(run_id)]
    puts = {k: {"put": K.presign("PUT", k, seconds), "get": K.presign("GET", k, seconds)} for k in keys}
    return {"version": 1, "run_id": run_id, "objects": objects, "puts": puts,
            "expires_at": (K._now() + datetime.timedelta(seconds=seconds)).isoformat()}


def verify(run_id):
    marker = json.load(urllib.request.urlopen(K.presign("GET", marker_key(run_id), 300), timeout=60))
    listed = dict(K.list_prefix(f"incitement/platform={PLATFORM}/"))
    problems = []
    if marker.get("status") != "complete":
        problems.append(f"marker status {marker.get('status')!r}")
    if marker.get("readback_rows") != marker.get("scored") or marker.get("readback_distinct") != marker.get("scored"):
        problems.append("readback counts do not match scored")
    if sum(c["rows"] for c in marker.get("chunks", [])) != marker.get("scored"):
        problems.append("chunk rows do not sum to scored")
    for c in marker.get("chunks", []):
        if listed.get(c["key"]) != c["bytes"]:
            problems.append(f"{c['key']}: listing {listed.get(c['key'])} vs marker {c['bytes']}")
    out = {"ok": not problems, "problems": problems[:5], "scored": marker.get("scored"),
           "pending_at_start": marker.get("pending_at_start"), "chunks": len(marker.get("chunks", [])),
           "peak_vram_gib": marker.get("peak_vram_gib"), "elapsed_s": marker.get("elapsed_s")}
    print(json.dumps(out))
    return 0 if out["ok"] else 1


cmd, run_id = sys.argv[1], sys.argv[2]
args = dict(zip(sys.argv[3::2], sys.argv[4::2]))
if cmd == "manifest":
    json.dump(manifest(run_id, int(args.get("--chunks", 400)), float(args.get("--hours", 36))), sys.stdout)
elif cmd == "verify":
    sys.exit(verify(run_id))
else:
    sys.exit(f"unknown command {cmd}")
PY_EOF
}

mike_up() { [ "$("${SSH[@]}" <<<'echo up' 2>/dev/null | tr -d '\r')" = up ]; }

ensure_up() {
  mike_up && return 0
  [ -n "${KMA_INCITEMENT_NO_WAKE:-}" ] && return 1
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

remote() { "${SSH[@]}" | tr -d '\r'; }

gpu_busy() {
  local used
  used=$(remote <<'REMOTE_EOF'
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
REMOTE_EOF
) || return 2
  [ -n "$used" ] && [ "$used" -gt 2000 ]
}

topics_running() { # the Tue/Fri topics job holds a kt-* tmux session on mike-wsl
  local n
  n=$(remote <<'REMOTE_EOF'
tmux ls 2>/dev/null | grep -c '^kt-' || true
REMOTE_EOF
) || return 2
  [ "${n:-0}" -gt 0 ]
}

adopt_active() { # prints the run id of a still-running run of ours, if any
  remote <<'REMOTE_EOF'
B="$HOME/projects/kenya-incitement"
[ -f "$B/active" ] || exit 0
rid=$(cat "$B/active")
if tmux has-session -t "ki-$rid" 2>/dev/null; then echo "$rid"; fi
REMOTE_EOF
}

take_awake() {
  local flag_before
  flag_before=$(remote <<'REMOTE_EOF'
awake status 2>/dev/null | head -1
REMOTE_EOF
)
  if [ "$flag_before" = "awake: off" ]; then
    remote <<REMOTE_EOF >/dev/null
awake on $AWAKE_FOR
cat "/mnt/c/Users/Tom French/.wsl-host-agent/awake" > "\$HOME/projects/kenya-incitement/our-hold"
REMOTE_EOF
    WE_SET_AWAKE=1
    log "awake on $AWAKE_FOR"
  else
    log "awake already held by someone else ($flag_before); leaving it"
  fi
}

attempt() {
  local n=$1 rid active out
  if [ "$FAIL_FIRST" -ge "$n" ]; then log "forced failure (KMA_INCITEMENT_FAIL_FIRST=$FAIL_FIRST)"; return 1; fi
  ensure_up || { log "could not reach mike-wsl"; return 1; }

  active=$(adopt_active) || { log "adopt check failed"; return 1; }
  take_awake || return 1
  if [ -n "$active" ]; then
    rid=$active
    log "adopting still-running run $rid"
  else
    if topics_running; then log "topics job running on mike-wsl; not starting"; return 1; fi
    if gpu_busy; then log "GPU busy (another job); not starting"; return 1; fi
    rid=$(date -u +%Y%m%dT%H%M%SZ)
    log "sync repo to $REF"
    out=$(remote <<REMOTE_EOF
set -e
B="\$HOME/projects/kenya-incitement"
export PATH="\$HOME/.local/bin:\$PATH"
cd "\$B/repo"
git fetch -q origin
git checkout -q -B "$REF" "origin/$REF"
cp deploy/mike/incitement/pyproject.toml "\$B/pyproject.toml"
cd "\$B"
uv sync -q
echo SYNC_OK
REMOTE_EOF
)
    [ "${out##*$'\n'}" = SYNC_OK ] || { log "sync failed: $out"; return 1; }
    log "ship manifest for $rid"
    # umask on both ends: the file holds signed urls valid for hours
    out=$({ printf '%s\n' 'umask 077' 'mkdir -p "$HOME/projects/kenya-incitement/cache"' \
        "base64 -d > \"\$HOME/projects/kenya-incitement/cache/manifest.json\" <<'B64_EOF'"
      r2py manifest "$rid" --chunks 400 --hours 36 | base64 -w0; echo
      printf '%s\n' B64_EOF 'wc -c < "$HOME/projects/kenya-incitement/cache/manifest.json"'; } | remote)
    case "${out##*$'\n'}" in ''|*[!0-9]*|0) log "manifest failed: $out"; return 1 ;; esac
    local envs=""
    [ -n "$BATCH" ] && envs="KMA_INCITEMENT_BATCH=$BATCH"
    [ -n "${KMA_INCITEMENT_LIMIT:-}" ] && envs="$envs KMA_INCITEMENT_LIMIT=$KMA_INCITEMENT_LIMIT"
    out=$(remote <<REMOTE_EOF
B="\$HOME/projects/kenya-incitement"
echo "$rid" > "\$B/active"
tmux new-session -d -s "ki-$rid" "env $envs \$B/repo/deploy/mike/run-incitement.sh $rid" && echo STARTED
REMOTE_EOF
)
    [ "${out##*$'\n'}" = STARTED ] || { log "tmux start failed: $out"; return 1; }
    log "started tmux ki-$rid"
  fi

  local t0 code
  t0=$(now)
  while :; do
    sleep "${KMA_INCITEMENT_POLL_S:-30}"
    code=$(remote <<REMOTE_EOF 2>/dev/null
tail -n 1 "\$HOME/projects/kenya-incitement/logs/$rid.log" 2>/dev/null | sed -n 's/^exit \([0-9]*\)\$/\1/p'
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
tail -n 6 "\$HOME/projects/kenya-incitement/logs/$rid.log" | cut -c1-300
REMOTE_EOF
  remote <<REMOTE_EOF >/dev/null 2>&1 || true
B="\$HOME/projects/kenya-incitement"
[ "\$(cat "\$B/active" 2>/dev/null)" = "$rid" ] && rm -f "\$B/active"
REMOTE_EOF
  case "$code" in
    0) ;;
    4) BATCH=16; log "GPU OOM: next attempt uses batch 16"; return 1 ;;
    *) return 1 ;;
  esac
  local res got
  res=$(r2py verify "$rid") || { log "R2 verify failed: $res"; return 1; }
  log "R2 verify: $res"
  got=$(printf '%s' "$res" | python3 -c 'import json,sys;print(json.load(sys.stdin)["scored"])')
  TOTAL_ASSIGNED=$((TOTAL_ASSIGNED + got))
  TOTAL_PENDING=$(printf '%s' "$res" | python3 -c 'import json,sys;print(json.load(sys.stdin)["pending_at_start"])')
  return 0
}

release_awake() {
  [ -n "$WE_SET_AWAKE" ] || return 0
  mike_up || return 0
  remote <<'REMOTE_EOF' || true
B="$HOME/projects/kenya-incitement"
flag="/mnt/c/Users/Tom French/.wsl-host-agent/awake"
if [ -f "$B/active" ] && tmux has-session -t "ki-$(cat "$B/active")" 2>/dev/null; then
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
  write_prom "$(now)" "$LAST_OK_TS" "$LAST_ASSIGNED" 0 "$LAST_PENDING"
  log "--- attempt $n ---"
  if attempt "$n"; then
    release_awake
    write_prom "$(now)" "$(now)" "$TOTAL_ASSIGNED" 1 "$TOTAL_PENDING"
    log "=== kenya-incitement SUCCESS after $n attempt(s), scored $TOTAL_ASSIGNED ==="
    exit 0
  fi
  left=$((SCHEDULED_DEADLINE - $(now)))
  if [ "$left" -le 0 ] || [ "$delay" -ge "$left" ]; then
    release_awake
    write_prom "$(now)" "$LAST_OK_TS" "$LAST_ASSIGNED" 0 "$LAST_PENDING"
    log "=== kenya-incitement GAVE UP after $n attempt(s), ${DEADLINE_S}s deadline; last success ts $LAST_OK_TS ==="
    exit 1
  fi
  log "attempt $n failed; retry in ${delay}s (${left}s left)"
  sleep "$delay"
  delay=$((delay * 2)); [ "$delay" -gt "$BACKOFF_CAP_S" ] && delay=$BACKOFF_CAP_S
done

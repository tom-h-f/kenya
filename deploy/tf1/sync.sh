#!/usr/bin/env bash
# Deploy master (or any ref) of the public repo to the tf1 enrich worker.
#
# Runs ON tf1, fed over ssh so the script itself always comes from the repo:
#   ssh tf1 'bash -s -- --dry-run' < deploy/tf1/sync.sh
#   ssh tf1 'bash -s -- --ref origin/master' < deploy/tf1/sync.sh
#   ssh tf1 'bash -s -- --rollback ~/backups/kma-tf1-<stamp>.tgz' < deploy/tf1/sync.sh
#
# Only the compose project `analysis`, service `enrich`, is ever touched.
# See deploy/tf1/README.md.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/tom-h-f/kenya.git}"
CLONE="${CLONE:-$HOME/kenya-monitor-2027-git}"
TREE="${TREE:-$HOME/kenya-monitor-2027}"
BACKUPS="${BACKUPS:-$HOME/backups}"
SERVICE=enrich
CONTAINER=analysis-enrich-1
IMAGE=analysis-enrich

REF="origin/master"
DRY_RUN=0
ROLLBACK=""
WAIT_IDLE_MAX_S="${WAIT_IDLE_MAX_S:-5400}"
VERIFY_WAIT_S="${VERIFY_WAIT_S:-180}"

usage() {
  echo "usage: sync.sh [--dry-run] [--ref REF] [--rollback TGZ]" >&2
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --ref) REF="${2:?--ref needs a value}"; shift ;;
    --rollback) ROLLBACK="${2:?--rollback needs a path}"; shift ;;
    *) usage ;;
  esac
  shift
done

log() { printf '%s sync: %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

compose() { (cd "$TREE/analysis" && docker compose "$@"); }

cid() { docker inspect -f '{{.Id}}' "$CONTAINER"; }

oom_kills() {
  local f
  f="/sys/fs/cgroup/system.slice/docker-$(cid).scope/memory.events"
  awk '$1 == "oom_kill" {print $2}' "$f"
}

pass_running() {
  local state procs
  state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo missing)"
  [ "$state" = "running" ] || return 1
  procs="$(docker top "$CONTAINER" -o args)" || return 0
  grep -q 'kma.enrich --once' <<<"$procs"
}

wait_between_passes() {
  local waited=0
  log "waiting for a gap between enrich passes (max ${WAIT_IDLE_MAX_S}s)"
  while pass_running; do
    [ "$waited" -ge "$WAIT_IDLE_MAX_S" ] && die "no idle gap in ${WAIT_IDLE_MAX_S}s, nothing recreated"
    sleep 1
    waited=$((waited + 1))
  done
  log "idle gap after ${waited}s"
}

verify() {
  local want_sha="$1" deadline state restarts oom got_sha cmd
  deadline=$(( $(date +%s) + VERIFY_WAIT_S ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo missing)"
    restarts="$(docker inspect -f '{{.RestartCount}}' "$CONTAINER" 2>/dev/null || echo -1)"
    if [ "$state" != "running" ] || [ "$restarts" != "0" ]; then
      log "verify: state=$state restarts=$restarts"
      return 1
    fi
    sleep 5
  done
  got_sha="$(docker exec "$CONTAINER" printenv GIT_SHA)"
  [ "$got_sha" = "$want_sha" ] || { log "verify: GIT_SHA $got_sha != $want_sha"; return 1; }
  cmd="$(docker inspect -f '{{json .Config.Cmd}}' "$CONTAINER")"
  log "verify: cmd=$cmd mem_limit=$(docker inspect -f '{{.HostConfig.Memory}}' "$CONTAINER")"
  oom="$(oom_kills)"
  [ "$oom" = "0" ] || { log "verify: oom_kill=$oom"; return 1; }
  if docker logs --since "${VERIFY_WAIT_S}s" "$CONTAINER" 2>&1 | grep -E 'Traceback|ERROR|CRITICAL' | head -5 | grep .; then
    log "verify: errors in log"
    return 1
  fi
  log "verify: ok state=running restarts=0 oom_kill=0 sha=$got_sha"
}

do_rollback() {
  local tgz="$1" prev
  [ -f "$tgz" ] || die "backup not found: $tgz"
  log "rollback from $tgz"
  rm -rf "$TREE/analysis/src" "$TREE/analysis/tests" "$TREE/analysis/notebooks"
  rm -f "$TREE/analysis/docker-compose.override.yaml"
  tar -xzf "$tgz" -C "$(dirname "$TREE")"
  prev="$(cat "$tgz.image" 2>/dev/null || true)"
  if [ -n "$prev" ] && docker image inspect "$prev" >/dev/null 2>&1; then
    docker tag "$prev" "$IMAGE:latest"
    compose up -d --force-recreate --no-build "$SERVICE"
  else
    log "previous image tag missing, rebuilding from restored tree"
    compose up -d --build --force-recreate "$SERVICE"
  fi
  log "rollback done"
}

if [ -n "$ROLLBACK" ]; then
  do_rollback "$ROLLBACK"
  exit 0
fi

[ -d "$TREE/analysis" ] || die "no tree at $TREE/analysis"
[ -f "$TREE/.env" ] || die "no $TREE/.env"

if [ -d "$CLONE/.git" ]; then
  git -C "$CLONE" fetch --quiet --tags origin
else
  git clone --quiet "$REPO_URL" "$CLONE"
fi
git -C "$CLONE" checkout --quiet --detach "$REF"
SHA="$(git -C "$CLONE" rev-parse --short HEAD)"
SRC="$CLONE/analysis"
log "ref $REF is $SHA"

FILES=(Dockerfile .dockerignore .python-version pyproject.toml uv.lock README.md compose.yaml)
DIRS=(src tests notebooks)
RSYNC_EXCLUDES=(--exclude __pycache__ --exclude .pytest_cache --exclude __marimo__)

rsync_all() {
  local mode=("$@") f d
  for f in "${FILES[@]}"; do
    rsync "${mode[@]}" "$SRC/$f" "$TREE/analysis/$f"
  done
  for d in "${DIRS[@]}"; do
    rsync "${mode[@]}" --delete "${RSYNC_EXCLUDES[@]}" "$SRC/$d/" "$TREE/analysis/$d/"
  done
  rsync "${mode[@]}" "$CLONE/deploy/tf1/docker-compose.override.yaml" \
    "$TREE/analysis/docker-compose.override.yaml"
}

log "changes vs tree (dry):"
rsync_all -rlpti --dry-run --checksum | grep -v '^$' || true

if [ "$DRY_RUN" = "1" ]; then
  log "dry run: nothing written, nothing built"
  exit 0
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BACKUPS"
TGZ="$BACKUPS/kma-tf1-$STAMP.tgz"
log "backup $TGZ"
(umask 077 && tar -czf "$TGZ" -C "$(dirname "$TREE")" \
  --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='collector/state' "$(basename "$TREE")")
PREV_TAG="$IMAGE:pre-sync-$STAMP"
docker tag "$IMAGE:latest" "$PREV_TAG"
echo "$PREV_TAG" > "$TGZ.image"
log "previous image tagged $PREV_TAG"

rsync_all -rlpti --checksum >/dev/null
log "tree updated"

log "build"
start=$(date +%s)
GIT_SHA="$SHA" compose build "$SERVICE"
log "build took $(( $(date +%s) - start ))s"

wait_between_passes
GIT_SHA="$SHA" compose up -d --force-recreate --no-build "$SERVICE"

if ! verify "$SHA"; then
  log "verify failed, rolling back"
  do_rollback "$TGZ"
  exit 1
fi
log "done: $SHA, rollback with --rollback $TGZ"

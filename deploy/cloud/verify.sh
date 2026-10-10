#!/usr/bin/env bash
# Verify the v2 daily job on this host.
#   deploy/cloud/verify.sh preflight   before the first run, and after any change
#   deploy/cloud/verify.sh postrun     after a run
#   deploy/cloud/verify.sh units       installed units match the repo's, timer armed
#   deploy/cloud/verify.sh all         preflight, units, postrun
# Exit status is non-zero if any check failed. Reads /etc/kma/v2-daily.env.
set -uo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
textfile="${KMA_TEXTFILE_DIR:-/opt/iot-cloud/state/textfile}"
uv="${UV:-$HOME/.local/bin/uv}"
unit_dir=/etc/systemd/system
failed=0

stage() {
  (
    set -a
    # shellcheck disable=SC1091
    . /etc/kma/v2-daily.env
    set +a
    cd "$repo/analysis" && "$uv" run --frozen --no-dev python -m kma.v2_daily_check "$1" \
      --textfile-dir "$textfile"
  ) || failed=1
}

units() {
  local name
  for name in kma-v2-daily.service kma-v2-daily.timer; do
    if cmp -s "$repo/deploy/cloud/$name" "$unit_dir/$name"; then
      echo "ok   $name matches the repo"
    else
      echo "FAIL $name differs from, or is missing at, $unit_dir"; failed=1
    fi
  done
  if systemctl is-enabled --quiet kma-v2-daily.timer && systemctl is-active --quiet kma-v2-daily.timer; then
    echo "ok   timer enabled and active: $(systemctl list-timers kma-v2-daily.timer --no-legend | awk '{print $1, $2, $3}')"
  else
    echo "FAIL timer is not enabled and active"; failed=1
  fi
  local result
  result="$(systemctl show kma-v2-daily.service -p Result --value)"
  case "$result" in
    success) echo "ok   last service result: success" ;;
    *) echo "FAIL last service result: $result"; failed=1 ;;
  esac
}

case "${1:-all}" in
  preflight) stage preflight ;;
  postrun) stage postrun ;;
  units) units ;;
  all) stage preflight; units; stage postrun ;;
  *) echo "usage: $0 preflight|postrun|units|all" >&2; exit 2 ;;
esac
exit "$failed"

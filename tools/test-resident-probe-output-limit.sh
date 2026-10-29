#!/usr/bin/env bash
# File-backed attack test for the resident health-probe output limit.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
SOCKET_ROOT="/tmp/cuda-r3-probe-$$"
printed=0

cleanup() {
  local rc=$?
  [[ ! -f "${WORK}/resident.pid" ]] || kill "$(cat "${WORK}/resident.pid")" 2>/dev/null || true
  rm -rf "${SOCKET_ROOT}"
  rm -rf "${WORK}"
  if [[ "${printed}" -eq 0 ]]; then
    printf 'test-resident-probe-output-limit.sh: FAIL: unexpected exit %s\n' "${rc}"
  fi
}
trap cleanup EXIT

fail() {
  printed=1
  printf 'test-resident-probe-output-limit.sh: FAIL: %s\n' "$1"
  exit 1
}

mkdir -p "${WORK}/weights" "${WORK}/logs"
printf x > "${WORK}/weights/model.safetensors"
printf 'MemTotal: 10000000 kB\nMemAvailable: 20000000 kB\n' > "${WORK}/meminfo"

cat > "${WORK}/worker" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == runtime-worker ]]; then
  printf '%s\n' "\$\$" > "${WORK}/probe.pid"
  mode="\$(cat "${WORK}/flood-mode")"
  if [[ "\${mode}" == close-hang ]]; then
    exec python3 - <<'PY'
import os
import time

os.close(1)
os.close(2)
time.sleep(60)
PY
  fi
  python3 - "\${mode}" <<'PY'
import sys
import time

stream = sys.stdout.buffer if sys.argv[1] == "stdout" else sys.stderr.buffer
stream.write(b"x" * (2 * 1024 * 1024))
stream.flush()
time.sleep(60)
PY
  exit 0
fi
printf '%s\n' "\$\$" > "${WORK}/resident.pid"
socket=""
while [[ \$# -gt 0 ]]; do
  if [[ "\$1" == --socket ]]; then
    socket="\$2"
    shift 2
  else
    shift
  fi
done
touch "\${socket}"
exec sleep 60
EOF
chmod +x "${WORK}/worker"

run_case() {
  local stream="$1" out="${WORK}/${1}.out" marker kept_stdout kept_stderr
  printf '%s\n' "${stream}" > "${WORK}/flood-mode"
  find "${WORK}/logs" -type f -delete
  rm -f "${WORK}/probe.pid" "${WORK}/resident.pid" "${WORK}/socket.path"

  if env TENSORFOLD_BENCH_WORKER="${WORK}/worker" \
      MLXFAST_TARGET_SNAPSHOT_DIR="${WORK}/weights" \
      SERVE_UP_SOCKET_DIR="${SOCKET_ROOT}" SERVE_UP_LOG_DIR="${WORK}/logs" \
      SERVE_UP_HEALTH_TIMEOUT_S=5 SERVE_UP_TEST_READY_FILE=1 \
      SERVE_UP_TEST_REQUIRE_PROBE=1 SERVE_UP_MEMINFO="${WORK}/meminfo" \
      SERVE_UP_LAUNCHER=host \
      "${ROOT}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
      --socket-out "${WORK}/socket.path" >"${out}" 2>&1; then
    fail "${stream} flood exited zero"
  fi

  [[ "$(grep -Fxc 'resident probe output exceeded limit' "${out}")" -eq 1 ]] \
    || fail "${stream} flood did not print the fixed refusal once: $(tr '\n' ' ' < "${out}")"
  [[ "$(wc -c < "${out}" | tr -d '[:space:]')" -lt 1048576 ]] \
    || fail "${stream} flood leaked buffered output to the runner log"
  marker="$(find "${WORK}/logs" -name '*.probe-overflow' -type f -print -quit)"
  [[ -n "${marker}" ]] || fail "${stream} flood wrote no retained-byte record"
  read -r kept_stdout kept_stderr < "${marker}"
  [[ "${kept_stdout}" =~ ^[0-9]+$ && "${kept_stderr}" =~ ^[0-9]+$ ]] \
    || fail "${stream} flood wrote an invalid retained-byte record"
  (( kept_stdout <= 1048576 && kept_stderr <= 1048576 )) \
    || fail "${stream} flood kept more than 1 MiB from one stream"
  (( kept_stdout + kept_stderr <= 2097152 )) \
    || fail "${stream} flood kept more than 2 MiB total"
  if [[ "${stream}" == stdout ]]; then
    (( kept_stdout == 1048576 )) || fail "stdout flood did not fill the stdout cap"
  else
    (( kept_stderr == 1048576 )) || fail "stderr flood did not fill the stderr cap"
  fi
  [[ -f "${WORK}/probe.pid" ]] || fail "${stream} probe pid was not recorded"
  ! kill -0 "$(cat "${WORK}/probe.pid")" 2>/dev/null \
    || fail "${stream} probe process survived overflow"
  ! kill -0 "$(cat "${WORK}/resident.pid")" 2>/dev/null \
    || fail "${stream} resident survived the failed boot"
}

run_case stdout
run_case stderr

printf '%s\n' close-hang > "${WORK}/flood-mode"
find "${WORK}/logs" -type f -delete
rm -f "${WORK}/probe.pid" "${WORK}/resident.pid" "${WORK}/socket.path"
started="$(date +%s)"
if env TENSORFOLD_BENCH_WORKER="${WORK}/worker" \
    MLXFAST_TARGET_SNAPSHOT_DIR="${WORK}/weights" \
    SERVE_UP_SOCKET_DIR="${SOCKET_ROOT}" SERVE_UP_LOG_DIR="${WORK}/logs" \
    SERVE_UP_HEALTH_TIMEOUT_S=2 SERVE_UP_TEST_READY_FILE=1 \
    SERVE_UP_TEST_REQUIRE_PROBE=1 SERVE_UP_MEMINFO="${WORK}/meminfo" \
    SERVE_UP_LAUNCHER=host \
    "${ROOT}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
    --socket-out "${WORK}/socket.path" >"${WORK}/close-hang.out" 2>&1; then
  fail "close-then-hang probe exited zero"
fi
elapsed=$(($(date +%s) - started))
[[ "$(grep -Fxc 'resident probe timed out' "${WORK}/close-hang.out")" -eq 1 ]] \
  || fail "close-then-hang probe did not print the fixed timeout once: $(tr '\n' ' ' < "${WORK}/close-hang.out")"
(( elapsed <= 4 )) || fail "close-then-hang probe exceeded its deadline: ${elapsed}s"
[[ -f "${WORK}/probe.pid" ]] || fail "close-then-hang probe pid was not recorded"
! kill -0 "$(cat "${WORK}/probe.pid")" 2>/dev/null \
  || fail "close-then-hang probe process survived timeout"
! kill -0 "$(cat "${WORK}/resident.pid")" 2>/dev/null \
  || fail "close-then-hang resident survived failed boot"

printed=1
printf 'test-resident-probe-output-limit.sh: PASS\n'

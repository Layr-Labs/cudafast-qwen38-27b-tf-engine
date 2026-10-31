#!/usr/bin/env bash
#
# test-stop-leaked-residents.sh -- prove that cleanup stops only exact resident
# container names, handles removal races, and bounds Docker calls.
#
# This test uses local stubs. It needs no GPU, Docker daemon, or network.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STOPPER="${STOPPER_UNDER_TEST:-${REPO_ROOT}/tools/stop-leaked-residents.sh}"
WORK="$(mktemp -d)" || exit 1
trap 'rm -rf "${WORK}"' EXIT

STUB_BIN="${WORK}/bin"
DOCKER_NAMES_FILE="${WORK}/docker-names"
DOCKER_LOG="${WORK}/docker.log"
GPU_FILE="${WORK}/gpu"
GPU_LOG="${WORK}/gpu.log"
FLOCK_LOG="${WORK}/flock.log"
LOCK="${WORK}/gpu.lock"
STOP_MARK="${WORK}/stopped"
PS_COUNT_FILE="${WORK}/ps-count"
DOCKER_NAMES_AFTER_LOCK_FILE="${WORK}/docker-names-after-lock"
mkdir -p "${STUB_BIN}"
: > "${LOCK}"

cat > "${STUB_BIN}/docker" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${DOCKER_LOG}"
if [[ "${DOCKER_HANG_CALL}" == "$1" ]]; then
  while :; do :; done
fi
case "$1" in
  ps)
    if [[ -e "${STOP_MARK}" && "${AUTO_REMOVE_AFTER_PS}" -ge 0 ]]; then
      count="$(cat "${PS_COUNT_FILE}")"
      count=$((count + 1))
      printf '%s\n' "${count}" > "${PS_COUNT_FILE}"
      if [[ "${count}" -gt "${AUTO_REMOVE_AFTER_PS}" ]]; then
        grep -Fxv "${AUTO_REMOVE_NAME}" "${DOCKER_NAMES_FILE}" || true
        exit 0
      fi
    fi
    cat "${DOCKER_NAMES_FILE}"
    ;;
  stop)
    name="${!#}"
    case "${DOCKER_STOP_MODE}" in
      missing-gone)
        grep -Fxv "${name}" "${DOCKER_NAMES_FILE}" > "${DOCKER_NAMES_FILE}.new" || true
        mv "${DOCKER_NAMES_FILE}.new" "${DOCKER_NAMES_FILE}"
        echo "Error response from daemon: No such container: ${name}" >&2
        exit 1
        ;;
      fail)
        echo "Error response from daemon: stop failed for ${name}" >&2
        exit 1
        ;;
      normal)
        : > "${STOP_MARK}"
        ;;
    esac
    ;;
  rm)
    name="$3"
    case "${DOCKER_RM_MODE}" in
      already-in-progress)
        echo "Error response from daemon: removal of container ${name} is already in progress" >&2
        exit 1
        ;;
      keep)
        exit 0
        ;;
    esac
    grep -Fxv "${name}" "${DOCKER_NAMES_FILE}" > "${DOCKER_NAMES_FILE}.new" || true
    mv "${DOCKER_NAMES_FILE}.new" "${DOCKER_NAMES_FILE}"
    ;;
  *)
    echo "unexpected docker call: $*" >&2
    exit 1
    ;;
esac
EOF

cat > "${STUB_BIN}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${GPU_LOG}"
[[ "$*" == "--query-compute-apps=pid,process_name --format=csv,noheader" ]] || exit 1
if [[ "${NVIDIA_SMI_HANG:-}" == "1" ]]; then
  while :; do :; done
fi
cat "${GPU_FILE}"
EOF

# macOS does not provide flock(1). This stub locks the inherited file
# descriptor with the same non-blocking and bounded-wait behavior used here.
cat > "${STUB_BIN}/flock" <<'PY'
#!/usr/bin/env python3
import fcntl
import os
import shutil
import sys
import time

with open(os.environ["FLOCK_LOG"], "a", encoding="utf-8") as log:
    log.write(" ".join(sys.argv[1:]) + "\n")

args = sys.argv[1:]
wait = None
if args[:1] == ["-n"]:
    wait = 0.0
    args = args[1:]
elif args[:1] == ["-w"] and len(args) >= 3:
    wait = float(args[1])
    args = args[2:]
if len(args) != 1:
    raise SystemExit(2)
fd = int(args[0])
deadline = None if wait is None else time.monotonic() + wait
while True:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.environ.get("SWAP_NAMES_ON_LOCK") == "1":
            shutil.copyfile(
                os.environ["DOCKER_NAMES_AFTER_LOCK_FILE"],
                os.environ["DOCKER_NAMES_FILE"],
            )
        raise SystemExit(0)
    except BlockingIOError:
        if deadline is not None and time.monotonic() >= deadline:
            raise SystemExit(1)
        time.sleep(0.02)
PY

# macOS does not provide GNU timeout(1). This stub supports the two forms used
# by the guard and this test. It kills the full child process group on timeout.
cat > "${STUB_BIN}/timeout" <<'PY'
#!/usr/bin/env python3
import os
import signal
import subprocess
import sys

args = sys.argv[1:]
kill_after = None
signal_name = "TERM"
while args and args[0].startswith("--"):
    option = args.pop(0)
    if option.startswith("--kill-after="):
        kill_after = float(option.split("=", 1)[1])
    elif option.startswith("--signal="):
        signal_name = option.split("=", 1)[1]
    else:
        raise SystemExit(125)
if len(args) < 2:
    raise SystemExit(125)
duration = float(args.pop(0))
child = subprocess.Popen(args, start_new_session=True)
try:
    raise SystemExit(child.wait(timeout=duration))
except subprocess.TimeoutExpired:
    os.killpg(child.pid, getattr(signal, "SIG" + signal_name))
    if kill_after is not None:
        try:
            child.wait(timeout=kill_after)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
    child.wait()
    raise SystemExit(124)
PY

chmod +x "${STUB_BIN}/docker" "${STUB_BIN}/nvidia-smi" \
  "${STUB_BIN}/flock" "${STUB_BIN}/timeout"
export PATH="${STUB_BIN}:${PATH}"
export FLOCK_LOG
[[ "$(command -v docker)" == "${STUB_BIN}/docker" ]] || exit 1
[[ "$(command -v nvidia-smi)" == "${STUB_BIN}/nvidia-smi" ]] || exit 1
[[ "$(command -v flock)" == "${STUB_BIN}/flock" ]] || exit 1
[[ "$(command -v timeout)" == "${STUB_BIN}/timeout" ]] || exit 1

failures=0
fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

reset_case() {
  : > "${DOCKER_NAMES_FILE}"
  : > "${DOCKER_LOG}"
  : > "${GPU_FILE}"
  : > "${GPU_LOG}"
  : > "${FLOCK_LOG}"
  : > "${PS_COUNT_FILE}"
  : > "${DOCKER_NAMES_AFTER_LOCK_FILE}"
  rm -f "${STOP_MARK}"
  AUTO_REMOVE_AFTER_PS=-1
  AUTO_REMOVE_NAME=""
  DOCKER_HANG_CALL=""
  DOCKER_STOP_MODE=normal
  DOCKER_RM_MODE=normal
  SWAP_NAMES_ON_LOCK=0
  TEST_DOCKER_TIMEOUT=1
  TEST_GPU_WAIT=2
  TEST_LOCK_WAIT=0
}

run_case() {
  local prefix="$1" mode="$2"
  OUT="$(env \
    PATH="${STUB_BIN}:${PATH}" \
    DOCKER_NAMES_FILE="${DOCKER_NAMES_FILE}" \
    DOCKER_LOG="${DOCKER_LOG}" \
    GPU_FILE="${GPU_FILE}" \
    GPU_LOG="${GPU_LOG}" \
    FLOCK_LOG="${FLOCK_LOG}" \
    STOP_MARK="${STOP_MARK}" \
    PS_COUNT_FILE="${PS_COUNT_FILE}" \
    DOCKER_NAMES_AFTER_LOCK_FILE="${DOCKER_NAMES_AFTER_LOCK_FILE}" \
    AUTO_REMOVE_AFTER_PS="${AUTO_REMOVE_AFTER_PS}" \
    AUTO_REMOVE_NAME="${AUTO_REMOVE_NAME}" \
    DOCKER_HANG_CALL="${DOCKER_HANG_CALL}" \
    DOCKER_STOP_MODE="${DOCKER_STOP_MODE}" \
    DOCKER_RM_MODE="${DOCKER_RM_MODE}" \
    SWAP_NAMES_ON_LOCK="${SWAP_NAMES_ON_LOCK}" \
    STOP_LEAKED_RESIDENTS_TEST_DOCKER_TIMEOUT="${TEST_DOCKER_TIMEOUT}" \
    SERVE_UP_CONTAINER_PREFIX="${prefix}" \
    "${STOPPER}" --mode "${mode}" --lock "${LOCK}" \
      --gpu-wait "${TEST_GPU_WAIT}" --lock-wait "${TEST_LOCK_WAIT}" 2>&1)"
  RC=$?
}

run_without_prefix() {
  OUT="$(env -u SERVE_UP_CONTAINER_PREFIX \
    PATH="${STUB_BIN}:${PATH}" \
    DOCKER_NAMES_FILE="${DOCKER_NAMES_FILE}" \
    DOCKER_LOG="${DOCKER_LOG}" \
    GPU_FILE="${GPU_FILE}" \
    GPU_LOG="${GPU_LOG}" \
    FLOCK_LOG="${FLOCK_LOG}" \
    STOP_MARK="${STOP_MARK}" \
    PS_COUNT_FILE="${PS_COUNT_FILE}" \
    DOCKER_NAMES_AFTER_LOCK_FILE="${DOCKER_NAMES_AFTER_LOCK_FILE}" \
    AUTO_REMOVE_AFTER_PS="${AUTO_REMOVE_AFTER_PS}" \
    AUTO_REMOVE_NAME="${AUTO_REMOVE_NAME}" \
    DOCKER_HANG_CALL="${DOCKER_HANG_CALL}" \
    DOCKER_STOP_MODE="${DOCKER_STOP_MODE}" \
    DOCKER_RM_MODE="${DOCKER_RM_MODE}" \
    SWAP_NAMES_ON_LOCK="${SWAP_NAMES_ON_LOCK}" \
    STOP_LEAKED_RESIDENTS_TEST_DOCKER_TIMEOUT="${TEST_DOCKER_TIMEOUT}" \
    "${STOPPER}" --mode preflight --lock "${LOCK}" \
      --gpu-wait "${TEST_GPU_WAIT}" --lock-wait "${TEST_LOCK_WAIT}" 2>&1)"
  RC=$?
}

run_hung_query() {
  OUT="$(env \
    PATH="${STUB_BIN}:${PATH}" \
    DOCKER_NAMES_FILE="${DOCKER_NAMES_FILE}" \
    DOCKER_LOG="${DOCKER_LOG}" \
    GPU_FILE="${GPU_FILE}" \
    GPU_LOG="${GPU_LOG}" \
    FLOCK_LOG="${FLOCK_LOG}" \
    NVIDIA_SMI_HANG=1 \
    STOP_MARK="${STOP_MARK}" \
    PS_COUNT_FILE="${PS_COUNT_FILE}" \
    DOCKER_NAMES_AFTER_LOCK_FILE="${DOCKER_NAMES_AFTER_LOCK_FILE}" \
    AUTO_REMOVE_AFTER_PS="${AUTO_REMOVE_AFTER_PS}" \
    AUTO_REMOVE_NAME="${AUTO_REMOVE_NAME}" \
    DOCKER_HANG_CALL="${DOCKER_HANG_CALL}" \
    DOCKER_STOP_MODE="${DOCKER_STOP_MODE}" \
    DOCKER_RM_MODE="${DOCKER_RM_MODE}" \
    SWAP_NAMES_ON_LOCK="${SWAP_NAMES_ON_LOCK}" \
    STOP_LEAKED_RESIDENTS_TEST_DOCKER_TIMEOUT="${TEST_DOCKER_TIMEOUT}" \
    SERVE_UP_CONTAINER_PREFIX="${PREFIX}" \
    timeout --kill-after=1 --signal=TERM 4 \
      "${STOPPER}" --mode preflight --lock "${LOCK}" \
        --gpu-wait 2 --lock-wait 0 2>&1)"
  RC=$?
}

run_hung_docker() {
  DOCKER_HANG_CALL="ps"
  OUT="$(env \
    PATH="${STUB_BIN}:${PATH}" \
    DOCKER_NAMES_FILE="${DOCKER_NAMES_FILE}" \
    DOCKER_LOG="${DOCKER_LOG}" \
    GPU_FILE="${GPU_FILE}" \
    GPU_LOG="${GPU_LOG}" \
    FLOCK_LOG="${FLOCK_LOG}" \
    STOP_MARK="${STOP_MARK}" \
    PS_COUNT_FILE="${PS_COUNT_FILE}" \
    DOCKER_NAMES_AFTER_LOCK_FILE="${DOCKER_NAMES_AFTER_LOCK_FILE}" \
    AUTO_REMOVE_AFTER_PS="${AUTO_REMOVE_AFTER_PS}" \
    AUTO_REMOVE_NAME="${AUTO_REMOVE_NAME}" \
    DOCKER_HANG_CALL="${DOCKER_HANG_CALL}" \
    DOCKER_STOP_MODE="${DOCKER_STOP_MODE}" \
    DOCKER_RM_MODE="${DOCKER_RM_MODE}" \
    SWAP_NAMES_ON_LOCK="${SWAP_NAMES_ON_LOCK}" \
    SERVE_UP_CONTAINER_PREFIX="${PREFIX}" \
    STOP_LEAKED_RESIDENTS_TEST_DOCKER_TIMEOUT=1 \
    timeout --kill-after=1 --signal=TERM 4 \
      "${STOPPER}" --mode preflight --lock "${LOCK}" \
        --gpu-wait 2 --lock-wait 0 2>&1)"
  RC=$?
}

actions() {
  grep -E '^(stop -t 10|rm -f) ' "${DOCKER_LOG}" || true
}

expect_rc() {
  local name="$1" want="$2"
  [[ "${RC}" -eq "${want}" ]] || fail "${name}: exit ${RC}, expected ${want}; ${OUT}"
}

expect_actions() {
  local name="$1" want="$2" got
  got="$(actions)"
  [[ "${got}" == "${want}" ]] \
    || fail "${name}: Docker actions were '${got}', expected '${want}'"
}

PREFIX="serve-up-"
MATCH="${PREFIX}2192061-20261031T185402Z"

# 1. An unset prefix uses serve-up- and stops only the exact name.
reset_case
printf '%s\n' \
  "${MATCH}" \
  'eager_joliot' \
  'qe1-serve-up-2192061-20261031T185402Z' \
  'serve-up-2192061-20261031T185402Z-extra' \
  'serve-up-2192061-20261031T1854Z' > "${DOCKER_NAMES_FILE}"
run_without_prefix
expect_rc "unset prefix default" 0
expect_actions "unset prefix default" "stop -t 10 ${MATCH}
rm -f ${MATCH}"

# 2. Invalid prefixes must not select any container.
for bad_prefix in serve-up 'a*-'; do
  reset_case
  printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
  run_case "${bad_prefix}" preflight
  expect_rc "bad prefix ${bad_prefix}" 2
  expect_actions "bad prefix ${bad_prefix}" ""
done

# 2a. The test-only Docker timeout accepts only 1 through 30.
for bad_timeout in 31 18446744073709551616; do
  reset_case
  TEST_DOCKER_TIMEOUT="${bad_timeout}"
  run_case "${PREFIX}" preflight
  expect_rc "bad Docker timeout ${bad_timeout}" 2
  expect_actions "bad Docker timeout ${bad_timeout}" ""
  [[ "${OUT}" == *"STOP_LEAKED_RESIDENTS_TEST_DOCKER_TIMEOUT"* ]] \
    || fail "bad Docker timeout ${bad_timeout}: error does not name the variable; ${OUT}"
done
reset_case
TEST_DOCKER_TIMEOUT=5
run_case "${PREFIX}" preflight
expect_rc "Docker timeout 5" 0

# 3. An idle box needs no cleanup.
reset_case
run_case "${PREFIX}" preflight
expect_rc "idle box" 0
expect_actions "idle box" ""

# 4. Preflight never stops random names or another prefix.
reset_case
printf '%s\n' \
  "${MATCH}" \
  'eager_joliot' \
  'qe1-serve-up-12-20261031T185402Z' \
  'serve-up-x-20261031T185402Z' \
  'serve-up-12-20261031T185402Z-extra' > "${DOCKER_NAMES_FILE}"
run_case "${PREFIX}" preflight
expect_rc "exact preflight match" 0
expect_actions "exact preflight match" "stop -t 10 ${MATCH}
rm -f ${MATCH}"
[[ "${OUT}" == *"::warning::"*"left by an earlier job"* ]] \
  || fail "exact preflight match: warning text is missing; ${OUT}"

# 5. Another GPU lock holder prevents all container actions.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
READY="${WORK}/lock-ready"
(
  exec 8< "${LOCK}"
  flock -n 8
  : > "${READY}"
  sleep 2
) &
holder=$!
for _ in $(seq 1 100); do
  [[ -e "${READY}" ]] && break
  sleep 0.01
done
run_case "${PREFIX}" preflight
wait "${holder}" 2>/dev/null || true
expect_rc "held lock" 1
expect_actions "held lock" ""
[[ "${OUT}" == *"${LOCK}"* && "${OUT}" == *"another GPU window holds it"* ]] \
  || fail "held lock: error does not name the lock; ${OUT}"

# 5a. Empty cleanup does not take or wait for the GPU lock.
reset_case
READY="${WORK}/lock-ready-empty-cleanup"
(
  exec 8< "${LOCK}"
  flock -n 8
  : > "${READY}"
  sleep 1
) &
holder=$!
for _ in $(seq 1 100); do
  [[ -e "${READY}" ]] && break
  sleep 0.01
done
: > "${FLOCK_LOG}"
TEST_LOCK_WAIT=2
run_case "${PREFIX}" cleanup
wait "${holder}" 2>/dev/null || true
expect_rc "empty cleanup with held lock" 0
expect_actions "empty cleanup with held lock" ""
[[ ! -s "${FLOCK_LOG}" ]] \
  || fail "empty cleanup with held lock: cleanup called flock"
[[ ! -s "${GPU_LOG}" ]] \
  || fail "empty cleanup with held lock: nvidia-smi was called"

# 6. A process that stays on the GPU fails after container cleanup.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
printf '%s\n' '4242, python3' > "${GPU_FILE}"
run_case "${PREFIX}" preflight
expect_rc "busy GPU" 1
expect_actions "busy GPU" "stop -t 10 ${MATCH}
rm -f ${MATCH}"
[[ "${OUT}" == *"4242, python3"* ]] \
  || fail "busy GPU: error does not list the process; ${OUT}"

# 7. Cleanup uses the same exact stop and remove calls.
reset_case
printf '%s\n' \
  "${MATCH}" \
  'eager_joliot' \
  'qe1-serve-up-12-20261031T185402Z' > "${DOCKER_NAMES_FILE}"
run_case "${PREFIX}" cleanup
expect_rc "cleanup match" 0
expect_actions "cleanup match" "stop -t 10 ${MATCH}
rm -f ${MATCH}"
[[ "${OUT}" != *"::warning::"* ]] \
  || fail "cleanup match: cleanup used a warning; ${OUT}"

# 7a. Cleanup uses only the fresh list read after taking the lock.
reset_case
STALE_MATCH="${PREFIX}1-20261031T185402Z"
FRESH_MATCH="${PREFIX}2-20261031T185402Z"
printf '%s\n' "${STALE_MATCH}" > "${DOCKER_NAMES_FILE}"
printf '%s\n' "${FRESH_MATCH}" > "${DOCKER_NAMES_AFTER_LOCK_FILE}"
SWAP_NAMES_ON_LOCK=1
run_case "${PREFIX}" cleanup
expect_rc "cleanup refreshes matches" 0
expect_actions "cleanup refreshes matches" "stop -t 10 ${FRESH_MATCH}
rm -f ${FRESH_MATCH}"
[[ "$(actions)" != *"${STALE_MATCH}"* ]] \
  || fail "cleanup refreshes matches: stale container was attempted"

# 8. A prefix-only name is not a resident container name.
reset_case
printf '%s\n' "${PREFIX}1-2026x" > "${DOCKER_NAMES_FILE}"
run_case "${PREFIX}" preflight
expect_rc "prefix-only name" 0
expect_actions "prefix-only name" ""

# 9. A hung GPU query must not exceed the GPU wait.
reset_case
run_hung_query
expect_rc "hung GPU query" 1
expect_actions "hung GPU query" ""
[[ "${OUT}" == *"::error::nvidia-smi query timed out"* ]] \
  || fail "hung GPU query: timeout error is missing; ${OUT}"

# 9a. A hung Docker call fails within the test-only timeout.
reset_case
run_hung_docker
expect_rc "hung docker ps" 1
expect_actions "hung docker ps" ""
[[ "${OUT}" == *"::error::docker ps timed out after 1 seconds"* ]] \
  || fail "hung docker ps: timeout error is missing; ${OUT}"

# A. Docker can report that automatic removal is still in progress.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
DOCKER_RM_MODE=already-in-progress
AUTO_REMOVE_AFTER_PS=0
AUTO_REMOVE_NAME="${MATCH}"
run_case "${PREFIX}" preflight
expect_rc "delayed automatic removal" 0
expect_actions "delayed automatic removal" "stop -t 10 ${MATCH}
rm -f ${MATCH}"

# A1. A failed stop is harmless when the container is confirmed gone.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
DOCKER_STOP_MODE=missing-gone
run_case "${PREFIX}" preflight
expect_rc "stop race with removed container" 0
expect_actions "stop race with removed container" "stop -t 10 ${MATCH}"

# A2. A failed stop is fatal when the container still exists.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
DOCKER_STOP_MODE=fail
run_case "${PREFIX}" preflight
expect_rc "stop failure with present container" 1
expect_actions "stop failure with present container" "stop -t 10 ${MATCH}"
[[ "${OUT}" == *"::error::docker stop failed for ${MATCH}"* ]] \
  || fail "stop failure with present container: stop error is missing; ${OUT}"

# B. A resident that stays past the removal deadline is an error.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
DOCKER_RM_MODE=keep
run_case "${PREFIX}" preflight
expect_rc "resident remains" 1
[[ "${OUT}" == *"::error::resident containers remain after cleanup: ${MATCH}"* ]] \
  || fail "resident remains: remain error is missing; ${OUT}"

# C. The box graphics contexts do not make the GPU busy.
reset_case
printf '%s\n' '123, /usr/lib/xorg/Xorg' '456, /usr/bin/gnome-shell' > "${GPU_FILE}"
run_case "${PREFIX}" preflight
expect_rc "graphics contexts" 0

# D. A vLLM process is foreign even when Xorg is present.
reset_case
printf '%s\n' '123, /usr/lib/xorg/Xorg' '789, VLLM::EngineCore' > "${GPU_FILE}"
run_case "${PREFIX}" preflight
expect_rc "foreign GPU process" 1
[[ "${OUT}" == *"789, VLLM::EngineCore"* ]] \
  || fail "foreign GPU process: process is missing from the error; ${OUT}"

# E. A bounded lock wait queues until the current GPU window ends.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
READY="${WORK}/lock-ready-wait"
(
  exec 8< "${LOCK}"
  flock -n 8
  : > "${READY}"
  sleep 2
) &
holder=$!
for _ in $(seq 1 100); do
  [[ -e "${READY}" ]] && break
  sleep 0.01
done
TEST_LOCK_WAIT=5
run_case "${PREFIX}" preflight
wait "${holder}" 2>/dev/null || true
expect_rc "bounded lock wait" 0
expect_actions "bounded lock wait" "stop -t 10 ${MATCH}
rm -f ${MATCH}"

# F. A lock wait deadline stops no container.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
READY="${WORK}/lock-ready-timeout"
(
  exec 8< "${LOCK}"
  flock -n 8
  : > "${READY}"
  sleep 3
) &
holder=$!
for _ in $(seq 1 100); do
  [[ -e "${READY}" ]] && break
  sleep 0.01
done
TEST_LOCK_WAIT=1
run_case "${PREFIX}" preflight
wait "${holder}" 2>/dev/null || true
expect_rc "lock wait deadline" 1
expect_actions "lock wait deadline" ""
[[ "${OUT}" == *"${LOCK}"* && "${OUT}" == *"another GPU window holds it"* ]] \
  || fail "lock wait deadline: lock error is missing; ${OUT}"

# G. Cleanup with a match fails closed when another process holds the lock.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
READY="${WORK}/lock-ready-cleanup-timeout"
(
  exec 8< "${LOCK}"
  flock -n 8
  : > "${READY}"
  sleep 3
) &
holder=$!
for _ in $(seq 1 100); do
  [[ -e "${READY}" ]] && break
  sleep 0.01
done
TEST_LOCK_WAIT=1
run_case "${PREFIX}" cleanup
wait "${holder}" 2>/dev/null || true
expect_rc "cleanup lock wait deadline" 3
expect_actions "cleanup lock wait deadline" ""
[[ "${OUT}" == "::warning::the GPU lock is still held; not stopping the matching resident container; the next job's preflight stops any leftover" ]] \
  || fail "cleanup lock wait deadline: warning is not exact; ${OUT}"
[[ ! -s "${GPU_LOG}" ]] \
  || fail "cleanup lock wait deadline: nvidia-smi was called"

if [[ "${failures}" -ne 0 ]]; then
  echo "test-stop-leaked-residents.sh: ${failures} case(s) failed" >&2
  exit 1
fi
echo "test-stop-leaked-residents.sh: all cases passed"

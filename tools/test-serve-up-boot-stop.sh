#!/usr/bin/env bash
# Test resident lifecycle, drafter plumbing, and container isolation.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TMP="$(mktemp -d)"
SANDBOX_TMP="/tmp/p6-sbx-$$"
SOCKET_BASE="/tmp/s$$"
cleanup() {
  rm -rf "${SOCKET_BASE}"
  [[ ! -f "${TMP}/socket.path" ]] || SERVE_UP_LAUNCHER=host SERVE_UP_LOG_DIR="${TMP}/logs" \
    "${ROOT}/tools/serve-up.sh" --stop --socket "$(sed -n '1p' "${TMP}/socket.path")" >/dev/null 2>&1 || true
  [[ ! -f "${TMP}/docker.pid" ]] || kill "$(cat "${TMP}/docker.pid")" 2>/dev/null || true
  rm -rf "/tmp/p6-serve-$$"
  rm -rf "${SANDBOX_TMP}"
  rm -rf "${TMP}"
}
trap cleanup EXIT
mkdir -p "${TMP}/weights" "${TMP}/drafter" "${TMP}/logs"
printf x > "${TMP}/weights/model.safetensors"
printf 'MemTotal: 10000000 kB\nMemAvailable: 20000000 kB\n' > "${TMP}/meminfo"
cat > "${TMP}/worker" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == runtime-worker ]]; then
  [[ ! -f "${TMP}/probe-slow" ]] || sleep 5  # longer than the 3 s deadline
  resident_pid="\$(cat "${TMP}/resident.pid")"
  printf '{"id":0,"ok":true,"protocol_version":1,"backend":"cuda","device":"mock","resident":{"pid":%s,"load_epoch":1}}\n' "\${resident_pid}"
  exit 0
fi
printf '%s\n' "\$@" > "${TMP}/worker.args"
[[ "\${1:-}" == resident ]] || exit 1
printf '%s\n' "\$\$" > "${TMP}/resident.pid"
while [[ \$# -gt 0 ]]; do
  if [[ "\$1" == --socket ]]; then socket="\$2"; shift 2; else shift; fi
done
touch "\${socket}"
exec sleep 60
EOF
chmod +x "${TMP}/worker"

common=(TENSORFOLD_BENCH_WORKER="${TMP}/worker" MLXFAST_TARGET_SNAPSHOT_DIR="${TMP}/weights"
  SERVE_UP_SOCKET_DIR="/tmp/p6-serve-$$" SERVE_UP_LOG_DIR="${TMP}/logs"
  SERVE_UP_HEALTH_TIMEOUT_S=5 SERVE_UP_TEST_READY_FILE=1 SERVE_UP_TEST_REQUIRE_PROBE=1
  SERVE_UP_MEMINFO="${TMP}/meminfo")

env "${common[@]}" SERVE_UP_LAUNCHER=host \
  "${ROOT}/tools/serve-up.sh" --boot --spec serial --draft-len 0 --socket-out "${TMP}/socket.path"
socket="$(sed -n '1p' "${TMP}/socket.path")"
[[ -f "${socket}" ]]
jq -e '.engine == "tensorfold" and .spec_config == "serial" and
  .hello.protocol_version == 1 and .hello.resident.load_epoch == 1' \
  "${TMP}/logs/serve-identity.json" >/dev/null
[[ "$(stat -c %a "${TMP}/logs/serve-identity.json" 2>/dev/null || stat -f %Lp "${TMP}/logs/serve-identity.json")" == 600 ]]
env SERVE_UP_LAUNCHER=host SERVE_UP_LOG_DIR="${TMP}/logs" \
  "${ROOT}/tools/serve-up.sh" --stop --socket "${socket}" >/dev/null

rm -f "${TMP}/socket.path"
env "${common[@]}" SERVE_UP_LAUNCHER=host MLXFAST_DRAFTER_SNAPSHOT_DIR="${TMP}/drafter" \
  "${ROOT}/tools/serve-up.sh" --boot --spec dflash --draft-len 1 --socket-out "${TMP}/socket.path"
grep -Fxq -- '--drafter' "${TMP}/worker.args"
grep -Fxq -- "$(cd "${TMP}/drafter" && pwd -P)" "${TMP}/worker.args"
socket="$(sed -n '1p' "${TMP}/socket.path")"
env SERVE_UP_LAUNCHER=host SERVE_UP_LOG_DIR="${TMP}/logs" \
  "${ROOT}/tools/serve-up.sh" --stop --socket "${socket}" >/dev/null

if env "${common[@]}" SERVE_UP_LAUNCHER=host \
    "${ROOT}/tools/serve-up.sh" --boot --spec dflash --draft-len 1 --socket-out "${TMP}/missing.path" \
    >"${TMP}/missing.out" 2>&1; then
  echo 'test-serve-up-boot-stop: DFlash started without a staged drafter' >&2
  exit 1
fi
grep -Fq 'MLXFAST_DRAFTER_SNAPSHOT_DIR is missing' "${TMP}/missing.out"

rm -f "${TMP}/resident.pid"
if env "${common[@]}" SERVE_UP_LAUNCHER=host \
    "${ROOT}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
    --socket-out /dev/null/socket >"${TMP}/handover.out" 2>&1; then
  echo 'test-serve-up-boot-stop: failed handover exited zero' >&2
  exit 1
fi
handover_pid="$(cat "${TMP}/resident.pid")"
if kill -0 "${handover_pid}" 2>/dev/null; then
  echo 'test-serve-up-boot-stop: failed handover left the resident running' >&2
  exit 1
fi
[[ -z "$(find "/tmp/p6-serve-$$" -type s -o -type f 2>/dev/null)" ]]

touch "${TMP}/probe-slow"
started="$(date +%s)"
if env "${common[@]}" SERVE_UP_LAUNCHER=host SERVE_UP_HEALTH_TIMEOUT_S=3 \
    "${ROOT}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
    --socket-out "${TMP}/slow.path" >"${TMP}/slow.out" 2>&1; then
  echo 'test-serve-up-boot-stop: slow health probe exited zero' >&2
  exit 1
fi
elapsed=$(($(date +%s) - started))
rm -f "${TMP}/probe-slow"
grep -Fxq 'resident probe timed out' "${TMP}/slow.out"
if (( elapsed >= 6 )); then
  echo "test-serve-up-boot-stop: 3-second health deadline took ${elapsed} seconds" >&2
  exit 1
fi
workflow="${ROOT}/.github/workflows/benchmark.yml"
grep -Fq 'z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4' "${workflow}"
preflight_line="$(grep -n 'run: ./tools/ranked-box-preflight.sh' "${workflow}" | cut -d: -f1)"
drafter_line="$(grep -n 'Stage model snapshots (dereferenced and verified)' "${workflow}" | cut -d: -f1)"
(( drafter_line > preflight_line ))

BIN="${TMP}/bin"
mkdir -p "${BIN}"
cat > "${BIN}/docker" <<'EOF'
#!/usr/bin/env bash
printf 'docker' >> "${DOCKER_LOG}"
printf ' <%s>' "$@" >> "${DOCKER_LOG}"
printf '\n' >> "${DOCKER_LOG}"
[[ "${1:-} ${2:-}" == "image inspect" ]] && exit 0
if [[ "${1:-}" == exec ]]; then
  pid="$(cat "${DOCKER_PID_FILE}")"
  printf '{"id":0,"ok":true,"protocol_version":1,"backend":"cuda","device":"mock","resident":{"pid":%s,"load_epoch":1}}\n' "${pid}"
  exit 0
fi
if [[ "${1:-}" == stop || "${1:-}" == rm ]]; then
  [[ ! -f "${DOCKER_PID_FILE}" ]] || kill "$(cat "${DOCKER_PID_FILE}")" 2>/dev/null || true
  exit 0
fi
[[ "${1:-}" == run ]] || exit 0
socket=""; prev=""
for arg in "$@"; do
  [[ "${prev}" == --socket ]] && socket="${arg}"
  prev="${arg}"
done
[[ -n "${socket}" ]] || exit 4
printf '%s\n' "$$" > "${DOCKER_PID_FILE}"
if [[ "${DOCKER_OVERFLOW:-0}" == 1 ]]; then
  /usr/bin/python3 -c 'import sys; sys.stdout.buffer.write(b"0" * (64 * 1024 * 1024 + 1))'
  exec sleep 60
fi
touch "${socket}"
exec sleep 60
EOF
chmod +x "${BIN}/docker"
cat > "${BIN}/getent" <<'EOF'
#!/usr/bin/env bash
[[ "${1:-}" == group ]] || exit 1
printf '%s:x:%s:\n' "${2}" "$(/usr/bin/id -g)"
EOF
chmod +x "${BIN}/getent"
cat > "${BIN}/sudo" <<'EOF'
#!/usr/bin/env bash
printf 'sudo <%s>\n' "$*" >> "${SUDO_LOG}"
if [[ "${1:-} ${2:-}" == "-n --" ]]; then
  shift 2
  if [[ "$1" == /usr/bin/rm && "$2" == -rf && "$3" == --one-file-system ]]; then
    shift 3
    exec /bin/rm -rf "$@"
  fi
  if [[ "$1" == /usr/bin/install ]]; then
    target="${!#}"
    mkdir -p "${target}" || exit 1
    chmod 0700 "${target}"
    exit $?
  fi
  exec "$@"
fi
[[ "${1:-}" == -n && "${2:-}" == -u && "${4:-}" == -- ]] || exit 2
shift 4
[[ "${1:-}" == kill || "${1:-}" == true ]] && exit 0
exec "$@"
EOF
cat > "${BIN}/pgrep" <<'EOF'
#!/usr/bin/env bash
[[ "${1:-}" == -u && "${2:-}" == "${SANDBOX_TEST_USER}" ]] && exit 1
exec /usr/bin/pgrep "$@"
EOF
cat > "${BIN}/ipcs" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' '------ header --------' 'key id owner perms'
EOF
cat > "${BIN}/keyctl" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in session) shift 2; exec "$@" ;; get_persistent) echo 424242 ;; link|clear|rlist) exit 0 ;; *) exit 1 ;; esac
EOF
chmod +x "${BIN}/sudo" "${BIN}/pgrep" "${BIN}/ipcs" "${BIN}/keyctl"
: > "${TMP}/docker.log"
rm -f "${TMP}/socket.path"
PATH="${BIN}:${PATH}" DOCKER_LOG="${TMP}/docker.log" DOCKER_PID_FILE="${TMP}/docker.pid" \
  SECRET_MUST_NOT_LEAK=hidden env "${common[@]}" \
  "${ROOT}/tools/serve-up.sh" --boot --spec serial --draft-len 0 --socket-out "${TMP}/socket.path"
run="$(grep '^docker <run>' "${TMP}/docker.log")"
for flag in '<--network> <none>' '<--read-only>' '<--cap-drop> <ALL>' \
  '<--security-opt> <no-new-privileges>' '<--gpus> <all>' '<--ipc> <host>' \
  '<--pids-limit> <4096>' '<--memory> <64g>' '<--log-opt> <max-size=64m>' \
  '<--log-opt> <max-file=1>'; do
  grep -Fq -- "${flag}" <<<"${run}"
done
worker_real="$(cd "$(dirname "${TMP}/worker")" && pwd -P)/worker"
weights_real="$(cd "${TMP}/weights" && pwd -P)"
grep -Fq -- "<type=bind,src=${worker_real},dst=${worker_real},readonly>" <<<"${run}"
grep -Fq -- "<type=bind,src=${weights_real},dst=${weights_real},readonly>" <<<"${run}"
! grep -Fq 'SECRET_MUST_NOT_LEAK' <<<"${run}"
socket="$(sed -n '1p' "${TMP}/socket.path")"
PATH="${BIN}:${PATH}" DOCKER_LOG="${TMP}/docker.log" DOCKER_PID_FILE="${TMP}/docker.pid" \
  SERVE_UP_LOG_DIR="${TMP}/logs" "${ROOT}/tools/serve-up.sh" --stop --socket "${socket}" >/dev/null

: > "${TMP}/docker.log"
rm -f "${TMP}/docker.pid" "${TMP}/socket.path"
if PATH="${BIN}:${PATH}" DOCKER_LOG="${TMP}/docker.log" DOCKER_PID_FILE="${TMP}/docker.pid" \
    DOCKER_OVERFLOW=1 env "${common[@]}" \
    "${ROOT}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
    --socket-out "${TMP}/socket.path" >"${TMP}/overflow.out" 2>&1; then
  echo 'test-serve-up-boot-stop: resident log overflow exited zero' >&2
  exit 1
fi
grep -Fxq 'serve-up.sh: RESIDENT-LOG-LIMIT-EXCEEDED' "${TMP}/overflow.out"
overflow_marker="$(find "${TMP}/logs" -name 'tensorfold-resident.*.overflow' -type f -print -quit)"
overflow_log="${overflow_marker%.overflow}.log"
grep -Fxq 'serve-up.sh: RESIDENT-LOG-LIMIT-EXCEEDED' "${overflow_log}"
[[ "$(wc -c < "${overflow_log}" | tr -d '[:space:]')" -le 67108864 ]]
! kill -0 "$(cat "${TMP}/docker.pid")" 2>/dev/null

cp "${TMP}/worker" "${TMP}/worker.sandboxed"
chmod +x "${TMP}/worker.sandboxed"
mkdir -p "${SANDBOX_TMP}/job"
mkdir -m 2750 "${SANDBOX_TMP}/job/sock"
mkdir -m 2770 "${SANDBOX_TMP}/job/zig-global-cache" "${SANDBOX_TMP}/job/zig-cache" \
  "${SANDBOX_TMP}/job/out" "${SANDBOX_TMP}/job/adapter-home"
mkdir -p "${TMP}/wipe" "${TMP}/wipe-home"
printf '%s\n' "${TMP}/wipe" > "${TMP}/wipe-list"
: > "${TMP}/docker.log"
: > "${TMP}/sudo.log"
rm -f "${TMP}/socket.path" "${TMP}/docker.pid"
PATH="${BIN}:${PATH}" DOCKER_LOG="${TMP}/docker.log" DOCKER_PID_FILE="${TMP}/docker.pid" \
  SUDO_LOG="${TMP}/sudo.log" SANDBOX_TEST_USER="$(id -un)" \
  env "${common[@]}" SERVE_UP_CONTAINER_USER="$(id -un)" SANDBOX_ROOT="${SANDBOX_TMP}" \
  SANDBOX_JOB=job SHARE_GROUP="$(id -gn)" SANDBOX_WIPE_LIST="${TMP}/wipe-list" \
  SANDBOX_WIPE_HOME="${TMP}/wipe-home" WAIT_FOR_MEMORY_TIMEOUT_S=1 \
  "${ROOT}/tools/serve-up.sh" --boot --spec serial --draft-len 0 --socket-out "${TMP}/socket.path"
run="$(grep '^docker <run>' "${TMP}/docker.log")"
grep -Fq -- "<--entrypoint> <${worker_real}.sandboxed>" <<<"${run}"
grep -Fq -- "<type=bind,src=${worker_real}.sandboxed,dst=${worker_real}.sandboxed,readonly>" <<<"${run}"
! grep -Fq -- "<${worker_real}>" <<<"${run}"
docker_pid="$(cat "${TMP}/docker.pid")"
printf 'leftover\n' > "${TMP}/wipe/leftover"
PATH="${BIN}:${PATH}" DOCKER_LOG="${TMP}/docker.log" DOCKER_PID_FILE="${TMP}/docker.pid" \
  SUDO_LOG="${TMP}/sudo.log" SANDBOX_TEST_USER="$(id -un)" \
  SERVE_UP_CONTAINER_USER="$(id -un)" SANDBOX_ROOT="${SANDBOX_TMP}" SANDBOX_JOB=job \
  SHARE_GROUP="$(id -gn)" SANDBOX_WIPE_LIST="${TMP}/wipe-list" \
  SANDBOX_WIPE_HOME="${TMP}/wipe-home" SERVE_UP_MEMINFO="${TMP}/meminfo" \
  SERVE_UP_LOG_DIR="${TMP}/logs" MLXFAST_TARGET_SNAPSHOT_DIR="${TMP}/weights" \
  WAIT_FOR_MEMORY_TIMEOUT_S=1 \
  "${ROOT}/tools/serve-up.sh" --stop --socket "$(sed -n '1p' "${TMP}/socket.path")" >/dev/null
! kill -0 "${docker_pid}" 2>/dev/null
[[ ! -f "${TMP}/wipe/leftover" ]]
rm -f "${TMP}/socket.path"

: > "${TMP}/docker.log"
rm -f "${TMP}/docker.pid"
PATH="${BIN}:${PATH}" DOCKER_LOG="${TMP}/docker.log" DOCKER_PID_FILE="${TMP}/docker.pid" \
  SUDO_LOG="${TMP}/sudo.log" SANDBOX_TEST_USER="$(id -un)" \
  env "${common[@]}" SERVE_UP_CONTAINER_USER="$(id -un)" SANDBOX_ROOT="${SANDBOX_TMP}" \
  SANDBOX_JOB=job SHARE_GROUP="$(id -gn)" SANDBOX_WIPE_LIST="${TMP}/wipe-list" \
  SANDBOX_WIPE_HOME="${TMP}/wipe-home" \
  "${ROOT}/tools/serve-up.sh" --boot --spec serial --draft-len 0 --socket-out "${TMP}/socket.path"
socket="$(sed -n '1p' "${TMP}/socket.path")"
if PATH="${BIN}:${PATH}" DOCKER_LOG="${TMP}/docker.log" DOCKER_PID_FILE="${TMP}/docker.pid" \
    SUDO_LOG="${TMP}/sudo.log" SANDBOX_TEST_USER="$(id -un)" \
    SERVE_UP_CONTAINER_USER="$(id -un)" SANDBOX_ROOT="${SANDBOX_TMP}" SANDBOX_JOB=job \
    SHARE_GROUP="$(id -gn)" SANDBOX_WIPE_LIST="${TMP}/missing-wipe-list" \
    SANDBOX_WIPE_HOME="${TMP}/wipe-home" SERVE_UP_MEMINFO="${TMP}/meminfo" \
    SERVE_UP_LOG_DIR="${TMP}/logs" \
    "${ROOT}/tools/serve-up.sh" --stop --socket "${socket}" >"${TMP}/wipe-fail.out" 2>&1; then
  echo 'test-serve-up-boot-stop: failed sandbox wipe exited zero' >&2
  exit 1
fi
grep -Fq 'sandbox-wipe.sh: no wipe list' "${TMP}/wipe-fail.out"
rm -f "${TMP}/socket.path"

TREE="${TMP}/reference-tree"
mkdir -p "${TREE}/tools" "${TREE}/fixtures"
cp "${ROOT}/tools/serve-up.sh" "${ROOT}/tools/sandbox-wipe.sh" \
  "${ROOT}/tools/wait-for-memory.sh" "${TREE}/tools/"
cp "${ROOT}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json" "${TREE}/fixtures/"
: > "${TMP}/docker.log"
rm -f "${TMP}/docker.pid"
PATH="${BIN}:${PATH}" DOCKER_LOG="${TMP}/docker.log" DOCKER_PID_FILE="${TMP}/docker.pid" \
  SUDO_LOG="${TMP}/sudo.log" SANDBOX_TEST_USER="$(id -un)" \
  env "${common[@]}" SANDBOX_USER="$(id -un)" SERVE_UP_SANDBOX_TREE="${ROOT}" \
  SANDBOX_ROOT="${SANDBOX_TMP}" SANDBOX_JOB=job SHARE_GROUP="$(id -gn)" \
  SANDBOX_WIPE_LIST="${TMP}/wipe-list" SANDBOX_WIPE_HOME="${TMP}/wipe-home" \
  "${ROOT}/tools/serve-up.sh" --boot --spec serial --draft-len 0 --socket-out "${TMP}/socket.path"
candidate_socket="$(sed -n '1p' "${TMP}/socket.path")"
[[ "${candidate_socket}" == "${SANDBOX_TMP}/job/sock/"*/tensorfold-resident.sock ]]
PATH="${BIN}:${PATH}" DOCKER_LOG="${TMP}/docker.log" DOCKER_PID_FILE="${TMP}/docker.pid" \
  SUDO_LOG="${TMP}/sudo.log" SANDBOX_TEST_USER="$(id -un)" SANDBOX_USER="$(id -un)" \
  SERVE_UP_SANDBOX_TREE="${ROOT}" SANDBOX_ROOT="${SANDBOX_TMP}" SANDBOX_JOB=job \
  SHARE_GROUP="$(id -gn)" SANDBOX_WIPE_LIST="${TMP}/wipe-list" \
  SANDBOX_WIPE_HOME="${TMP}/wipe-home" SERVE_UP_MEMINFO="${TMP}/meminfo" \
  SERVE_UP_LOG_DIR="${TMP}/logs" MLXFAST_TARGET_SNAPSHOT_DIR="${TMP}/weights" \
  WAIT_FOR_MEMORY_TIMEOUT_S=1 \
  "${ROOT}/tools/serve-up.sh" --stop --socket "${candidate_socket}" >/dev/null

: > "${TMP}/docker.log"
rm -f "${TMP}/docker.pid"
PATH="${BIN}:${PATH}" DOCKER_LOG="${TMP}/docker.log" DOCKER_PID_FILE="${TMP}/docker.pid" \
  SUDO_LOG="${TMP}/sudo.log" SANDBOX_TEST_USER="$(id -un)" \
  env "${common[@]}" SANDBOX_USER="$(id -un)" SERVE_UP_SANDBOX_TREE="${ROOT}" \
  "${TREE}/tools/serve-up.sh" --boot --spec serial --draft-len 0 --socket-out "${TMP}/socket.path"
reference_socket="$(sed -n '1p' "${TMP}/socket.path")"
[[ "${reference_socket}" == "/tmp/p6-serve-$$/"*/tensorfold-resident.sock ]]
run="$(grep '^docker <run>' "${TMP}/docker.log")"
grep -Fq -- "<--entrypoint> <${worker_real}>" <<<"${run}"
! grep -Fq -- "<${worker_real}.sandboxed>" <<<"${run}"
PATH="${BIN}:${PATH}" DOCKER_LOG="${TMP}/docker.log" DOCKER_PID_FILE="${TMP}/docker.pid" \
  SERVE_UP_LOG_DIR="${TMP}/logs" SANDBOX_USER="$(id -un)" SERVE_UP_SANDBOX_TREE="${ROOT}" \
  "${TREE}/tools/serve-up.sh" --stop --socket "${reference_socket}" >/dev/null

# Default socket directory: with no SERVE_UP_SOCKET_DIR, a repo at a long path
# still gets a socket under the AF_UNIX limit, in a private per-user directory.
# SERVE_UP_TEST_SOCKET_BASE moves that directory under a private test base, so
# these cases never depend on, or change, the real /tmp default.
default_dir="${SOCKET_BASE}/tf-resident-$(id -u)"
nosock=()
for kv in "${common[@]}"; do [[ "${kv}" == SERVE_UP_SOCKET_DIR=* ]] || nosock+=("${kv}"); done
nosock+=(SERVE_UP_TEST_SOCKET_BASE="${SOCKET_BASE}")
DEEP="${TMP}/a-participant-clone-path-that-is-long/cudafast-qwen38-27b-tf-solver-work-checkout"
mkdir -p "${DEEP}/tools" "${DEEP}/fixtures" "${SOCKET_BASE}"
cp "${ROOT}/tools/serve-up.sh" "${ROOT}/tools/sandbox-wipe.sh" \
  "${ROOT}/tools/wait-for-memory.sh" "${DEEP}/tools/"
cp "${ROOT}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json" "${DEEP}/fixtures/"
(( ${#DEEP} + 72 > 107 ))

rm -f "${TMP}/socket.path"
env "${nosock[@]}" SERVE_UP_LAUNCHER=host \
  "${DEEP}/tools/serve-up.sh" --boot --spec serial --draft-len 0 --socket-out "${TMP}/socket.path"
socket="$(sed -n '1p' "${TMP}/socket.path")"
default_real="$(cd "${default_dir}" && pwd -P)"
[[ "${socket}" == "${default_real}/"*/tensorfold-resident.sock && ${#socket} -le 107 ]]
[[ "$(stat -c %a "${default_dir}" 2>/dev/null || stat -f %Lp "${default_dir}")" == 700 ]]
env SERVE_UP_LAUNCHER=host SERVE_UP_LOG_DIR="${TMP}/logs" \
  "${DEEP}/tools/serve-up.sh" --stop --socket "${socket}" >/dev/null
rm -rf "${default_dir}"

mkdir "${default_dir}"
chmod 777 "${default_dir}"
if env "${nosock[@]}" SERVE_UP_LAUNCHER=host \
    "${DEEP}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
    --socket-out "${TMP}/hostile.path" >"${TMP}/hostile.out" 2>&1; then
  echo 'test-serve-up-boot-stop: a group/other-writable socket dir was accepted' >&2
  exit 1
fi
grep -Fq "resident socket directory ${default_dir} is group- or other-writable" "${TMP}/hostile.out"

# A sandboxed resident uses its job's sock directory and must boot even when a
# hostile default directory is present.
rm -f "${TMP}/socket.path"
: > "${TMP}/docker.log"
rm -f "${TMP}/docker.pid"
PATH="${BIN}:${PATH}" DOCKER_LOG="${TMP}/docker.log" DOCKER_PID_FILE="${TMP}/docker.pid" \
  SUDO_LOG="${TMP}/sudo.log" SANDBOX_TEST_USER="$(id -un)" \
  env "${nosock[@]}" SANDBOX_USER="$(id -un)" SERVE_UP_SANDBOX_TREE="${ROOT}" \
  SANDBOX_ROOT="${SANDBOX_TMP}" SANDBOX_JOB=job SHARE_GROUP="$(id -gn)" \
  SANDBOX_WIPE_LIST="${TMP}/wipe-list" SANDBOX_WIPE_HOME="${TMP}/wipe-home" \
  "${ROOT}/tools/serve-up.sh" --boot --spec serial --draft-len 0 --socket-out "${TMP}/socket.path"
sandbox_socket="$(sed -n '1p' "${TMP}/socket.path")"
[[ "${sandbox_socket}" == "${SANDBOX_TMP}/job/sock/"*/tensorfold-resident.sock ]]
PATH="${BIN}:${PATH}" DOCKER_LOG="${TMP}/docker.log" DOCKER_PID_FILE="${TMP}/docker.pid" \
  SUDO_LOG="${TMP}/sudo.log" SANDBOX_TEST_USER="$(id -un)" SANDBOX_USER="$(id -un)" \
  SERVE_UP_SANDBOX_TREE="${ROOT}" SANDBOX_ROOT="${SANDBOX_TMP}" SANDBOX_JOB=job \
  SHARE_GROUP="$(id -gn)" SANDBOX_WIPE_LIST="${TMP}/wipe-list" \
  SANDBOX_WIPE_HOME="${TMP}/wipe-home" SERVE_UP_MEMINFO="${TMP}/meminfo" \
  SERVE_UP_LOG_DIR="${TMP}/logs" MLXFAST_TARGET_SNAPSHOT_DIR="${TMP}/weights" \
  WAIT_FOR_MEMORY_TIMEOUT_S=1 \
  "${ROOT}/tools/serve-up.sh" --stop --socket "${sandbox_socket}" >/dev/null
rmdir "${default_dir}"

ln -s "${TMP}" "${default_dir}"
if env "${nosock[@]}" SERVE_UP_LAUNCHER=host \
    "${DEEP}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
    --socket-out "${TMP}/hostile.path" >"${TMP}/hostile.out" 2>&1; then
  echo 'test-serve-up-boot-stop: a symlinked socket dir was accepted' >&2
  exit 1
fi
rm -f "${default_dir}"
grep -Fq "resident socket directory ${default_dir} is not a directory owned by this user" "${TMP}/hostile.out"

printf 'test-serve-up-boot-stop: PASS\n'

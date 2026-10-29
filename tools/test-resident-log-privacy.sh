#!/usr/bin/env bash
# Linux ACL test for ranked resident output isolation.
set -uo pipefail

TEST_NAME='test-resident-log-privacy.sh'
if ! command -v setfacl >/dev/null 2>&1; then
  printf '%s: SKIP: setfacl is absent\n' "${TEST_NAME}"
  exit 0
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
SOCKET_ROOT="/tmp/cuda-r4-resident-log-$$"
printed=0
socket_path=""

cleanup() {
  local rc=$?
  if [[ -n "${socket_path}" && -f "${socket_path}" ]]; then
    SERVE_UP_LOG_DIR="${PRIVATE_LOG:-${WORK}/none}" \
      "${ROOT}/tools/serve-up.sh" --stop --socket "$(sed -n '1p' "${socket_path}")" \
      >/dev/null 2>&1 || true
  fi
  [[ ! -f "${WORK}/resident.pid" ]] \
    || kill "$(cat "${WORK}/resident.pid")" >/dev/null 2>&1 || true
  rm -rf "${SOCKET_ROOT}" "${WORK}"
  if [[ "${printed}" == 0 ]]; then
    printf '%s: FAIL: unexpected exit %s\n' "${TEST_NAME}" "${rc}"
  fi
}
trap cleanup EXIT

fail() {
  printed=1
  printf '%s: FAIL: %s\n' "${TEST_NAME}" "$1"
  exit 1
}

SECOND_USER="${RESIDENT_LOG_TEST_USER:-}"
if [[ -z "${SECOND_USER}" ]]; then
  for candidate in qwen38sub nobody daemon; do
    if id "${candidate}" >/dev/null 2>&1 && [[ "${candidate}" != "$(id -un)" ]]; then
      SECOND_USER="${candidate}"
      break
    fi
  done
fi
[[ -n "${SECOND_USER}" ]] || fail 'no second OS user is available'

as_second_user() {
  if [[ "$(id -u)" == 0 ]] && command -v runuser >/dev/null 2>&1; then
    runuser -u "${SECOND_USER}" -- "$@"
  else
    sudo -n -u "${SECOND_USER}" -- "$@"
  fi
}
as_second_user true >/dev/null 2>&1 \
  || fail "cannot run a read check as ${SECOND_USER}"

chmod 0755 "${WORK}"
mkdir -m 0755 "${WORK}/weights" "${WORK}/readable-tree"
printf x > "${WORK}/weights/model.safetensors"
printf 'MemTotal: 10000000 kB\nMemAvailable: 20000000 kB\n' > "${WORK}/meminfo"

cat > "${WORK}/worker" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == runtime-worker ]]; then
  resident_pid="\$(cat "${WORK}/resident.pid")"
  printf '{"id":0,"ok":true,"protocol_version":1,"backend":"cuda","device":"mock","resident":{"pid":%s,"load_epoch":1}}\n' "\${resident_pid}"
  exit 0
fi
printf '%s\n' 'pair-1-private-prompt-and-answer' >&2
printf '%s\n' "\$\$" > "${WORK}/resident.pid"
socket=''
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
chmod 0755 "${WORK}/worker"

common=(
  TENSORFOLD_BENCH_WORKER="${WORK}/worker"
  MLXFAST_TARGET_SNAPSHOT_DIR="${WORK}/weights"
  SERVE_UP_SOCKET_DIR="${SOCKET_ROOT}"
  SERVE_UP_HEALTH_TIMEOUT_S=5
  SERVE_UP_TEST_READY_FILE=1
  SERVE_UP_TEST_REQUIRE_PROBE=1
  SERVE_UP_MEMINFO="${WORK}/meminfo"
  SERVE_UP_LAUNCHER=host
  SANDBOX_USER="${SECOND_USER}"
  SERVE_UP_SANDBOX_TREE="${WORK}/readable-tree"
)

# The checkout model has the same access and default ACL used by the workflow.
setfacl -m "u:${SECOND_USER}:r-x" -- "${WORK}/readable-tree" \
  || fail 'cannot set the sandbox access ACL'
setfacl -d -m "u:${SECOND_USER}:r-x" -- "${WORK}/readable-tree" \
  || fail 'cannot set the sandbox default ACL'
(umask 077; printf '%s\n' inherited-readable > "${WORK}/readable-tree/acl-probe")
as_second_user grep -Fq inherited-readable "${WORK}/readable-tree/acl-probe" \
  || fail 'the default ACL did not grant the second user read access'
rm -f "${WORK}/readable-tree/acl-probe"
inside_log="${WORK}/readable-tree/resident-logs"
inside_socket="${WORK}/inside.socket"
if env "${common[@]}" SERVE_UP_LOG_DIR="${inside_log}" \
    "${ROOT}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
    --socket-out "${inside_socket}" >"${WORK}/inside.out" 2>&1; then
  socket_path="${inside_socket}"
  fail 'an inside-checkout log directory with a default ACL was accepted'
fi
[[ "$(grep -Fxc 'serve-up.sh: REFUSE resident-log-dir' "${WORK}/inside.out")" == 1 ]] \
  || fail 'the inside-checkout refusal did not print its fixed class once'
if find "${inside_log}" -type f -name 'tensorfold-resident.*.log' -print -quit | grep -q .; then
  fail 'the refused inside-checkout case started a resident log'
fi

# A pre-existing ACL is refused. serve-up must not strip an ACL it did not own.
acl_log="${WORK}/acl-log"
mkdir -m 0700 "${acl_log}"
setfacl -m "u:${SECOND_USER}:---" -- "${acl_log}" \
  || fail 'cannot set the pre-existing ACL case'
if env "${common[@]}" SERVE_UP_LOG_DIR="${acl_log}" \
    "${ROOT}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
    --socket-out "${WORK}/acl.socket" >"${WORK}/acl.out" 2>&1; then
  fail 'a pre-existing resident log ACL was accepted'
fi
getfacl -cp -- "${acl_log}" | grep -Eq '^(user|group):[^:]|^mask:|^default:' \
  || fail 'serve-up stripped an ACL from a directory it did not create'

# Group-readable mode is also unsafe, even without an extended ACL.
readable_log="${WORK}/readable-log"
mkdir -m 0750 "${readable_log}"
if env "${common[@]}" SERVE_UP_LOG_DIR="${readable_log}" \
    "${ROOT}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
    --socket-out "${WORK}/readable.socket" >"${WORK}/readable.out" 2>&1; then
  fail 'a group-readable resident log directory was accepted'
fi
[[ "$(grep -Fxc 'serve-up.sh: REFUSE resident-log-dir' "${WORK}/readable.out")" == 1 ]] \
  || fail 'the group-readable refusal did not print its fixed class once'

# A created directory can inherit an ACL. serve-up removes that ACL, sets 0700,
# and then writes every resident file with mode 0600.
outside_parent="${WORK}/outside-parent"
mkdir -m 0711 "${outside_parent}"
setfacl -m "u:${SECOND_USER}:--x" -- "${outside_parent}" \
  || fail 'cannot set the private parent access ACL'
setfacl -d -m "u:${SECOND_USER}:r-x" -- "${outside_parent}" \
  || fail 'cannot set the private parent default ACL'
PRIVATE_LOG="${outside_parent}/resident-logs"
socket_path="${WORK}/pair-1.socket"
env "${common[@]}" SERVE_UP_LOG_DIR="${PRIVATE_LOG}" \
  "${ROOT}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
  --socket-out "${socket_path}" >"${WORK}/pair-1.out" 2>&1 \
  || fail 'the runner-private resident did not boot'

[[ "$(stat -c %a -- "${PRIVATE_LOG}")" == 700 ]] \
  || fail 'the created resident log directory is not mode 0700'
if getfacl -cp -- "${PRIVATE_LOG}" | grep -Eq '^(user|group):[^:]|^mask:|^default:'; then
  fail 'the created resident log directory kept an inherited ACL'
fi
resident_log="$(find "${PRIVATE_LOG}" -type f -name 'tensorfold-resident.*.log' -print -quit)"
[[ -n "${resident_log}" ]] || fail 'pair 1 wrote no resident log'
grep -Fq 'pair-1-private-prompt-and-answer' "${resident_log}" \
  || fail 'pair 1 did not write the replay probe to its resident log'
while IFS= read -r file; do
  [[ "$(stat -c %a -- "${file}")" == 600 ]] \
    || fail "resident output file is not mode 0600: ${file}"
done < <(find "${PRIVATE_LOG}" -type f -print)
if as_second_user test -r "${resident_log}" >/dev/null 2>&1; then
  fail 'the sandbox user can read the created resident log file'
fi

pair1_pids="$(tr -d '[:space:]' < "${WORK}/resident.pid" 2>/dev/null)"
[[ -n "${pair1_pids}" ]] || fail 'pair 1 left no resident pid file'
SERVE_UP_LOG_DIR="${PRIVATE_LOG}" "${ROOT}/tools/serve-up.sh" --stop --socket "$(sed -n '1p' "${socket_path}")" \
  >/dev/null 2>&1 || fail 'pair 1 resident stop failed'
socket_path=""
for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "${pair1_pids}" 2>/dev/null || break; sleep 0.5; done
if kill -0 "${pair1_pids}" 2>/dev/null; then
  fail 'pair 1 resident is still alive after stop'
fi

# Pair 2 gets a writable directory but cannot replay the known pair 1 log path.
pair2_dir="${WORK}/pair-2"
mkdir -m 0777 "${pair2_dir}"
if as_second_user /bin/sh -c 'cat "$1" > "$2"' sh "${resident_log}" \
    "${pair2_dir}/replay" >/dev/null 2>&1; then
  fail 'pair 2 replayed a log line written in pair 1'
fi
if grep -Fq 'pair-1-private-prompt-and-answer' "${pair2_dir}/replay" 2>/dev/null; then
  fail 'pair 1 resident output reached the pair 2 replay file'
fi

workflow="${ROOT}/.github/workflows/benchmark.yml"
grep -Fq 'resident_log_dir="$(mktemp -d "${RUNNER_TEMP}/resident-logs.XXXXXX")"' "${workflow}" \
  || fail 'the ranked workflow does not create its resident directory under RUNNER_TEMP'
grep -Fq "printf 'SERVE_UP_LOG_DIR=%s\\n' \"\${resident_log_dir}\" >> \"\${GITHUB_ENV}\"" "${workflow}" \
  || fail 'the ranked workflow does not export SERVE_UP_LOG_DIR'

printed=1
printf '%s: PASS\n' "${TEST_NAME}"

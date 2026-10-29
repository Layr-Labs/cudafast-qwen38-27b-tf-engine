#!/usr/bin/env bash
#
# run-engine-sandboxed.sh -- the engine benchd spawns for the CANDIDATE leg when
# participant code runs as the sandbox user (see docs/sandbox-design.md,
# "Execution boundary"), through the shim tools/stage-cuda-engine.sh writes at
# the staged engine path. TRUSTED: not in editablePaths.
#
# It runs the real adapter as ${SANDBOX_USER} and nothing else changes:
#
#   * stdin and stdout are benchd's pipes, handed straight to the adapter. The
#     Engine Protocol is stdio, so nothing here reads, buffers or parses a
#     message; sudo adds a few milliseconds at spawn, outside every timed window.
#   * the adapter's STDERR goes to a runner-private 0600 file, never to the job
#     log. The engine sees the hidden prompts, and the job log must not become a
#     way to carry them out.
#   * when the adapter exits, EVERY sandbox process is killed and the wrapper
#     checks that none remain, so nothing the candidate started runs during the
#     next control leg. That includes the candidate's resident, whose container
#     runs as the same user; benchd stops that leg's serve right after anyway.
#   * the exit status is the adapter's; a sandbox process that survives the kill
#     is a failure of its own.
#   * SIGTERM and SIGINT are forwarded to the adapter.
#   * the adapter runs under prlimit --fsize (64 MiB), --nproc and --as: no
#     file it writes, its stderr file included, grows past the fsize cap; a
#     write beyond it kills the writer with SIGXFSZ. Pipes are not files, so
#     stdio to benchd is unaffected. nproc/as bound fork and address-space bombs
#     (defaults 4096 and 16 GiB; override with ADAPTER_NPROC_LIMIT /
#     ADAPTER_AS_LIMIT bytes).
#   * the adapter's HOME and TMPDIR are the per-job $SANDBOX_ROOT/$SANDBOX_JOB/
#     adapter-home (2770, made by tools/engine/build.sh).
#     tools/sandbox-wipe.sh replaces it after every candidate leg.
#
# Env (the staged shim exports every one; benchd spawns it with a cleared env):
#   SANDBOX_USER        the sandbox user (qwen38sub)
#   SANDBOX_ENGINE_BIN  the real adapter, staged where the sandbox user can run
#                       it but not change it
#   SANDBOX_ROOT, SANDBOX_JOB  the per-job sandbox directory
#   SANDBOX_ENGINE_LOG_DIR  the job's private directory (RUNNER_TEMP at staging);
#                       the stderr file goes here. No fallback: benchd clears the
#                       environment, so the staged shim sets all of these.
# No -e: every step's status is checked explicitly (the adapter's comes from
# `wait`, and kill -1's is deliberately ignored; pgrep decides).
set -uo pipefail

die() { echo "run-engine-sandboxed.sh: $*" >&2; exit 1; }
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
# The adapter starts through this helper, which denies the kernel keyring calls
# (add_key, request_key, keyctl): keys outlive processes, and the sandbox user
# is shared by every submission. The resident and build containers get the
# same rule from Docker's default seccomp profile.
NO_KEYRING="${ROOT_DIR}/tools/no-keyring-exec.py"
# Largest file the adapter may write (bytes), its stderr file included.
ADAPTER_FSIZE=67108864
ADAPTER_NPROC_LIMIT="${ADAPTER_NPROC_LIMIT:-4096}"
ADAPTER_AS_LIMIT="${ADAPTER_AS_LIMIT:-17179869184}"

[[ -n "${SANDBOX_USER:-}" ]] || die "SANDBOX_USER is unset"
[[ -r "${NO_KEYRING}" ]] || die "the keyring guard ${NO_KEYRING} is missing"
# By absolute path: the adapter's environment has no PATH to search.
PYTHON3="$(command -v python3)" || die "python3 is required for the keyring guard"
[[ -x "${SANDBOX_ENGINE_BIN:-}" ]] || die "SANDBOX_ENGINE_BIN is not an executable adapter: '${SANDBOX_ENGINE_BIN:-}'"
[[ -n "${SANDBOX_ROOT:-}" && -n "${SANDBOX_JOB:-}" ]] || die "SANDBOX_ROOT or SANDBOX_JOB is unset"
adapter_home="${SANDBOX_ROOT}/${SANDBOX_JOB}/adapter-home"
[[ -d "${adapter_home}" && ! -L "${adapter_home}" ]] || die "the adapter's per-job home ${adapter_home} is missing (tools/engine/build.sh creates it)"

# mktemp creates the file 0600 and exclusively, so a name or symlink planted in a
# shared directory cannot redirect it.
log_dir="${SANDBOX_ENGINE_LOG_DIR:-}"
[[ -n "${log_dir}" && -d "${log_dir}" ]] || die "SANDBOX_ENGINE_LOG_DIR is not a directory: '${log_dir}'"
log="$(mktemp "${log_dir}/engine-stderr.XXXXXX")" || die "cannot create the adapter's stderr file in ${log_dir}"
echo "run-engine-sandboxed.sh: the adapter runs as ${SANDBOX_USER}; its stderr goes to ${log}, not the job log" >&2

# In the background so this shell can forward signals while it waits. stdin is
# redirected explicitly: a non-interactive `&` would otherwise give it /dev/null.
# HOME/TMPDIR are set inside the command: glibc strips TMPDIR from any setuid
# program's environment (sudo included), so --preserve-env cannot carry it.
started="${EPOCHREALTIME//[.,]/}"
sudo -n -u "${SANDBOX_USER}" --preserve-env=DS4_RESIDENT_SOCKET,BENCH_WORKER_RESIDENT_SOCKET \
  -- prlimit --fsize="${ADAPTER_FSIZE}" --nproc="${ADAPTER_NPROC_LIMIT}" --as="${ADAPTER_AS_LIMIT}" \
  -- env HOME="${adapter_home}" TMPDIR="${adapter_home}" \
  "${PYTHON3}" -I -S "${NO_KEYRING}" -- "${SANDBOX_ENGINE_BIN}" "$@" <&0 2>>"${log}" &
child=$!
forwarded=0
trap 'forwarded=1; kill -TERM "${child}" 2>/dev/null' TERM
trap 'forwarded=1; kill -INT "${child}" 2>/dev/null' INT

# Proof for the job log, from kernel state rather than anything the adapter
# writes: poll /proc for the started process and up to two levels below it
# (sudo with use_pty runs the command under a monitor process) until one of
# them reports Seccomp 2 and NoNewPrivs 1, then record that in a runner-owned
# file beside the stderr file. The workflow counts these files. The poll stops
# after about 3 s or when the process is gone, so an adapter that exits within
# a few milliseconds (setup's self-test) can leave no proof file; the run time
# recorded below lets the workflow tell those starts from the long ones.
guard_on() { # guard_on PID
  [[ "$(awk '/^Seccomp:/ { s = $2 } /^NoNewPrivs:/ { n = $2 } END { print s " " n }' \
    "/proc/$1/status" 2>/dev/null)" == "2 1" ]]
}
for (( poll = 0; poll < 300; poll++ )); do
  kill -0 "${child}" 2>/dev/null || break
  pids="${child}"; level="${child}"
  for _ in 1 2; do
    level="$(ps -o pid= --ppid "${level}" 2>/dev/null | xargs | tr ' ' ',')"
    [[ -n "${level}" ]] || break
    pids+=",${level}"
  done
  for pid in ${pids//,/ }; do
    if guard_on "${pid}"; then
      ( umask 077; echo "Seccomp=2 NoNewPrivs=1" > "${log}.guard" )
      break 2
    fi
  done
  sleep 0.01
done

# A trapped signal returns `wait` early; wait again for the adapter's own status.
wait "${child}"
status=$?
while [[ "${forwarded}" == 1 ]]; do
  forwarded=0
  wait "${child}"
  status=$?
done
( umask 077; echo "ms=$(( (${EPOCHREALTIME//[.,]/} - started) / 1000 ))" > "${log}.time" )

# Kill every sandbox process, then check none remain. `kill -1` exits non-zero
# when there is nothing left to kill, so its status means nothing; pgrep decides.
# Only pids are printed: a surviving process could carry prompt text in its
# command line.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  sudo -n -u "${SANDBOX_USER}" -- kill -KILL -1 >/dev/null 2>&1
  left="$(pgrep -u "${SANDBOX_USER}" 2>/dev/null | tr '\n' ' ')"
  [[ -z "${left// /}" ]] && exit "${status}"
  sleep 0.5
done
echo "::error::run-engine-sandboxed.sh: ${SANDBOX_USER} processes survived kill -KILL -1: ${left}" >&2
exit 1

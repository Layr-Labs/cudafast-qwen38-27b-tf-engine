#!/usr/bin/env bash
# Boot one native TensorFold resident for one benchmark leg.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKER="${TENSORFOLD_BENCH_WORKER:-${ROOT}/.build/release/bench-worker}"
FIXTURE="${ROOT}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json"
SANDBOX_USER="${SANDBOX_USER:-}"
SERVE_UP_CONTAINER_USER="${SERVE_UP_CONTAINER_USER:-}"
CONTAINER_PREFIX="${SERVE_UP_CONTAINER_PREFIX:-serve-up-}"
HOST_OVERHEAD_GIB=8
MEMORY_UTILIZATION=0.85
RESIDENT_LOG_LIMIT_BYTES=$((64 * 1024 * 1024))
RESIDENT_LOG_LIMIT_LINE='serve-up.sh: RESIDENT-LOG-LIMIT-EXCEEDED'
RESIDENT_LOG_DIR_REFUSAL='serve-up.sh: REFUSE resident-log-dir'
PROBE_OUTPUT_LIMIT_BYTES=$((1024 * 1024))
PROBE_OUTPUT_LIMIT_LINE='resident probe output exceeded limit'
PROBE_TIMEOUT_LINE='resident probe timed out'

die() { printf 'serve-up.sh: %s\n' "$*" >&2; exit 1; }
log() { printf 'serve-up.sh: %s\n' "$*" >&2; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"; }

refuse_log_dir() {
  printf '%s\n' "${RESIDENT_LOG_DIR_REFUSAL}" >&2
  die "$*"
}

prepare_log_dir() {
  local requested="$1" created=0 acl mode owner privacy_tree
  if [[ ! -e "${requested}" && ! -L "${requested}" ]]; then
    (umask 077; mkdir -p -- "${requested}") || die "cannot create resident log directory: ${requested}"
    created=1
  fi
  [[ -d "${requested}" && ! -L "${requested}" ]] \
    || refuse_log_dir "resident log directory is not a regular directory: ${requested}"
  log_dir="$(cd -- "${requested}" && pwd -P)"

  # Local runs use one OS user and can keep the checkout default. Ranked runs
  # have a distinct sandbox user. Their resident output must stay outside every
  # tree that user can read.
  if [[ -z "${SANDBOX_USER}" || "${SANDBOX_USER}" == "$(id -un)" ]]; then
    return 0
  fi
  [[ -n "${SERVE_UP_SANDBOX_TREE:-}" ]] \
    || refuse_log_dir "SERVE_UP_SANDBOX_TREE is unset for sandbox user ${SANDBOX_USER}"
  privacy_tree="$(cd -- "${SERVE_UP_SANDBOX_TREE}" 2>/dev/null && pwd -P)" \
    || refuse_log_dir "sandbox-readable tree is invalid: ${SERVE_UP_SANDBOX_TREE}"

  # GOTCHA: A default ACL can grant read access even when the caller used
  # umask 077. Remove ACLs only from a directory this process created.
  if [[ "${created}" == "1" ]] && command -v setfacl >/dev/null 2>&1; then
    setfacl -b -k -- "${log_dir}" >/dev/null 2>&1 \
      || refuse_log_dir "cannot remove inherited ACLs from created directory ${log_dir}"
    chmod 0700 -- "${log_dir}" \
      || refuse_log_dir "cannot set mode 0700 on created directory ${log_dir}"
  fi
  case "${log_dir}/" in
    "${privacy_tree}/"*)
      refuse_log_dir "resident log directory is inside sandbox-readable tree ${privacy_tree}"
      ;;
  esac
  command -v getfacl >/dev/null 2>&1 \
    || refuse_log_dir "getfacl is required for a ranked resident log directory"
  acl="$(getfacl -cp -- "${log_dir}" 2>/dev/null)" \
    || refuse_log_dir "cannot inspect ACLs on resident log directory ${log_dir}"
  if grep -Eq '^(user|group):[^:]|^mask:|^default:' <<< "${acl}"; then
    refuse_log_dir "resident log directory has an access or default ACL: ${log_dir}"
  fi
  mode="$(stat -c %a -- "${log_dir}" 2>/dev/null)" \
    || refuse_log_dir "cannot inspect resident log directory mode: ${log_dir}"
  owner="$(stat -c %u -- "${log_dir}" 2>/dev/null)" \
    || refuse_log_dir "cannot inspect resident log directory owner: ${log_dir}"
  [[ "${owner}" == "$(id -u)" ]] \
    || refuse_log_dir "resident log directory is not owned by the runner: ${log_dir}"
  [[ "${mode}" == "700" ]] \
    || refuse_log_dir "resident log directory mode is ${mode}, not 700: ${log_dir}"
}

bounded_resident_log() {
  python3 /dev/fd/3 "$1" "$2" "$3" "$4" "${RESIDENT_LOG_LIMIT_LINE}" 3<<'PY'
import pathlib
import subprocess
import sys

output, overflow_file, container, limit_text, class_line = sys.argv[1:]
limit = int(limit_text)
line = class_line.encode()
remaining = limit - len(line) - 2
source = sys.stdin.buffer
with open(output, "wb") as sink:
    while remaining:
        chunk = source.read(min(1024 * 1024, remaining))
        if not chunk:
            raise SystemExit(0)
        sink.write(chunk)
        remaining -= len(chunk)
    if not source.read(1):
        raise SystemExit(0)
    sink.write(b"\n" + line + b"\n")
pathlib.Path(overflow_file).touch(mode=0o600)
pathlib.Path(overflow_file).chmod(0o600)
subprocess.run(
    ["docker", "stop", "-t", "10", container],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
)
raise SystemExit(1)
PY
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -- "$1" | awk '{print $1}'
  else
    shasum -a 256 -- "$1" | awk '{print $1}'
  fi
}

refuse_snapshot_symlinks() {
  local dir="$1" label="$2"
  if find "$dir" -type l -print -quit | grep -q .; then
    die "${label} snapshot contains a symlink; run the ranked workflow snapshot staging step"
  fi
}

verify_pinned_files() {
  local dir="$1" pins="$2" label="$3" rel want_bytes want_sha file got_bytes got_sha count=0
  while IFS=$'\t' read -r rel want_bytes want_sha; do
    [[ -n "$rel" ]] || continue
    case "$rel" in /*|../*|*/../*|*/..) die "invalid ${label} pin path: ${rel}" ;; esac
    file="${dir}/${rel}"
    [[ -f "$file" && ! -L "$file" ]] || die "pinned ${label} file is missing: ${rel}"
    got_bytes="$(wc -c < "$file" | tr -d '[:space:]')"
    [[ "$got_bytes" == "$want_bytes" ]] \
      || die "byte-count mismatch for ${label} ${rel}: staged ${got_bytes}, pinned ${want_bytes}"
    got_sha="$(sha256_of "$file")"
    [[ "$got_sha" == "$want_sha" ]] \
      || die "sha256 mismatch for ${label} ${rel}: staged ${got_sha}, pinned ${want_sha}"
    count=$((count + 1))
  done <<< "$pins"
  (( count > 0 )) || die "the fixture pins no ${label} files"
  log "verified ${count} pinned ${label} files"
}

stage_snapshots() {
  local dest_root="" env_out="" share_group="" target_source drafter_source
  local target_dir drafter_dir target_pins drafter_pins runner_uid
  shift
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dest-root) need_value "$@"; dest_root="$2"; shift 2 ;;
      --env-out) need_value "$@"; env_out="$2"; shift 2 ;;
      --group) need_value "$@"; share_group="$2"; shift 2 ;;
      *) die "--stage-snapshots: unknown argument '$1'" ;;
    esac
  done
  [[ -n "$dest_root" ]] || die "--stage-snapshots requires --dest-root DIR"
  [[ -n "$env_out" ]] || die "--stage-snapshots requires --env-out FILE"
  [[ -n "$share_group" ]] || die "--stage-snapshots requires --group GROUP"
  [[ ! -e "$dest_root" && ! -L "$dest_root" ]] || die "snapshot staging destination already exists: <sandbox-root>/weights"
  target_source="${MLXFAST_TARGET_SNAPSHOT_DIR:-}"
  drafter_source="${MLXFAST_DRAFTER_SNAPSHOT_DIR:-}"
  [[ -d "$target_source" ]] || die "MLXFAST_TARGET_SNAPSHOT_DIR is missing"
  [[ -d "$drafter_source" ]] || die "MLXFAST_DRAFTER_SNAPSHOT_DIR is missing"
  [[ -f "$FIXTURE" ]] || die "track fixture is missing: fixtures/qwen3_8_27b_tf_cuda_v1_track.json"

  target_dir="${dest_root}/target"
  drafter_dir="${dest_root}/drafter"
  mkdir -p "$target_dir" "$drafter_dir"
  # cp's own message would print box paths; refuse with role names instead
  cp -RL -- "${target_source}/." "${target_dir}/" 2>/dev/null \
    || die "cannot copy <target-snapshot> into <sandbox-root>/weights/target"
  cp -RL -- "${drafter_source}/." "${drafter_dir}/" 2>/dev/null \
    || die "cannot copy <drafter-snapshot> into <sandbox-root>/weights/drafter"
  refuse_snapshot_symlinks "$target_dir" "staged target"
  refuse_snapshot_symlinks "$drafter_dir" "staged drafter"

  target_pins="$(jq -r '.target.files[]? | [.path, (.bytes|tostring), .sha256] | @tsv' "$FIXTURE")"
  verify_pinned_files "$target_dir" "$target_pins" "target"
  drafter_pins="$(jq -r '(.target.drafter.files // .drafter.files // [])[] | [.path, (.bytes|tostring), .sha256] | @tsv' "$FIXTURE")"
  if [[ -n "$drafter_pins" ]]; then
    verify_pinned_files "$drafter_dir" "$drafter_pins" "drafter"
  else
    log "the fixture has no drafter file list; recording staged drafter sha256 values"
    (
      cd "$drafter_dir"
      count=0
      while IFS= read -r -d '' file; do
        printf '%s  %s\n' "$(sha256_of "$file")" "$file"
        count=$((count + 1))
      done < <(find . -type f -print0)
      (( count > 0 ))
    ) || die "the staged drafter snapshot has no files"
  fi

  # The runner owns the copies. The sandbox group can only read and traverse.
  chgrp -R -- "$share_group" "$dest_root"
  find "$dest_root" -type d -exec chmod 2750 {} +
  find "$dest_root" -type f -exec chmod 0640 {} +
  runner_uid="$(id -u)"
  if find "$dest_root" ! -user "$runner_uid" -print -quit | grep -q .; then
    die "a staged snapshot entry is not owned by the runner"
  fi
  {
    printf 'MLXFAST_TARGET_SNAPSHOT_DIR=%s\n' "$(cd -- "$target_dir" && pwd -P)"
    printf 'MLXFAST_DRAFTER_SNAPSHOT_DIR=%s\n' "$(cd -- "$drafter_dir" && pwd -P)"
    printf 'MLXFAST_WEIGHTS_PATH=%s\n' "$(cd -- "$target_dir" && pwd -P)"
    printf 'SERVE_UP_WEIGHTS_DIR=%s\n' "$(cd -- "$target_dir" && pwd -P)"
  } >> "$env_out"
  log "staged dereferenced snapshots in <sandbox-root>/weights"
}

if [[ "${1:-}" == "--stage-snapshots" ]]; then
  stage_snapshots "$@"
  exit 0
fi

# benchd boots both trees with one environment. Only the named candidate tree
# runs as the sandbox user.
if [[ -z "${SERVE_UP_CONTAINER_USER}" && -n "${SERVE_UP_SANDBOX_TREE:-}" ]]; then
  sandbox_tree="$(cd -- "${SERVE_UP_SANDBOX_TREE}" 2>/dev/null && pwd -P)" \
    || die "invalid SERVE_UP_SANDBOX_TREE: ${SERVE_UP_SANDBOX_TREE}"
  if [[ "${sandbox_tree}" == "${ROOT}" ]]; then
    [[ -n "${SANDBOX_USER}" ]] \
      || die "SERVE_UP_SANDBOX_TREE names this tree, but SANDBOX_USER is unset"
    SERVE_UP_CONTAINER_USER="${SANDBOX_USER}"
  fi
fi

resident_tag() {
  local socket="$1" base
  base="$(basename -- "${socket}")"
  if [[ "${base}" == "tensorfold-resident.sock" ]]; then
    basename -- "$(dirname -- "${socket}")"
  else
    printf '%s\n' "${base#tensorfold-resident.}" | sed 's/\.sock$//'
  fi
}

sandbox_leg_teardown_wipe() {
  local user="$1" left=""
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sudo -n -u "${user}" -- kill -KILL -1 >/dev/null 2>&1 || true
    left="$(pgrep -u "${user}" 2>/dev/null | tr '\n' ' ' || true)"
    [[ -z "${left// /}" ]] && break
    sleep 0.5
  done
  [[ -z "${left// /}" ]] || die "${user} processes survived kill -KILL -1: ${left}"
  SANDBOX_USER="${user}" "${ROOT}/tools/sandbox-wipe.sh"
  bash "${ROOT}/tools/wait-for-memory.sh"
}

stop_resident() {
  local socket="$1" tag pidfile overflow_file pid="" container="" overflow=0
  tag="$(resident_tag "${socket}")"
  pidfile="${SERVE_UP_LOG_DIR:-${ROOT}/.build/resident}/tensorfold-resident.${tag}.pid"
  overflow_file="${SERVE_UP_LOG_DIR:-${ROOT}/.build/resident}/tensorfold-resident.${tag}.overflow"
  [[ ! -f "${overflow_file}" ]] || overflow=1
  if [[ -f "${pidfile}" ]]; then
    pid="$(sed -n '1p' "${pidfile}" | tr -d '[:space:]')"
    container="$(sed -n '2p' "${pidfile}" | tr -d '[:space:]')"
  fi
  if [[ "${container}" =~ ^${CONTAINER_PREFIX}[0-9]+-[0-9]{8}T[0-9]{6}Z$ ]] \
      && command -v docker >/dev/null 2>&1; then
    docker stop -t 10 "${container}" >/dev/null 2>&1 || true
    docker rm -f "${container}" >/dev/null 2>&1 || true
    log "resident container ${container} stopped"
  elif [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 0.2
    done
    kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}" 2>/dev/null || true
    log "resident pid ${pid} stopped"
  else
    log "no running resident for ${socket}; nothing to stop"
  fi
  rm -f "${socket}" "${pidfile}" 2>/dev/null || true
  if [[ -n "${SERVE_UP_CONTAINER_USER}" ]]; then
    sandbox_leg_teardown_wipe "${SERVE_UP_CONTAINER_USER}"
  fi
  if [[ "${overflow}" == "1" ]]; then
    printf '%s\n' "${RESIDENT_LOG_LIMIT_LINE}" >&2
    return 1
  fi
}

if [[ "${1:-}" == "--stop" ]]; then
  shift
  [[ "${1:-}" == "--socket" && -n "${2:-}" ]] || die "--stop requires --socket PATH"
  stop_resident "$2"
  exit 0
fi

mode="wrapper"
spec=""
depth=""
socket_out=""
if [[ "${1:-}" == "--boot" ]]; then
  mode="boot"
  shift
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --spec) need_value "$@"; spec="$2"; shift 2 ;;
      --draft-len) need_value "$@"; depth="$2"; shift 2 ;;
      --socket-out) need_value "$@"; socket_out="$2"; shift 2 ;;
      *) die "--boot: unknown argument '$1'" ;;
    esac
  done
else
  [[ $# -gt 0 ]] || die "use --boot, --stop, or a command"
  if [[ "${SERVE_UP_SPECULATIVE:-0}" == "1" ]]; then
    spec="dflash"
    depth="${SERVE_UP_SPEC_DRAFT_LEN:-1}"
  else
    spec="serial"
    depth="0"
  fi
fi

case "${spec}:${depth}" in
  serial:0) ;;
  dflash:1) ;;
  *) die "supported forms are serial depth 0 and dflash depth 1" ;;
esac

weights="${SERVE_UP_WEIGHTS_DIR:-${MLXFAST_TARGET_SNAPSHOT_DIR:-}}"
[[ -d "${weights}" ]] || die "target snapshot directory is missing: '${weights:-unset}'"
refuse_snapshot_symlinks "${weights}" "target"
weights="$(cd -- "${weights}" && pwd -P)"
[[ -x "${WORKER}" ]] || die "worker is not executable: ${WORKER}"
WORKER="$(cd -- "$(dirname -- "${WORKER}")" && pwd -P)/$(basename -- "${WORKER}")"
CONTAINER_WORKER="${WORKER}"
if [[ -n "${SERVE_UP_CONTAINER_USER}" ]]; then
  CONTAINER_WORKER="${WORKER}.sandboxed"
  [[ -f "${CONTAINER_WORKER}" && ! -L "${CONTAINER_WORKER}" && -x "${CONTAINER_WORKER}" ]] \
    || die "sandboxed worker is not a regular executable: ${CONTAINER_WORKER}"
fi
[[ -f "${FIXTURE}" ]] || die "track fixture is missing: ${FIXTURE}"

drafter="${MLXFAST_DRAFTER_SNAPSHOT_DIR:-}"
if [[ "${spec}" == "dflash" ]]; then
  [[ -d "${drafter}" ]] || die "MLXFAST_DRAFTER_SNAPSHOT_DIR is missing"
  refuse_snapshot_symlinks "${drafter}" "drafter"
  drafter="$(cd -- "${drafter}" && pwd -P)"
fi

memory_plan() {
  local meminfo="${SERVE_UP_MEMINFO:-/proc/meminfo}"
  local min_cache_gib="${SERVE_UP_MIN_CACHE_GIB:-${HOST_OVERHEAD_GIB}}"
  [[ -r "${meminfo}" ]] || die "cannot read ${meminfo}; refusing to start a load blind"
  local total_kb available_kb body_bytes total_bytes available_bytes reserve_bytes want_bytes
  total_kb="$(awk '/^MemTotal:/ {print $2; exit}' "${meminfo}")"
  available_kb="$(awk '/^MemAvailable:/ {print $2; exit}' "${meminfo}")"
  [[ "${total_kb}" =~ ^[0-9]+$ && "${available_kb}" =~ ^[0-9]+$ ]] \
    || die "${meminfo} lacks MemTotal/MemAvailable; refusing to start a load blind"
  body_bytes="$(python3 - "${weights}" <<'PY'
import pathlib, sys
print(sum(p.stat().st_size for p in pathlib.Path(sys.argv[1]).glob("*.safetensors")))
PY
)"
  (( body_bytes > 0 )) || die "memory plan REFUSES the load: no *.safetensors in ${weights}"
  total_bytes=$((total_kb * 1024))
  available_bytes=$((available_kb * 1024))
  reserve_bytes="$(awk -v t="${total_bytes}" -v u="${MEMORY_UTILIZATION}" 'BEGIN {printf "%.0f", t*u}')"
  want_bytes=$((body_bytes + min_cache_gib * 1024 * 1024 * 1024))
  printf 'serve-up.sh: memory plan for this leg (bytes)\n' >&2
  printf '  weights=%s reserve=%s available=%s required=%s\n' \
    "${body_bytes}" "${reserve_bytes}" "${available_bytes}" "${want_bytes}" >&2
  (( want_bytes <= reserve_bytes )) || die "memory plan REFUSES the load: weights plus cache exceed the reservation"
  (( want_bytes <= available_bytes )) || die "memory plan REFUSES the load: available memory is too low"
  log "memory plan ACCEPTS the load"
}

memory_plan

launcher="${SERVE_UP_LAUNCHER:-docker}"
case "${launcher}" in docker|host) ;; *) die "SERVE_UP_LAUNCHER must be docker or host" ;; esac
[[ -z "${SERVE_UP_CONTAINER_USER}" || "${launcher}" == docker ]] \
  || die "SERVE_UP_CONTAINER_USER requires the docker launcher"

# AF_UNIX socket paths stop at 107 bytes, and the leg suffix below takes about
# 72, so a default under the repo breaks for ordinary clone paths. The default
# is a short private per-user directory in /tmp (not $TMPDIR, which can itself
# be long); SERVE_UP_SOCKET_DIR still overrides it. A sandboxed resident uses
# its job's sock directory below and never touches the default.
if [[ -n "${SERVE_UP_SOCKET_DIR:-}" ]]; then
  socket_dir="${SERVE_UP_SOCKET_DIR}"
  mkdir -p "${socket_dir}"
elif [[ -n "${SERVE_UP_CONTAINER_USER}" ]]; then
  socket_dir=""
else
  socket_dir="${SERVE_UP_TEST_SOCKET_BASE:-/tmp}/tf-resident-$(id -u)"
  [[ -e "${socket_dir}" || -L "${socket_dir}" ]] || mkdir -m 700 "${socket_dir}" 2>/dev/null || true
  [[ -d "${socket_dir}" && ! -L "${socket_dir}" && -O "${socket_dir}" ]] \
    || die "resident socket directory ${socket_dir} is not a directory owned by this user"
  [[ -z "$(find "${socket_dir}" -maxdepth 0 \( -perm -020 -o -perm -002 \) -print)" ]] \
    || die "resident socket directory ${socket_dir} is group- or other-writable"
fi
log_dir=""
prepare_log_dir "${SERVE_UP_LOG_DIR:-${ROOT}/.build/resident}"
[[ -z "${socket_dir}" ]] || socket_dir="$(cd -- "${socket_dir}" && pwd -P)"
tag="$$-$(date -u +%Y%m%dT%H%M%SZ)"
socket="${socket_dir}/tensorfold-resident.${tag}.sock"
container_name="${CONTAINER_PREFIX}${tag}"
container_uid="$(id -u):$(id -g)"
if [[ -n "${SERVE_UP_CONTAINER_USER}" ]]; then
  [[ -n "${SANDBOX_ROOT:-}" && -n "${SANDBOX_JOB:-}" ]] \
    || die "sandbox resident needs SANDBOX_ROOT and SANDBOX_JOB"
  socket_dir="${SANDBOX_ROOT}/${SANDBOX_JOB}/sock"
  [[ -d "${socket_dir}" ]] || die "sandbox socket directory is missing: ${socket_dir}"
  leg_dir="${socket_dir}/${tag}"
  mkdir -m 2770 "${leg_dir}"
  socket="${leg_dir}/tensorfold-resident.sock"
  share_gid="$(getent group "${SHARE_GROUP:-qwen38share}" | cut -d: -f3)"
  [[ -n "${share_gid}" ]] || die "group ${SHARE_GROUP:-qwen38share} does not exist"
  container_uid="$(id -u "${SERVE_UP_CONTAINER_USER}"):${share_gid}"
else
  leg_dir="${socket_dir}/${tag}"
  mkdir -m 700 "${leg_dir}"
  socket="${leg_dir}/tensorfold-resident.sock"
fi
[[ ${#socket} -le 107 ]] || die "resident socket path is over the 107-byte AF_UNIX limit: ${socket}"
resident_log="${log_dir}/tensorfold-resident.${tag}.log"
overflow_file="${log_dir}/tensorfold-resident.${tag}.overflow"
probe_overflow_file="${log_dir}/tensorfold-resident.${tag}.probe-overflow"
probe_error_file="${log_dir}/tensorfold-resident.${tag}.probe-error"
run_home="${leg_dir}/run"
if [[ -n "${SERVE_UP_CONTAINER_USER}" ]]; then
  mkdir -m 2770 "${run_home}"
else
  mkdir -m 700 "${run_home}"
fi
args=(resident --weights "${weights}" --socket "${socket}")
[[ "${spec}" == "serial" ]] || args+=(--drafter "${drafter}")

probe_resident() {
  local probe_timeout="$1"
  local -a command
  if [[ "${launcher}" == docker ]]; then
    command=(docker exec -i --user "${container_uid}"
      --env "HOME=${run_home}" --env "TMPDIR=${run_home}"
      --env "BENCH_WORKER_RESIDENT_SOCKET=${socket}"
      "${container_name}" "${CONTAINER_WORKER}" runtime-worker --weights "${weights}")
  else
    command=(env -i HOME="${run_home}" TMPDIR="${run_home}" PATH=/usr/bin:/bin
      BENCH_WORKER_RESIDENT_SOCKET="${socket}"
      "${WORKER}" runtime-worker --weights "${weights}")
  fi
  python3 - "${probe_timeout}" "${PROBE_OUTPUT_LIMIT_BYTES}" \
    "${probe_overflow_file}" "$([[ "${launcher}" == docker ]] && printf '%s' "${container_name}")" \
    "${command[@]}" <<'PY'
import json
import os
import pathlib
import selectors
import signal
import subprocess
import sys
import time

timeout = float(sys.argv[1])
limit = int(sys.argv[2])
overflow_file = sys.argv[3]
container = sys.argv[4]
command = sys.argv[5:]

process = subprocess.Popen(
    command,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    start_new_session=True,
)
selector = selectors.DefaultSelector()
selector.register(process.stdout, selectors.EVENT_READ, "stdout")
selector.register(process.stderr, selectors.EVENT_READ, "stderr")
kept = {"stdout": bytearray(), "stderr": bytearray()}
deadline = time.monotonic() + timeout

def kill_probe(stop_container=False):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    remaining = max(0.0, deadline - time.monotonic())
    try:
        process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            pass
    if stop_container and container:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        try:
            subprocess.run(
                ["docker", "stop", "-t", "0", container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=remaining,
            )
        except subprocess.TimeoutExpired:
            pass

def refuse_timeout():
    kill_probe(stop_container=True)
    print("resident probe timed out", file=sys.stderr)
    raise SystemExit(1)

overflow = False
while selector.get_map():
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        refuse_timeout()
    events = selector.select(remaining)
    if not events:
        refuse_timeout()
    for key, _ in events:
        output = kept[key.data]
        allowance = limit - len(output)
        chunk = os.read(key.fileobj.fileno(), min(65536, allowance + 1))
        if not chunk:
            selector.unregister(key.fileobj)
            key.fileobj.close()
            continue
        output.extend(chunk[:allowance])
        if len(chunk) > allowance:
            overflow = True
            break
    if overflow:
        break

if overflow:
    kill_probe(stop_container=True)
    pathlib.Path(overflow_file).write_text(
        f"{len(kept['stdout'])} {len(kept['stderr'])}\n",
        encoding="ascii",
    )
    pathlib.Path(overflow_file).chmod(0o600)
    print("resident probe output exceeded limit", file=sys.stderr)
    raise SystemExit(1)

remaining = deadline - time.monotonic()
if remaining <= 0:
    refuse_timeout()
try:
    returncode = process.wait(timeout=remaining)
except subprocess.TimeoutExpired:
    refuse_timeout()
stdout = kept["stdout"].decode("utf-8", errors="replace")
stderr = kept["stderr"].decode("utf-8", errors="replace")
if returncode != 0:
    print(stderr.rstrip(), file=sys.stderr)
    raise SystemExit(1)
try:
    hello = json.loads(stdout.splitlines()[0])
    # The outer runtime-worker hello may omit `resident`: benchd
    # retains one hello per paired run, and a re-booted control resident changed it. The
    # identity is optional here (benchd official.rs:215 accepts both absent); when present it must be an
    # object with a valid pid and load_epoch (engine-protocol-v1 schema), so `null` is refused.
    resident = hello.get("resident")
except (IndexError, TypeError, AttributeError, json.JSONDecodeError) as error:
    print(f"invalid resident hello: {error}", file=sys.stderr)
    raise SystemExit(1)
if hello.get("id") != 0 or hello.get("ok") is not True or hello.get("protocol_version") != 1:
    print("resident hello has the wrong protocol identity", file=sys.stderr)
    raise SystemExit(1)
if not isinstance(hello.get("backend"), str) or not hello["backend"]:
    print("resident hello has no backend", file=sys.stderr)
    raise SystemExit(1)
if "resident" in hello:
    if not isinstance(resident, dict):
        print("resident hello has an invalid resident field", file=sys.stderr)
        raise SystemExit(1)
    if type(resident.get("pid")) is not int or not 0 < resident["pid"] <= 0xFFFFFFFF:
        print("resident hello has no valid pid", file=sys.stderr)
        raise SystemExit(1)
    if type(resident.get("load_epoch")) is not int or not 0 <= resident["load_epoch"] <= 0xFFFFFFFFFFFFFFFF:
        print("resident hello has no valid load_epoch", file=sys.stderr)
        raise SystemExit(1)
print(json.dumps(hello, separators=(",", ":")))
PY
}

(umask 077; : > "${resident_log}")
rm -f "${probe_overflow_file}" "${probe_error_file}"
log "booting native resident: ${weights} (${spec}${depth}) via ${launcher}"
if [[ "${launcher}" == docker ]]; then
  command -v docker >/dev/null 2>&1 || die "docker is required"
  image="$(jq -r '.serve_configuration.container_image // empty' "${FIXTURE}")"
  digest="$(jq -r '.serve_configuration.container_digest // empty' "${FIXTURE}")"
  [[ -n "${image}" && "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || die "the fixture pins no CUDA container image and digest"
  image_ref="${image%%:*}@${digest}"
  docker image inspect "${image_ref}" >/dev/null 2>&1 \
    || die "the pinned CUDA image is not present: ${image_ref}"
  mounts=(--mount "type=bind,src=${ROOT},dst=${ROOT},readonly"
    --mount "type=bind,src=${CONTAINER_WORKER},dst=${CONTAINER_WORKER},readonly"
    --mount "type=bind,src=${weights},dst=${weights},readonly"
    --mount "type=bind,src=${leg_dir},dst=${leg_dir}")
  [[ "${spec}" == serial ]] || mounts+=(--mount "type=bind,src=${drafter},dst=${drafter},readonly")
  docker run --rm --init --name "${container_name}" \
    --pids-limit 4096 --memory 64g \
    --log-opt max-size=64m --log-opt max-file=1 \
    --gpus all --ipc host --network none \
    --user "${container_uid}" --cap-drop ALL --security-opt no-new-privileges \
    --read-only --tmpfs /tmp:rw,exec,size=8g \
    "${mounts[@]}" \
    --env "HOME=${run_home}" --env "TMPDIR=${run_home}" \
    --entrypoint "${CONTAINER_WORKER}" "${image_ref}" "${args[@]}" 2>&1 \
    | bounded_resident_log "${resident_log}" "${overflow_file}" "${container_name}" \
        "${RESIDENT_LOG_LIMIT_BYTES}" >/dev/null 2>&1 &
else
  env -i HOME="${run_home}" TMPDIR="${run_home}" PATH=/usr/bin:/bin \
    "${WORKER}" "${args[@]}" >"${resident_log}" 2>&1 &
fi
pid=$!
(umask 077; printf '%s\n%s\n' "${pid}" \
  "$([[ "${launcher}" == docker ]] && printf '%s' "${container_name}")" \
  > "${log_dir}/tensorfold-resident.${tag}.pid")
keep_resident=0
teardown() {
  [[ "${keep_resident}" == "1" ]] || stop_resident "${socket}"
}
trap teardown EXIT
trap 'exit 130' INT TERM

timeout="${SERVE_UP_HEALTH_TIMEOUT_S:-1800}"
[[ "${timeout}" =~ ^[1-9][0-9]*$ ]] || die "SERVE_UP_HEALTH_TIMEOUT_S must be a positive integer"
deadline=$(($(date +%s) + timeout))
ready=0
hello_json=""
while (( $(date +%s) < deadline )); do
  [[ ! -f "${overflow_file}" ]] || die "RESIDENT-LOG-LIMIT-EXCEEDED"
  if [[ -S "${socket}" || ( "${SERVE_UP_TEST_REQUIRE_PROBE:-0}" == "1" \
      && "${SERVE_UP_TEST_READY_FILE:-0}" == "1" && -f "${socket}" ) ]]; then
    remaining=$((deadline - $(date +%s)))
    (( remaining > 0 )) || break
    probe_timeout=10
    (( remaining >= probe_timeout )) || probe_timeout="${remaining}"
    (umask 077; : > "${probe_error_file}")
    if hello_json="$(probe_resident "${probe_timeout}" 2>"${probe_error_file}")"; then
      ready=1
      break
    fi
    if [[ -f "${probe_overflow_file}" ]]; then
      printf '%s\n' "${PROBE_OUTPUT_LIMIT_LINE}" >&2
      exit 1
    fi
    if grep -Fxq "${PROBE_TIMEOUT_LINE}" "${probe_error_file}"; then
      printf '%s\n' "${PROBE_TIMEOUT_LINE}" >&2
      exit 1
    fi
  elif [[ "${SERVE_UP_TEST_READY_FILE:-0}" == "1" && -f "${socket}" ]]; then
    hello_json="$(jq -cn --argjson pid "${pid}" \
      '{id:0,ok:true,protocol_version:1,backend:"cuda",device:"test",resident:{pid:$pid,load_epoch:0}}')"
    ready=1
    break
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    sed -n '1,120p' "${resident_log}" >&2 || true
    die "resident exited before it opened the socket"
  fi
  (( $(date +%s) < deadline )) || break
  sleep 1
done
[[ ! -f "${overflow_file}" ]] || die "RESIDENT-LOG-LIMIT-EXCEEDED"
if [[ "${ready}" != "1" ]]; then
  die "resident did not open the socket in ${timeout} seconds"
fi
identity_file="${log_dir}/serve-identity.json"
(umask 077; jq -n \
  --arg socket "${socket}" --arg pid "${pid}" --arg container "${container_name}" \
  --arg launcher "${launcher}" --arg weights "${weights}" --arg spec "${spec}" \
  --argjson depth "${depth}" --argjson hello "${hello_json}" \
  '{engine:"tensorfold",weight_owner:"tensorfold-resident",resident_socket:$socket,
    resident_pid:$pid,container:$container,launcher:$launcher,weights_dir:$weights,
    spec_config:$spec,declared_draft_len:$depth,hello:$hello}' > "${identity_file}")
chmod 0600 "${identity_file}"
log "resident healthy at ${socket}"

if [[ "${mode}" == "boot" ]]; then
  [[ -n "${socket_out}" ]] || die "--boot requires --socket-out FILE"
  mkdir -p "$(dirname -- "${socket_out}")"
  printf '%s\n' "${socket}" > "${socket_out}"
  keep_resident=1
  exit 0
fi

BENCH_WORKER_RESIDENT_SOCKET="${socket}" DS4_RESIDENT_SOCKET="${socket}" "$@"

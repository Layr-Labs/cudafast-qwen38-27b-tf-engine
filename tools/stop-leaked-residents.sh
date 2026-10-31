#!/usr/bin/env bash
#
# stop-leaked-residents.sh -- remove exact resident containers that a cancelled
# job could not stop. Preflight also waits until no compute process uses the GPU.
# Exit 3 means that cleanup was deferred to the next job's preflight.
set -euo pipefail

usage() {
  echo "Usage: tools/stop-leaked-residents.sh --mode preflight|cleanup [--lock PATH] [--gpu-wait SECONDS] [--lock-wait SECONDS]" >&2
}

error() {
  echo "::error::$*" >&2
}

invalid() {
  error "$*"
  usage
  exit 2
}

mode=""
lock_path="/tmp/mtplx-gpu-exclusive.lock"
gpu_wait="60"
lock_wait="0"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ "$#" -ge 2 ]] || invalid "--mode needs a value"
      mode="$2"
      shift 2
      ;;
    --lock)
      [[ "$#" -ge 2 && -n "$2" ]] || invalid "--lock needs a path"
      lock_path="$2"
      shift 2
      ;;
    --gpu-wait)
      [[ "$#" -ge 2 ]] || invalid "--gpu-wait needs a value"
      gpu_wait="$2"
      shift 2
      ;;
    --lock-wait)
      [[ "$#" -ge 2 ]] || invalid "--lock-wait needs a value"
      lock_wait="$2"
      shift 2
      ;;
    *)
      invalid "unknown argument: $1"
      ;;
  esac
done

[[ "${mode}" == "preflight" || "${mode}" == "cleanup" ]] \
  || invalid "--mode must be preflight or cleanup"
[[ "${gpu_wait}" =~ ^[0-9]+$ ]] || invalid "--gpu-wait must be a non-negative integer"
[[ "${lock_wait}" =~ ^[0-9]+$ ]] || invalid "--lock-wait must be a non-negative integer"
gpu_wait=$((10#${gpu_wait}))
lock_wait=$((10#${lock_wait}))

prefix="${SERVE_UP_CONTAINER_PREFIX:-serve-up-}"
[[ "${prefix}" =~ ^[a-z0-9][a-z0-9-]*-$ ]] || {
  error "SERVE_UP_CONTAINER_PREFIX is invalid; refusing to stop any container"
  exit 2
}

command -v timeout >/dev/null 2>&1 || { error "timeout is required to bound Docker calls"; exit 1; }

# Tests can lower this bound. Production leaves the test-only variable unset.
docker_timeout="${STOP_LEAKED_RESIDENTS_TEST_DOCKER_TIMEOUT:-30}"
[[ "${docker_timeout}" =~ ^([1-9]|[12][0-9]|30)$ ]] \
  || invalid "STOP_LEAKED_RESIDENTS_TEST_DOCKER_TIMEOUT must be an integer from 1 to 30"
docker_timeout=$((10#${docker_timeout}))
stop_grace=10
stop_timeout=$((stop_grace + docker_timeout))

docker_call() {
  local call_name="$1" call_timeout="$2"
  shift 2
  if DOCKER_OUTPUT="$(timeout --signal=KILL "${call_timeout}" "$@" 2>&1)"; then
    DOCKER_RC=0
    return 0
  else
    DOCKER_RC=$?
  fi
  DOCKER_CALL_NAME="${call_name}"
  DOCKER_CALL_TIMEOUT="${call_timeout}"
  return 1
}

report_docker_failure() {
  if [[ "${DOCKER_RC}" -eq 124 || "${DOCKER_RC}" -eq 137 ]]; then
    error "${DOCKER_CALL_NAME} timed out after ${DOCKER_CALL_TIMEOUT} seconds"
  else
    error "${DOCKER_CALL_NAME} failed: ${DOCKER_OUTPUT//$'\n'/; }"
  fi
}

list_containers() {
  if ! docker_call "docker ps" "${docker_timeout}" \
      docker ps -a --format '{{.Names}}'; then
    report_docker_failure
    return 1
  fi
  printf '%s\n' "${DOCKER_OUTPUT}"
}

find_matches() {
  local name
  while IFS= read -r name; do
    if [[ "${name}" =~ ^${prefix}[0-9]+-[0-9]{8}T[0-9]{6}Z$ ]]; then
      printf '%s\n' "${name}"
    fi
  done
}

load_matches() {
  local containers name
  containers="$(list_containers)" || return 1
  matches=()
  while IFS= read -r name; do
    [[ -n "${name}" ]] && matches+=("${name}")
  done < <(find_matches <<< "${containers}")
}

container_exists() {
  local name="$1" listed
  if ! docker_call "docker ps existence check for ${name}" "${docker_timeout}" \
      docker ps -a --filter "name=^/${name}$" --format '{{.Names}}'; then
    report_docker_failure
    return 2
  fi
  while IFS= read -r listed; do
    [[ "${listed}" == "${name}" ]] && return 0
  done <<< "${DOCKER_OUTPUT}"
  return 1
}

if [[ "${mode}" == "cleanup" ]]; then
  load_matches || exit 1
  [[ "${#matches[@]}" -gt 0 ]] || exit 0
fi

command -v flock >/dev/null 2>&1 || { error "flock is required for the GPU lock"; exit 1; }
if ! exec 9< "${lock_path}"; then
  error "cannot open the GPU lock for reading: ${lock_path}"
  exit 1
fi
if [[ "${lock_wait}" -eq 0 ]]; then
  flock_args=(-n)
else
  flock_args=(-w "${lock_wait}")
fi
if flock "${flock_args[@]}" 9; then
  :
else
  lock_rc=$?
  if [[ "${mode}" == "cleanup" && "${lock_rc}" -eq 1 ]]; then
    echo "::warning::the GPU lock is still held; not stopping the matching resident container; the next job's preflight stops any leftover" >&2
    exit 3
  fi
  error "another GPU window holds it: ${lock_path}"
  exit 1
fi

load_matches || exit 1
[[ "${mode}" != "cleanup" || "${#matches[@]}" -gt 0 ]] || exit 0

if [[ "${#matches[@]}" -gt 0 ]]; then
  for name in "${matches[@]}"; do
    if [[ "${mode}" == "preflight" ]]; then
      echo "::warning::stopping resident container left by an earlier job: ${name}" >&2
    else
      echo "notice: stopping resident container during cleanup: ${name}" >&2
    fi
    if ! docker_call "docker stop for ${name}" "${stop_timeout}" \
        docker stop -t "${stop_grace}" "${name}"; then
      stop_rc="${DOCKER_RC}"
      stop_output="${DOCKER_OUTPUT}"
      if [[ "${stop_rc}" -eq 124 || "${stop_rc}" -eq 137 ]]; then
        report_docker_failure
        exit 1
      fi
      if container_exists "${name}"; then
        error "docker stop failed for ${name}: ${stop_output//$'\n'/; }"
        exit 1
      else
        exists_rc=$?
        [[ "${exists_rc}" -eq 1 ]] && continue
        exit 1
      fi
    fi
    if ! docker_call "docker rm for ${name}" "${docker_timeout}" \
        docker rm -f "${name}"; then
      rm_rc="${DOCKER_RC}"
      rm_output="${DOCKER_OUTPUT}"
      if [[ "${rm_rc}" -eq 124 || "${rm_rc}" -eq 137 ]]; then
        report_docker_failure
        exit 1
      fi
      if container_exists "${name}"; then
        error "docker rm failed for ${name}: ${rm_output//$'\n'/; }"
        exit 1
      else
        exists_rc=$?
        [[ "${exists_rc}" -eq 1 ]] || exit 1
      fi
    fi
  done
fi

container_deadline=$((SECONDS + gpu_wait))
while :; do
  containers="$(list_containers)" || exit 1
  remaining_matches=()
  while IFS= read -r name; do
    [[ -n "${name}" ]] && remaining_matches+=("${name}")
  done < <(find_matches <<< "${containers}")
  [[ "${#remaining_matches[@]}" -eq 0 ]] && break
  if [[ "${SECONDS}" -ge "${container_deadline}" ]]; then
    error "resident containers remain after cleanup: ${remaining_matches[*]}"
    exit 1
  fi
  sleep 1
done

command -v nvidia-smi >/dev/null 2>&1 || { error "nvidia-smi is required to check GPU compute processes"; exit 1; }
deadline=$((SECONDS + gpu_wait))
while :; do
  remaining=$((deadline - SECONDS))
  [[ "${remaining}" -gt 0 ]] || remaining=1
  if gpu_apps="$(timeout --signal=KILL "${remaining}" \
      nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1)"; then
    :
  else
    query_rc=$?
    if [[ "${query_rc}" -eq 124 || "${query_rc}" -eq 137 ]]; then
      error "nvidia-smi query timed out after ${remaining} seconds"
      exit 1
    fi
    error "nvidia-smi failed: ${gpu_apps//$'\n'/; }"
    exit 1
  fi
  foreign="$(printf '%s\n' "${gpu_apps}" | grep -vE '^\s*$' | grep -vE 'Xorg|gnome-shell' || true)"
  [[ -z "${foreign}" ]] && break
  if [[ "${SECONDS}" -ge "${deadline}" ]]; then
    error "GPU compute processes remain after ${gpu_wait} seconds: ${foreign//$'\n'/; }"
    exit 1
  fi
  delay=$((deadline - SECONDS))
  [[ "${delay}" -gt 2 ]] && delay=2
  sleep "${delay}"
done

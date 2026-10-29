#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
HELPER="${SCRIPT_DIR}/wait-for-memory.sh"
TEST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wait-for-memory.XXXXXX")" || {
  printf 'test-wait-for-memory: FAIL -- cannot make the test directory\n' >&2
  exit 1
}
trap 'rm -rf -- "${TEST_DIR}"' EXIT

failures=""
record_failure() {
  if [[ -n "${failures}" ]]; then
    failures="${failures}; $*"
  else
    failures="$*"
  fi
}

if [[ ! -x "${HELPER}" ]]; then
  printf 'test-wait-for-memory: FAIL -- missing executable %s\n' "${HELPER}" >&2
  exit 1
fi

WEIGHTS="${TEST_DIR}/weights"
ZERO_WEIGHTS="${TEST_DIR}/zero-weights"
MEMINFO="${TEST_DIR}/meminfo"
mkdir -p "${WEIGHTS}" "${ZERO_WEIGHTS}" || record_failure "cannot make weights directories"
printf 'a' >"${WEIGHTS}/model-00001.safetensors" || record_failure "cannot write shard 1"
printf 'bc' >"${WEIGHTS}/model-00002.safetensors" || record_failure "cannot write shard 2"
: >"${ZERO_WEIGHTS}/model.safetensors" || record_failure "cannot write the zero-byte shard"

GIB=$((1024 * 1024 * 1024))
WEIGHT_BYTES=3
CACHE_BYTES=$((8 * GIB))
MARGIN_BYTES=$((2 * GIB))
TOTAL_KB=$((100 * 1024 * 1024))
# TensorFold unified-memory reserve: max(4 GiB, MemTotal / 10) (cuda_memory.zig reserveBytes).
RESERVE_BYTES=$((TOTAL_KB * 1024 / 10))
REQUIRED_BYTES=$((WEIGHT_BYTES + CACHE_BYTES + RESERVE_BYTES + MARGIN_BYTES))
THRESHOLD_KB=$(((REQUIRED_BYTES + 1023) / 1024))

write_meminfo() {
  printf 'MemTotal: %s kB\nMemAvailable: %s kB\n' \
    "${TOTAL_KB}" "$1" >"${MEMINFO}"
}

write_meminfo "${THRESHOLD_KB}"
if env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" MLXFAST_TARGET_SNAPSHOT_DIR= \
    SERVE_UP_MEMINFO="${MEMINFO}" WAIT_FOR_MEMORY_TIMEOUT_S=0 \
    WAIT_FOR_MEMORY_POLL_S=0.01 "${HELPER}" >"${TEST_DIR}/enough.log" 2>&1; then
  grep -Fq "weights=${WEIGHT_BYTES} bytes + cache=${CACHE_BYTES} bytes (8 GiB) + engine reserve=${RESERVE_BYTES} bytes + margin=${MARGIN_BYTES} bytes (2 GiB) = required=${REQUIRED_BYTES} bytes" \
    "${TEST_DIR}/enough.log" || record_failure "enough memory did not log the byte formula"
  grep -Fq 'ready after' "${TEST_DIR}/enough.log" \
    || record_failure "enough memory did not pass"
else
  record_failure "enough memory failed"
fi

write_meminfo "$((THRESHOLD_KB - 1))"
if env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" MLXFAST_TARGET_SNAPSHOT_DIR= \
    SERVE_UP_MEMINFO="${MEMINFO}" WAIT_FOR_MEMORY_TIMEOUT_S=0 \
    WAIT_FOR_MEMORY_POLL_S=0.01 "${HELPER}" >"${TEST_DIR}/little.log" 2>&1; then
  record_failure "too little memory passed"
elif ! grep -Fq 'timed out after' "${TEST_DIR}/little.log"; then
  record_failure "too little memory did not fail by timeout"
fi

if env SERVE_UP_WEIGHTS_DIR= MLXFAST_TARGET_SNAPSHOT_DIR= \
    SERVE_UP_MEMINFO="${MEMINFO}" WAIT_FOR_MEMORY_TIMEOUT_S=0 \
    WAIT_FOR_MEMORY_POLL_S=0.01 "${HELPER}" >"${TEST_DIR}/unset.log" 2>&1; then
  record_failure "an unset weights directory passed"
elif ! grep -Fq 'SERVE_UP_WEIGHTS_DIR/MLXFAST_TARGET_SNAPSHOT_DIR is unset' \
    "${TEST_DIR}/unset.log"; then
  record_failure "an unset weights directory did not fail closed"
fi

if env SERVE_UP_WEIGHTS_DIR="${TEST_DIR}/missing" MLXFAST_TARGET_SNAPSHOT_DIR= \
    SERVE_UP_MEMINFO="${MEMINFO}" WAIT_FOR_MEMORY_TIMEOUT_S=0 \
    WAIT_FOR_MEMORY_POLL_S=0.01 "${HELPER}" >"${TEST_DIR}/missing.log" 2>&1; then
  record_failure "a missing weights directory passed"
elif ! grep -Fq 'target weights directory is missing' "${TEST_DIR}/missing.log"; then
  record_failure "a missing weights directory did not fail closed"
fi

if env SERVE_UP_WEIGHTS_DIR="${ZERO_WEIGHTS}" MLXFAST_TARGET_SNAPSHOT_DIR= \
    SERVE_UP_MEMINFO="${MEMINFO}" WAIT_FOR_MEMORY_TIMEOUT_S=0 \
    WAIT_FOR_MEMORY_POLL_S=0.01 "${HELPER}" >"${TEST_DIR}/zero.log" 2>&1; then
  record_failure "zero-byte weights passed"
elif ! grep -Fq 'target weights total is 0 bytes' "${TEST_DIR}/zero.log"; then
  record_failure "zero-byte weights did not fail closed"
fi

start_ns="$(python3 -c 'import time; print(time.monotonic_ns())')"
if env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" MLXFAST_TARGET_SNAPSHOT_DIR= \
    SERVE_UP_MEMINFO="${MEMINFO}" WAIT_FOR_MEMORY_TIMEOUT_S=0.15 \
    WAIT_FOR_MEMORY_POLL_S=0.02 "${HELPER}" >"${TEST_DIR}/timeout.log" 2>&1; then
  timeout_status=0
else
  timeout_status=$?
fi
end_ns="$(python3 -c 'import time; print(time.monotonic_ns())')"
elapsed_ms=$(((end_ns - start_ns) / 1000000))
if [[ "${timeout_status}" -ne 1 ]]; then
  record_failure "the bounded timeout exited ${timeout_status}"
elif ! grep -Fq 'timed out after' "${TEST_DIR}/timeout.log"; then
  record_failure "the bounded timeout was not reported"
elif ! grep -Fq 'waiting:' "${TEST_DIR}/timeout.log"; then
  record_failure "the bounded timeout did not poll"
elif (( elapsed_ms < 100 )); then
  record_failure "the 0.15 second timeout returned after ${elapsed_ms} ms"
fi

# Reserve floor: a 20 GiB box reserves the 4 GiB floor, not 2 GiB (10 %).
SMALL_TOTAL_KB=$((20 * 1024 * 1024))
SMALL_REQUIRED=$((WEIGHT_BYTES + CACHE_BYTES + 4 * GIB + MARGIN_BYTES))
SMALL_THRESHOLD_KB=$(((SMALL_REQUIRED + 1023) / 1024))
printf 'MemTotal: %s kB\nMemAvailable: %s kB\n' "${SMALL_TOTAL_KB}" "$((SMALL_THRESHOLD_KB - 1))" >"${MEMINFO}"
if env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" MLXFAST_TARGET_SNAPSHOT_DIR= SERVE_UP_MEMINFO="${MEMINFO}" \
    WAIT_FOR_MEMORY_TIMEOUT_S=0 WAIT_FOR_MEMORY_POLL_S=0.01 "${HELPER}" >"${TEST_DIR}/floor.log" 2>&1; then
  record_failure "reserve floor: one KiB under weights+cache+4 GiB+margin passed"
fi
printf 'MemTotal: %s kB\nMemAvailable: %s kB\n' "${SMALL_TOTAL_KB}" "${SMALL_THRESHOLD_KB}" >"${MEMINFO}"
env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" MLXFAST_TARGET_SNAPSHOT_DIR= SERVE_UP_MEMINFO="${MEMINFO}" \
  WAIT_FOR_MEMORY_TIMEOUT_S=0 WAIT_FOR_MEMORY_POLL_S=0.01 "${HELPER}" >"${TEST_DIR}/floor-ok.log" 2>&1 \
  || record_failure "reserve floor: exactly weights+cache+4 GiB+margin failed"

if [[ -n "${failures}" ]]; then
  printf 'test-wait-for-memory: FAIL -- %s\n' "${failures}" >&2
  exit 1
fi
printf 'test-wait-for-memory: PASS\n'

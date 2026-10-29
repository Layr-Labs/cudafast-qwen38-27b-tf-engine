#!/usr/bin/env bash
# test-serve-up-plan-memory.sh -- prove the memory plan in tools/serve-up.sh.
#
# THE PLAN. GB10 memory is unified. TensorFold loads the target weights and one
# admitted cache. serve-up.sh refuses BEFORE the load unless
#   (a) the weights plus the minimum cache are free right now, and
#   (b) that load plan fits inside serve-up.sh's local reservation ceiling.
#
# WHAT THIS PROVES
#   1. the free-memory boundary, to the kilobyte: the plan ACCEPTS exactly the
#      required amount and REFUSES one kilobyte below it
#   2. weights that do not fit the reservation REFUSE by name
#   3. a meminfo with no MemAvailable REFUSES by name (never a blind load)
#   4. the refusal has NO environment bypass
#   5. the plan prints its table before deciding
#
# Hermetic: the real serve-up.sh, the real resident on its fake backend via the
# host launcher, a synthetic snapshot and a synthetic meminfo.
#
# Usage: tools/test-serve-up-plan-memory.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
SERVE_UP="${ROOT_DIR}/tools/serve-up.sh"
# shellcheck source=tools/testlib/synthetic-window.sh
. "${ROOT_DIR}/tools/testlib/synthetic-window.sh"
synthetic_window_env

# serve-up.sh's minimum cache, restated on purpose.
MIN_CACHE_GIB=8

WORK="$(mktemp -d)"
SOCKDIR="$(mktemp -d /tmp/planmem.XXXXXX)"
trap 'rm -rf "${WORK}" "${SOCKDIR}"' EXIT

fails=0
pass() { printf 'test-serve-up-plan-memory: PASS -- %s\n' "$*"; }
fail() { printf 'test-serve-up-plan-memory: FAIL -- %s\n' "$*" >&2; fails=$((fails + 1)); }

WEIGHTS="${WORK}/w"
synthetic_window_weights "${WEIGHTS}" || { echo "cannot write the synthetic snapshot" >&2; exit 1; }
TOTAL_KB=125000000

cat > "${WORK}/worker" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == runtime-worker ]]; then
  printf '%s\n' '{"id":0,"ok":true,"protocol_version":1,"backend":"cuda","device":"mock","resident":{"pid":123,"load_epoch":1}}'
  exit 0
fi
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --socket ]]; then socket="$2"; shift 2; else shift; fi
done
touch "${socket}"
exec sleep 60
EOF
chmod +x "${WORK}/worker"
export TENSORFOLD_BENCH_WORKER="${WORK}/worker" SERVE_UP_TEST_READY_FILE=1

# The required free bytes, computed the way serve-up.sh computes them.
required_kb="$(python3 - "${WEIGHTS}" "${MIN_CACHE_GIB}" <<'PY'
import sys
from pathlib import Path
body = sum(p.stat().st_size for p in Path(sys.argv[1]).glob("*.safetensors"))
want = body + int(sys.argv[2]) * 1024**3
print(-(-want // 1024))  # ceil to whole kB
PY
)"

boot() { # boot MEMINFO [env...] -- --boot a serial leg; prints output, returns rc
  local meminfo="$1"; shift
  env "$@" SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_LOG_DIR="${WORK}/log" \
      SERVE_UP_SOCKET_DIR="${SOCKDIR}" SERVE_UP_MEMINFO="${meminfo}" \
      SERVE_UP_HEALTH_TIMEOUT_S=60 \
      "${SERVE_UP}" --boot --spec serial --draft-len 0 --socket-out "${WORK}/sock.txt" 2>&1
}
stop_booted() {
  [[ -s "${WORK}/sock.txt" ]] && "${SERVE_UP}" --stop --socket "$(head -1 "${WORK}/sock.txt")" >/dev/null 2>&1
  rm -f "${WORK}/sock.txt"
}

# --- 1. the boundary --------------------------------------------------------
synthetic_window_meminfo "${WORK}/exact" "${TOTAL_KB}" "${required_kb}"
out="$(boot "${WORK}/exact")"; rc=$?
if [[ ${rc} -eq 0 ]] && grep -q 'memory plan ACCEPTS' <<<"${out}"; then
  pass "1a: exactly the required free memory is accepted"
else
  fail "1a: the exact boundary was refused (rc=${rc}): $(tail -1 <<<"${out}")"
fi
stop_booted
synthetic_window_meminfo "${WORK}/short" "${TOTAL_KB}" "$(( required_kb - 1 ))"
out="$(boot "${WORK}/short")"; rc=$?
if [[ ${rc} -ne 0 ]] && grep -q 'available memory is too low' <<<"${out}"; then
  pass "1b: one kilobyte short is refused by name"
else
  fail "1b: one kilobyte short was not refused (rc=${rc})"
fi
stop_booted

# --- 2. weights that do not fit the reservation -----------------------------
# A tiny box: the reservation is smaller than a sparse 8 GiB shard plus cache.
BIG="${WORK}/big"
synthetic_window_weights "${BIG}"
truncate -s 8G "${BIG}/model-00001-of-00001.safetensors"
synthetic_window_meminfo "${WORK}/tiny" 16000000 16000000
out="$(env SERVE_UP_WEIGHTS_DIR="${BIG}" SERVE_UP_LOG_DIR="${WORK}/log" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      SERVE_UP_MEMINFO="${WORK}/tiny" "${SERVE_UP}" --boot --spec serial --draft-len 0 \
      --socket-out "${WORK}/sock.txt" 2>&1)"; rc=$?
if [[ ${rc} -ne 0 ]] && grep -q 'exceed the reservation' <<<"${out}"; then
  pass "2: weights beyond the reservation are refused by name"
else
  fail "2: oversized weights were not refused (rc=${rc}): $(tail -1 <<<"${out}")"
fi
stop_booted

# --- 3. no MemAvailable -----------------------------------------------------
printf 'MemTotal: %s kB\n' "${TOTAL_KB}" > "${WORK}/nomem"
out="$(boot "${WORK}/nomem")"; rc=$?
if [[ ${rc} -ne 0 ]] && grep -q 'refusing to start a load blind' <<<"${out}"; then
  pass "3: a meminfo with no MemAvailable is refused by name"
else
  fail "3: a blind load was not refused (rc=${rc})"
fi
stop_booted

# --- 4. no environment bypass ------------------------------------------------
out="$(boot "${WORK}/short" SERVE_UP_MEM_HEADROOM_GB=0 SERVE_UP_SKIP_MEMORY_PLAN=1 RESIDENT_GPU_MEMORY_UTILIZATION=0.01)"; rc=$?
if [[ ${rc} -ne 0 ]] && grep -q 'memory plan REFUSES' <<<"${out}"; then
  pass "4: no environment variable relaxes the plan"
else
  fail "4: an environment variable relaxed the plan (rc=${rc})"
fi
stop_booted

# --- 5. the table is printed before the decision -----------------------------
out="$(boot "${WORK}/short")"
if grep -q 'memory plan for this leg' <<<"${out}" && grep -q 'reserve=' <<<"${out}"; then
  pass "5: the plan table is printed with the refusal"
else
  fail "5: the plan table is missing"
fi
stop_booted

if [[ ${fails} -eq 0 ]]; then
  echo "test-serve-up-plan-memory: all cases passed"
  exit 0
fi
echo "test-serve-up-plan-memory: ${fails} case(s) failed" >&2
exit 1

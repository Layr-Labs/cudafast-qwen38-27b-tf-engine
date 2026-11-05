#!/usr/bin/env bash
# Check the ranked workflow's keyring guard proof step against planted proof files:
# it must print one line per adapter start and a summary, warn when a long or
# unknown-length start has no guard proof, and never exit non-zero because a
# start has no .time file (benchd kills a worker it is done with, so the
# wrapper cannot always write one).
set -uo pipefail

TEST_NAME='test-keyring-guard-proof.sh'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)" || {
  printf '%s: FAIL: mktemp failed\n' "${TEST_NAME}"
  exit 1
}
trap 'rm -rf "${WORK}"' EXIT

fail() {
  printf '%s: FAIL: %s\n' "${TEST_NAME}" "$1"
  exit 1
}

awk '
  /^      - name: Keyring guard proof/ { step=1; next }
  step && /^        run: \|/ { copy=1; next }
  copy && /^      - name:/ { exit }
  copy { sub(/^          /, ""); print }
' "${ROOT}/.github/workflows/benchmark.yml" > "${WORK}/step"
grep -Fq 'set -euo pipefail' "${WORK}/step" || fail 'could not extract the keyring guard proof step'

GUARD='Seccomp=2 NoNewPrivs=1'
start() { # start CASE NAME [guard] [ms]
  : > "${WORK}/$1/engine-stderr.$2"
  if [[ "${3:-}" == guard ]]; then echo "${GUARD}" > "${WORK}/$1/engine-stderr.$2.guard"; fi
  if [[ -n "${4:-}" ]]; then echo "ms=$4" > "${WORK}/$1/engine-stderr.$2.time"; fi
}
run_case() { # run_case CASE -> sets out, rc
  out="$(RUNNER_TEMP="${WORK}/$1" bash "${WORK}/step" 2>&1)"
  rc=$?
}

# 1. A guarded start with no .time file (killed before the wrapper wrote it):
#    reported as an unknown-length start with proof, no warning, exit 0.
mkdir "${WORK}/c1"; start c1 AAAAAA guard
run_case c1
[[ "${rc}" -eq 0 ]] || fail "a start with no .time file made the step exit ${rc}: ${out}"
grep -Fq 'engine-stderr.AAAAAA: proof yes, ran ? ms' <<<"${out}" || fail "no per-start line for the start with no .time: ${out}"
grep -Fq '1 of 1 adapter starts ran with Seccomp=2 NoNewPrivs=1; long or unknown-length starts: 1 of 1' <<<"${out}" \
  || fail "wrong summary for case 1: ${out}"
grep -Fq '::warning' <<<"${out}" && fail "case 1 warned although every start has proof: ${out}"

# 2. An unguarded start with no .time file: counted as long, so it warns.
mkdir "${WORK}/c2"; start c2 AAAAAA
run_case c2
[[ "${rc}" -eq 0 ]] || fail "case 2 exited ${rc}: ${out}"
grep -Fq '::warning title=Keyring guard::1 of 1 long or unknown-length adapter starts have no guard proof' <<<"${out}" \
  || fail "an unguarded start with no .time did not warn: ${out}"

# 3. A short unguarded self-test start and a long guarded start: no warning.
mkdir "${WORK}/c3"; start c3 AAAAAA '' 3; start c3 BBBBBB guard 9000
run_case c3
[[ "${rc}" -eq 0 ]] || fail "case 3 exited ${rc}: ${out}"
grep -Fq '1 of 2 adapter starts ran with Seccomp=2 NoNewPrivs=1; long or unknown-length starts: 1 of 1' <<<"${out}" \
  || fail "wrong summary for case 3: ${out}"
grep -Fq '::warning' <<<"${out}" && fail "case 3 warned for a short self-test start: ${out}"

# 4. A long unguarded start with a .time file: warns, exit 0.
mkdir "${WORK}/c4"; start c4 AAAAAA '' 5000
run_case c4
[[ "${rc}" -eq 0 ]] || fail "case 4 exited ${rc}: ${out}"
grep -Fq '1 of 1 long or unknown-length adapter starts have no guard proof' <<<"${out}" || fail "case 4 did not warn: ${out}"

# 5. No adapter start at all: summary only, exit 0.
mkdir "${WORK}/c5"
run_case c5
[[ "${rc}" -eq 0 ]] || fail "case 5 exited ${rc}: ${out}"
grep -Fq '0 of 0 adapter starts' <<<"${out}" || fail "no summary for case 5: ${out}"

printf '%s: PASS\n' "${TEST_NAME}"

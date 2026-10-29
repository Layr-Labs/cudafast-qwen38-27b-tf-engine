#!/usr/bin/env bash
# Check the ranked workflow's fixed GPU lock contract.
set -uo pipefail

TEST_NAME='test-benchmark-gpu-lock.sh'
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
  /^[[:space:]]*# -- box-wide GPU lock/ { copy=1; next }
  /# -- host quiescence/ { copy=0 }
  copy { print }
' "${ROOT}/.github/workflows/benchmark.yml" \
  | sed \
      -e "s#lock=/tmp/mtplx-gpu-exclusive.lock#lock='${WORK}/gpu.lock'#" \
      -e '/^[[:space:]]*echo "waiting for the box GPU lock/,$d' \
  > "${WORK}/check"
printf '%s\n' 'exit 0' >> "${WORK}/check"

grep -Fq "lock_error='::error::GPU lock contract failed'" "${WORK}/check" \
  || fail 'the workflow has no fixed lock error class'
grep -Fq 'sudo -n -u "${SANDBOX_USER:-qwen38sub}" -- test ! -w "${lock}"' "${WORK}/check" \
  || fail 'the workflow does not check write access as the sandbox user'
grep -Fq 'exec 9< "${lock}"' "${WORK}/check" \
  || fail 'the workflow does not open the lock without create access'
if grep -Fq 'exec 9<> "${lock}"' "${WORK}/check"; then
  fail 'the workflow can create the lock while opening it'
fi

mkdir "${WORK}/bin"
cat > "${WORK}/bin/stat" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "${LOCK_IDENTITY}"
EOF
cat > "${WORK}/bin/sudo" <<'EOF'
#!/usr/bin/env bash
[[ "$1 $2" == '-n -u' && "$4" == '--' && "$5 $6 $7" == 'test ! -w' ]] || exit 97
[[ "${LOCK_WRITABLE:-0}" != 1 ]]
EOF
chmod +x "${WORK}/bin/stat" "${WORK}/bin/sudo"

run_check() {
  env PATH="${WORK}/bin:${PATH}" LOCK_IDENTITY="$1" LOCK_WRITABLE="${2:-0}" \
    SANDBOX_USER=qwen38sub /bin/bash "${WORK}/check" > "${WORK}/out" 2>&1
}

expect_refusal() {
  local label="$1" identity="$2" writable="${3:-0}" status
  run_check "${identity}" "${writable}"
  status=$?
  [[ "${status}" -eq 1 ]] \
    || fail "${label} returned ${status}"
  [[ "$(grep -c '^::error::GPU lock contract failed$' "${WORK}/out")" -eq 1 ]] \
    || fail "${label} did not print the fixed error class once"
}

rm -f "${WORK}/gpu.lock"
expect_refusal 'missing lock' 'root:bench 660'

: > "${WORK}/gpu.lock"
expect_refusal 'mode 0666' 'root:bench 666'
expect_refusal 'wrong owner' 'runner:bench 660'
expect_refusal 'wrong group' 'root:runner 660'
expect_refusal 'sandbox-writable lock' 'root:bench 660' 1

rm -f "${WORK}/gpu.lock"
: > "${WORK}/target"
ln -s "${WORK}/target" "${WORK}/gpu.lock"
expect_refusal 'symlink lock' 'root:bench 660'

rm -f "${WORK}/gpu.lock"
mkfifo "${WORK}/gpu.lock"
expect_refusal 'non-regular lock' 'root:bench 660'

rm -f "${WORK}/gpu.lock"
: > "${WORK}/gpu.lock"
run_check 'root:bench 660' 0
status=$?
[[ "${status}" -eq 0 && ! -s "${WORK}/out" ]] \
  || fail "root:bench 0660 lock returned ${status}"

printf '%s: PASS\n' "${TEST_NAME}"

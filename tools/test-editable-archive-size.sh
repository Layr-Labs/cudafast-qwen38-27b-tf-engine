#!/usr/bin/env bash
# Tests for tools/check-editable-archive-size.sh: a small surface passes; an
# oversized UNTRACKED file inside the surface fails (yukon submit archives it too);
# a missing editable path fails; a second editable path is counted.
set -euo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/editable-archive-test.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT
ROOT="${WORK}/repo"
mkdir -p "${ROOT}/tools" "${ROOT}/surface" "${ROOT}/second"
cp "${SOURCE_ROOT}/tools/check-editable-archive-size.sh" "${ROOT}/tools/"
fail() { echo "test-editable-archive-size: FAIL: $1" >&2; exit 1; }
run() { (cd "${ROOT}" && tools/check-editable-archive-size.sh) >"${WORK}/out" 2>&1; }

printf '%s\n' '{"editablePaths":["surface"]}' > "${ROOT}/benchmark.json"
printf 'small\n' > "${ROOT}/surface/small.txt"
run || fail "a small surface was refused: $(cat "${WORK}/out")"
grep -q 'check-editable-archive-size: PASS' "${WORK}/out" || fail "no PASS line"

dd if=/dev/urandom of="${ROOT}/surface/large.bin" bs=1048576 count=21 2>/dev/null
if run; then fail "an oversized untracked file passed"; fi
grep -q 'archive exceeds 20971520 bytes' "${WORK}/out" || fail "wrong refusal: $(cat "${WORK}/out")"
rm -f "${ROOT}/surface/large.bin"

printf '%s\n' '{"editablePaths":["surface","missing"]}' > "${ROOT}/benchmark.json"
if run; then fail "a missing editable path passed"; fi
grep -q "editable path 'missing' does not exist" "${WORK}/out" || fail "wrong refusal: $(cat "${WORK}/out")"

printf '%s\n' '{"editablePaths":["surface","second"]}' > "${ROOT}/benchmark.json"
dd if=/dev/urandom of="${ROOT}/second/large.bin" bs=1048576 count=21 2>/dev/null
if run; then fail "an oversized file in the second editable path passed"; fi

echo "test-editable-archive-size: PASS"

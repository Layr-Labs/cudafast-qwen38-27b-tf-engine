#!/usr/bin/env bash
# Check the local, pinned baseline staging plan.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAGER="${ROOT}/tools/stage-baseline-workspace.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
PIN=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
FIXTURE="${WORK}/fixture.json"
SOURCE="${WORK}/reference.bundle"
DEST="${WORK}/reference"
printf '{"baseline_reference_commit":"%s"}\n' "${PIN}" > "${FIXTURE}"
: > "${SOURCE}"

out="$("${STAGER}" "${DEST}" --source "${SOURCE}" --fixture "${FIXTURE}" --dry-run)"
grep -Fq "git clone --no-local --no-checkout '${SOURCE}' '${DEST}'" <<<"${out}"
grep -Fq "checkout --detach ${PIN}" <<<"${out}"
grep -Fq "${DEST}/tools/engine/build.sh" <<<"${out}"
grep -Fq "${DEST}/tools/stage-cuda-engine.sh" <<<"${out}"
grep -Fq "git -C '${DEST}' rev-parse HEAD" <<<"${out}"
[[ ! -e "${DEST}" ]]

if env -u TENSORFOLD_BASELINE_SOURCE "${STAGER}" "${DEST}" --fixture "${FIXTURE}" --dry-run >"${WORK}/missing" 2>&1; then
  echo "FAIL: a missing local source was accepted" >&2
  exit 1
fi
grep -Fq 'REFUSE missing-source' "${WORK}/missing"
grep -Fq 'holds no credential' "${WORK}/missing"

if "${STAGER}" "${DEST}" --source "${WORK}/absent.bundle" --fixture "${FIXTURE}" --dry-run >"${WORK}/absent" 2>&1; then
  echo "FAIL: an absent local source was accepted" >&2
  exit 1
fi
grep -Fq 'REFUSE missing-source' "${WORK}/absent"

mkdir "${DEST}"
if "${STAGER}" "${DEST}" --source "${SOURCE}" --fixture "${FIXTURE}" >"${WORK}/exists" 2>&1; then
  echo "FAIL: an existing destination was accepted" >&2
  exit 1
fi
grep -Fq 'REFUSE destination-exists' "${WORK}/exists"
rmdir "${DEST}"

printf '{}\n' > "${WORK}/no-reference.json"
if "${STAGER}" "${DEST}" --source "${SOURCE}" --fixture "${WORK}/no-reference.json" --dry-run >"${WORK}/no-reference" 2>&1; then
  echo "FAIL: a fixture without a reference commit was accepted" >&2
  exit 1
fi
grep -Fq 'REFUSE missing-reference-commit' "${WORK}/no-reference"

if "${STAGER}" "${DEST}" --nope >"${WORK}/unknown" 2>&1; then
  echo "FAIL: an unknown argument was accepted" >&2
  exit 1
fi
grep -Fq 'unknown argument --nope' "${WORK}/unknown"

echo "test-stage-baseline-workspace: PASS"

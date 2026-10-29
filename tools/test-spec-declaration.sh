#!/usr/bin/env bash
# Validate the DFlash declaration without a GPU or model weights.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DECL="${ROOT}/tools/spec-declaration.sh"
CONTRACT="${ROOT}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json"
TMP="$(mktemp -d)"
trap 'find "${TMP}" -depth -delete' EXIT
failures=0
case_id=0

manifest() {
  case_id=$((case_id + 1))
  local path="${TMP}/manifest-${case_id}.json"
  printf '%s\n' "$1" >"${path}"
  printf '%s\n' "${path}"
}

expect_ok() {
  local label="$1" json="$2" action="$3" want="$4" got rc
  got="$(SPEC_DECLARATION_MANIFEST="$(manifest "${json}")" \
    SPEC_DECLARATION_CONTRACT="${CONTRACT}" "${DECL}" "${action}" 2>/dev/null)"
  rc=$?
  if [[ "${rc}" -eq 0 && "${got}" == "${want}" ]]; then
    printf 'PASS %s\n' "${label}"
  else
    printf 'FAIL %s: rc=%s got=%q want=%q\n' "${label}" "${rc}" "${got}" "${want}"
    failures=$((failures + 1))
  fi
}

expect_refusal() {
  local label="$1" json="$2" needle="$3" out rc
  out="$(SPEC_DECLARATION_MANIFEST="$(manifest "${json}")" \
    SPEC_DECLARATION_CONTRACT="${CONTRACT}" "${DECL}" validate 2>&1)"
  rc=$?
  if [[ "${rc}" -ne 0 && "${out}" == *"${needle}"* ]]; then
    printf 'PASS %s\n' "${label}"
  else
    printf 'FAIL %s: rc=%s output=%q\n' "${label}" "${rc}" "${out}"
    failures=$((failures + 1))
  fi
}

[[ "$(jq -c '.permitted_speculative_depths' "${CONTRACT}")" == '[1]' ]] \
  || { printf 'FAIL contract depths\n'; failures=$((failures + 1)); }

serial='{"version":1,"source":"pinned","spec":{"enabled":false,"tokens":0}}'
dflash='{"version":1,"source":"pinned","spec":{"enabled":true,"tokens":1}}'
expect_ok serial-label "${serial}" describe serial
expect_ok serial-json "${serial}" json '{"mode":"serial"}'
expect_ok serial-flag "${serial}" speculative 0
expect_ok dflash-label "${dflash}" describe dflash1
expect_ok dflash-depth "${dflash}" draft-len 1
expect_ok dflash-flag "${dflash}" speculative 1
expect_ok dflash-json "${dflash}" json '{"mode":"dflash","dflash":{"depth":1}}'
expect_ok shipped-declaration "$(cat "${ROOT}/mtp-head.manifest.json")" describe dflash1
expect_ok max-bytes-not-read '{"source":"pinned","max_bytes":3000000000,"spec":{"enabled":false,"tokens":0}}' describe serial

expect_refusal non-object '[]' 'must be a JSON object'
expect_refusal missing-spec '{"version":1,"source":"pinned"}' 'has no spec block'
expect_refusal missing-enabled '{"spec":{"tokens":0}}' 'spec.enabled is required'
expect_refusal missing-tokens '{"spec":{"enabled":false}}' 'spec.tokens is required'
expect_refusal disabled-nonzero '{"spec":{"enabled":false,"tokens":1}}' 'requires spec.tokens=0'
expect_refusal depth-zero '{"spec":{"enabled":true,"tokens":0}}' 'not a contract-permitted DFlash depth'
expect_refusal depth-two '{"spec":{"enabled":true,"tokens":2}}' 'not a contract-permitted DFlash depth'
expect_refusal negative-depth '{"spec":{"enabled":true,"tokens":-1}}' 'outside the permitted range 0..8'
expect_refusal structural-ceiling '{"spec":{"enabled":true,"tokens":9}}' 'outside the permitted range 0..8'
expect_refusal non-integer '{"spec":{"enabled":true,"tokens":1.5}}' 'must be an integer'
expect_refusal string-tokens '{"spec":{"enabled":true,"tokens":"1"}}' 'must be an integer'
expect_refusal bad-enabled '{"spec":{"enabled":"yes","tokens":1}}' 'must be a boolean'
expect_refusal bad-spec '{"spec":[]}' 'spec" declaration must be an object'
expect_refusal bad-source '{"source":"remote","spec":{"enabled":false,"tokens":0}}' 'only accepted source'
expect_refusal in-branch-source '{"source":"in_branch","spec":{"enabled":false,"tokens":0}}' 'is not accepted'
expect_refusal unknown-key '{"spec":{"enabled":true,"tokens":1,"extra":1}}' 'unknown key'
expect_refusal unknown-top '{"arm":"dflash","spec":{"enabled":true,"tokens":1}}' 'unknown top-level key'

if [[ "${failures}" -eq 0 ]]; then
  printf 'test-spec-declaration: PASS\n'
  exit 0
fi
printf 'test-spec-declaration: FAIL (%s)\n' "${failures}"
exit 1

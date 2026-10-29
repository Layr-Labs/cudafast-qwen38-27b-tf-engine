#!/usr/bin/env bash
#
# spec-declaration.sh -- the SINGLE trusted source that reads the organizer's
# fixed speculative-decode DECLARATION and derives the serve-boot config
# from it. tools/serve-up.sh, .github/workflows/benchmark.yml and
# tools/ranked-box-preflight.sh ALL resolve the serve spec THROUGH this one
# script, so the fail-closed validation and the derivation can never drift
# between the arm gate and the serve that actually boots.
#
# NOT AN EDITABLE PATH. benchmark.json permits only Vendor/TensorFold. The
# declaration and this derivation are trusted control-plane files, so a
# submission cannot rewrite the declared value or relax the envelope.
#
# WHY THIS EXISTS. The candidate resident boots DFlash depth 1 with the pinned
# drafter; the serial resident is the control. Benchd must request the matching
# mode from each runtime worker. This script derives both values from the
# organizer-fixed declaration outside the editable surface. Entrants change
# only Vendor/TensorFold.
#
# THE DECLARATION is the organizer's serve knob, in mtp-head.manifest.json:
#
#   "spec": { "enabled": <bool>, "tokens": <int> }
#
# `enabled: false` with `tokens: 0` is serial. `enabled: true` with a permitted
# token depth is DFlash.
#
# THE REST OF THE DECLARATION is validated here too, because this is the one
# trusted reader of the file. `pinned` is the only accepted source and there
# is no `arm` key -- there is one arm, so there is nothing to select. Only the
# `spec`
# block was ever read, so `"source": "remote"` and an unknown top-level key both
# passed a `validate`. `max_bytes`, `bytes` and `sha256` are recorded, not read.
#
# VALIDATION is fail-closed. A REFUSAL (exit 1, message on stderr) for:
#   * a declaration that is not a JSON object, or carries an unknown top-level
#     key (allowed: version, source, max_bytes, bytes, sha256, spec);
#   * a `source` other than "pinned" -- "remote" and "in_branch" by name;
#   * a `spec` block that is not an object, or carries an unknown key;
#   * missing keys, `enabled` that is not a boolean, or `tokens` not an integer;
#   * a value outside the structural range 0..8 (the structural sanity ceiling);
#   * an ENABLED depth that is not in the contract's
#     permitted_speculative_depths. An out-of-envelope depth is REFUSED, never
#     clamped -- so the serve that boots and the `effective_spec` the adapter
#     seals can never disagree.
#
# Usage:
#   tools/spec-declaration.sh speculative   # 0 (serial) or 1 (DFlash)
#   tools/spec-declaration.sh draft-len     # tokens (0 = serial)
#   tools/spec-declaration.sh describe      # "serial" or "dflash<N>"
#   tools/spec-declaration.sh json          # Engine Protocol spec JSON
#   tools/spec-declaration.sh validate      # exit 0 if valid, else refuse
# Every subcommand validates the declaration first and refuses on an invalid one.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MANIFEST="${SPEC_DECLARATION_MANIFEST:-${REPO_ROOT}/mtp-head.manifest.json}"
CONTRACT="${SPEC_DECLARATION_CONTRACT:-${REPO_ROOT}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json}"

# The structural sanity ceiling for a declared draft length (the
# knob accepts a declared integer 0..8, not a hardcoded value). The contract's
# permitted_speculative_depths is the tighter, AUTHORITATIVE set enforced below when
# the value is enabled; this is the outer type/range guard around it.
SPEC_MAX_TOKENS=8


fail() {
  echo "spec-declaration.sh: REFUSING -- $*" >&2
  exit 1
}

command -v jq >/dev/null 2>&1 || fail "jq is required to read the declaration"

# --- read the declaration ---------------------------------------------------
# The declaration must exist and carry a complete spec block.
[[ -f "${MANIFEST}" ]] || fail "the declaration is missing: ${MANIFEST}"
enabled=""
raw_tokens=""
if [[ -f "${MANIFEST}" ]]; then
  jq -e . >/dev/null 2>&1 < "${MANIFEST}" || fail "mtp-head.manifest.json is not valid JSON"

  # --- the whole declaration, not just the spec block ------------------------
  # Everything the file promises is checked in the same fail-closed style as
  # the spec block below. The declaration and spec block are required. The
  # source key may be absent; when present, it must name the pinned source.
  [[ "$(jq -r 'type' "${MANIFEST}")" == "object" ]] \
    || fail "mtp-head.manifest.json must be a JSON object with keys {version, source, max_bytes, bytes, sha256, spec}"

  # A typo'd or invented top-level key must not read as its default, exactly as
  # for the spec block. `arm` is the one worth naming: this track has a single
  # speculative arm, so a declaration that tries to select one is stating
  # something the runner will not honour, and silence there is the wrong answer.
  unknown_top="$(jq -r 'keys[] | select(. != "version" and . != "source" and . != "max_bytes" and . != "bytes" and . != "sha256" and . != "spec")' "${MANIFEST}")"
  [[ -z "${unknown_top}" ]] \
    || fail "the declaration carries unknown top-level key(s): $(printf '%s' "${unknown_top}" | tr '\n' ' '); allowed keys are version, source, max_bytes, bytes, sha256, spec"

  if [[ "$(jq -r 'has("source")' "${MANIFEST}")" == "true" ]]; then
    declared_source="$(jq -r '.source | if type == "string" then . else tojson end' "${MANIFEST}")"
    [[ "${declared_source}" == "pinned" ]] \
      || fail "source \"${declared_source}\" is not accepted; the head is the organizer-staged pinned head, so the only accepted source is \"pinned\""
  fi

  # max_bytes, bytes and sha256 are recorded, not read: nothing bounds a load
  # by them. setup.sh verifies the target snapshot against the fixture;
  # serve-up.sh records the hashes of the staged drafter files it serves.

  if [[ "$(jq -r 'has("spec")' "${MANIFEST}")" == "true" ]]; then
    [[ "$(jq -r '.spec | type' "${MANIFEST}")" == "object" ]] \
      || fail "the \"spec\" declaration must be an object with keys {enabled, tokens}"
    # A typo'd key must not silently read as its default: config drift is a
    # refusal.
    unknown="$(jq -r '.spec | keys[] | select(. != "enabled" and . != "tokens")' "${MANIFEST}")"
    [[ -z "${unknown}" ]] \
      || fail "the \"spec\" declaration carries unknown key(s): $(printf '%s' "${unknown}" | tr '\n' ' '); allowed keys are enabled, tokens"
    [[ "$(jq -r '.spec | has("enabled")' "${MANIFEST}")" == "true" ]] \
      || fail "spec.enabled is required"
    [[ "$(jq -r '.spec.enabled | type' "${MANIFEST}")" == "boolean" ]] \
      || fail "spec.enabled must be a boolean (got $(jq -r '.spec.enabled | type' "${MANIFEST}"))"
    enabled="$(jq -r '.spec.enabled' "${MANIFEST}")"
    [[ "$(jq -r '.spec | has("tokens")' "${MANIFEST}")" == "true" ]] \
      || fail "spec.tokens is required"
    [[ "$(jq -r '.spec.tokens | type' "${MANIFEST}")" == "number" ]] \
      || fail "spec.tokens must be an integer (got $(jq -r '.spec.tokens | type' "${MANIFEST}"))"
    raw_tokens="$(jq -r '.spec.tokens' "${MANIFEST}")"
  else
    fail "the declaration has no spec block"
  fi
fi

# integer + structural 0..SPEC_MAX_TOKENS ceiling
printf '%s' "${raw_tokens}" | grep -Eq '^-?[0-9]+$' \
  || fail "spec.tokens must be an integer (got '${raw_tokens}')"
if (( raw_tokens < 0 || raw_tokens > SPEC_MAX_TOKENS )); then
  fail "spec.tokens=${raw_tokens} is outside the permitted range 0..${SPEC_MAX_TOKENS}"
fi

# --- derive the effective serve spec ----------------------------------------
# enabled:false with zero tokens is serial. enabled:true must use a permitted depth.
spec="0"
draft="0"
if [[ "${enabled}" == "true" ]]; then
  # permitted_speculative_depths is the fixture's authority. Enforce
  # membership so the serve cannot boot a depth the scored envelope refuses.
  if jq -e '.permitted_speculative_depths | arrays' >/dev/null 2>&1 < "${CONTRACT}"; then
    if [[ "$(jq -r --argjson n "${raw_tokens}" '(.permitted_speculative_depths | index($n)) != null' "${CONTRACT}")" != "true" ]]; then
      permitted="$(jq -r '.permitted_speculative_depths | map(tostring) | join(", ")' "${CONTRACT}")"
      fail "spec.tokens=${raw_tokens} is not a contract-permitted DFlash depth (permitted: ${permitted})"
    fi
  fi
  spec="1"
  draft="${raw_tokens}"
elif [[ "${raw_tokens}" != "0" ]]; then
  fail "spec.enabled=false requires spec.tokens=0"
fi

case "${1:-}" in
  speculative) echo "${spec}" ;;
  draft-len)   echo "${draft}" ;;
  describe)    if [[ "${spec}" == "1" ]]; then echo "dflash${draft}"; else echo "serial"; fi ;;
  json)        if [[ "${spec}" == "1" ]]; then printf '{"mode":"dflash","dflash":{"depth":%s}}\n' "${draft}"; else echo '{"mode":"serial"}'; fi ;;
  validate)    : ;;  # validation already ran above; a clean exit means valid
  *)
    echo "spec-declaration.sh: usage: spec-declaration.sh {speculative|draft-len|describe|json|validate}" >&2
    exit 2
    ;;
esac

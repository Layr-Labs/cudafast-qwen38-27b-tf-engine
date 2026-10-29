#!/usr/bin/env bash
#
# qwen38-measure-and-score.sh -- benchmark.json's benchmarkCommand /
# preSubmitCommand entry point for track qwen3.8-27b-tf-cuda-v1.
#
# It drives `benchd iterate --mode official` -- the SOLE scored path -- against
# the track's LIVE golden. `iterate --mode official` is timed-first, spawns the
# sandboxed runtime worker, runs the full correctness set, gates on the official
# floor/bands, and SEALS the artifact itself: it writes score.json in the
# {score, metrics} shape Yukon's ScoreFileSchema reads, its `.sha256` sidecar,
# and the per-mode benchmark-integrity sidecar. There is NO results.json to
# convert -- benchd is the sole writer of the sealed score, so this script no
# longer post-processes one. This script is TRUSTED-side tooling: it is NOT in
# editablePaths, so a submission cannot rewrite the measurement pipeline from
# inside its own archive.
#
# PAIRED, WITH A PER-BOX BASELINE. The scored model is TWO
# LEGS on the SAME box in the SAME job, over ONE live golden per pair:
#
#   leg 1  the SERIAL CONTROL, on the organizer-staged REFERENCE tree
#          (--baseline-workspace), speculation off, always;
#   leg 2  the CANDIDATE, on this checkout, at its declared draft depth.
#
# composite = (ref_prefill_spt / cand_prefill_spt)^0.25
#           * (ref_decode_spt / cand_decode_spt)^0.75, floors unchanged.
#
# NO PAIR IS STORED ANYWHERE. Not in a benchd constant, not in the fixture, not
# in a golden: the denominator is MEASURED, here, next to the candidate. The
# per-box calibration file (--baseline-calibration) is a HEALTH BAND for leg 1
# only -- benchd refuses the run when leg 1 lands outside this box's band -- and
# it is never divided by.
#
# The LIVE goldens are read FROM the fixture, never hardcoded: the fixture's
# `live_goldens` lists them, pair k measures golden (k - 1) mod N, and each
# prompt's `timed_prompt_pool[]` entry pins it ({sha256, bytes}). The pins are forwarded to benchd as
# --golden-sha256/--golden-bytes, which re-verifies the raw bytes BEFORE parse
# and refuses on any mismatch (the integrity pin). --contract carries the arm
# gate: benchd refuses, pre-GPU, to seal an official artifact unless the fixture
# declares official_scoring_enabled: true.
#
# The scoring and state rules are in docs/participant-contract.md.
#
# Usage:
#   ./tools/qwen38-measure-and-score.sh                 # full measure + seal
#   ./tools/qwen38-measure-and-score.sh --preflight-only # pre-GPU dry-run, no engine
#
# Env:
#   MLXFAST_SCORE_PATH               Where benchd seals the {score, metrics} JSON
#                                     (--score-path). Defaults to score.json
#                                     (benchmark.json always sets this explicitly).
#                                     The .sha256 sidecar and benchmark-integrity
#                                     sidecar are sealed beside it by benchd.
#   MLXFAST_BASELINE_WORKSPACE        The built REFERENCE tree the serial-control
#                                     leg runs on, passed as --baseline-workspace.
#                                     Box-only, staged out of band at the fixture's
#                                     baseline_reference_commit and verified by
#                                     tools/ranked-box-preflight.sh section 8.
#                                     REQUIRED on a real run; refused by name when
#                                     unset.
#   MLXFAST_BASELINE_CALIBRATION      This box's calibration file, passed as
#                                     --baseline-calibration. It is the HEALTH
#                                     BAND for the serial-control leg, never the
#                                     denominator. Written by tools/calibrate-box.sh.
#                                     REQUIRED on a real run; refused by name when
#                                     unset.
#   MLXFAST_LIVE_GOLDEN_CHOICE        Must be unset or empty: every job measures
#                                     every live golden, so a drawn prompt is
#                                     refused by name.
#   MLXFAST_QWEN38_GOLDEN_DIR         Directory holding the staged timed-pool
#                                     golden files. Each live golden is resolved
#                                     from it as <name>.golden.json (the
#                                     basename of the fixture entry's r2_path).
#                                     Box-only, staged out of band and
#                                     pin-verified by tools/ranked-box-preflight.sh.
#   MLXFAST_ENGINE_BIN               The Engine Protocol v1 adapter benchd spawns
#                                     (`--engine`). Default: the fixed staged path
#                                     tools/stage-cuda-engine.sh writes,
#                                     .build/release/bench-worker. The
#                                     adapter attaches to the leg's TensorFold resident
#                                     (booted per leg by benchd, below).
#   SANDBOX_USER                       When set (ranked box: qwen38sub), the
#                                     staged engine must be the sandbox shim
#                                     tools/stage-cuda-engine.sh writes: it execs
#                                     tools/run-engine-sandboxed.sh on the real
#                                     adapter at <engine>.sandboxed. benchd still
#                                     gets the normal staged path as `--engine`,
#                                     so the serial-control leg resolves the
#                                     reference workspace's own binary at that
#                                     path. Refused when the shim, the
#                                     .sandboxed adapter, the wrapper or
#                                     passwordless sudo to SANDBOX_USER is
#                                     unusable.
#   MLXFAST_WEIGHTS_PATH             Weights directory, passed as `--weights`.
#                                     Default: the on-box target snapshot
#                                     ./setup.sh verifies in place,
#                                     MLXFAST_TARGET_SNAPSHOT_DIR.
#   BENCHD                           Path to the benchd binary. Default: the
#                                     binary ./tools/fetch-benchd.sh resolves from
#                                     the dist channel and verifies against the
#                                     channel's benchd.manifest.json
#                                     (benchd-bin/benchd). No cargo build: the
#                                     ranked box has no Rust toolchain, so benchd
#                                     ships prebuilt and pinned by sha256.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

PREFLIGHT_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    *)
      echo "qwen38-measure-and-score.sh: unrecognized argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

# jq is required for reading the fixture (live_goldens + pins) and the trackId.
if ! command -v jq >/dev/null 2>&1; then
  echo "qwen38-measure-and-score.sh: jq is required (it reads the fixture pins and trackId)." >&2
  exit 1
fi

# Resolve the PINNED benchd. benchd ships as a channel PREBUILT
# (./tools/fetch-benchd.sh, verified against the channel's
# benchd.manifest.json and the pinned distribution and source commits in
# fetch-benchd.sh). fetch-benchd.sh
# accepts an already-present benchd-bin/benchd whose sha256 and bytes match the
# manifest beside it -- the offline path on the ranked box, which has no Rust
# toolchain -- and otherwise downloads and verifies it. It never yields an
# unverified binary, so this refuses rather than measuring against unpinned
# scoring code.
#
# BENCHD= from the caller is honoured and NOT hash-checked: that is a
# deliberate "use this other binary" for benchd development.
if [[ -z "${BENCHD:-}" ]]; then
  BENCHD="$("${SCRIPT_DIR}/tools/fetch-benchd.sh")"
fi
if [[ ! -x "${BENCHD}" ]]; then
  echo "qwen38-measure-and-score.sh: benchd not found at <benchd-dir>/benchd." >&2
  echo "  fetch it: ./tools/fetch-benchd.sh   (resolves the dist channel, verifies the manifest)" >&2
  echo "  or set BENCHD to an existing binary." >&2
  exit 1
fi

CONTRACT="${SCRIPT_DIR}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json"
if [[ ! -f "${CONTRACT}" ]]; then
  echo "qwen38-measure-and-score.sh: track contract fixture missing: fixtures/qwen3_8_27b_tf_cuda_v1_track.json" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# THE TRACK ID. `benchd iterate` REQUIRES MLXFAST_QWEN_MTP_TRACK_ID in every
# mode -- its `-{platform}-v{N}` suffix keys the acceptance bands, and the
# official path resolves it BEFORE the timed run. The
# value is benchmark.json's `trackId` (the PLATFORM id). benchd treats
# env != contract as a hard error, so a caller that pre-set a DIFFERENT value is
# refused here rather than silently overridden.
MANIFEST_TRACK_ID="$(jq -r '.trackId // empty' "${SCRIPT_DIR}/benchmark.json" 2>/dev/null || true)"
if [[ -z "${MANIFEST_TRACK_ID}" ]]; then
  echo "qwen38-measure-and-score.sh: benchmark.json carries no trackId; benchd iterate requires MLXFAST_QWEN_MTP_TRACK_ID and this script will not guess one." >&2
  exit 1
fi
if [[ -n "${MLXFAST_QWEN_MTP_TRACK_ID:-}" && "${MLXFAST_QWEN_MTP_TRACK_ID}" != "${MANIFEST_TRACK_ID}" ]]; then
  echo "qwen38-measure-and-score.sh: MLXFAST_QWEN_MTP_TRACK_ID is set to '${MLXFAST_QWEN_MTP_TRACK_ID}' but benchmark.json trackId is '${MANIFEST_TRACK_ID}'; the track id is ONE value and this script will not override either." >&2
  exit 1
fi
export MLXFAST_QWEN_MTP_TRACK_ID="${MANIFEST_TRACK_ID}"

# ---------------------------------------------------------------------------
# THE LIVE GOLDENS + THEIR PINS, read FROM the fixture (never hardcoded). The
# fixture's `live_goldens` lists the prompts this track scores, in order. Each
# name's timed_prompt_pool[] entry -- matched by the basename of its r2_path
# (<name>.golden.json) -- carries the {sha256, bytes} pin. A change to the list
# needs no edit here.
#
# benchd measures the pairs in turn: pair k measures golden (k - 1) mod N, so
# official_pairs is a multiple of N and every live golden gets the same number
# of pairs. The score is the mean of the per-pair composites. Every golden flag below is
# repeated once per live golden, in fixture order, and benchd matches the flags
# by position.
#
# NO PROMPT IS DRAWN. Every job measures every live golden, so a prompt choice
# handed in by the workflow (MLXFAST_LIVE_GOLDEN_CHOICE) is refused: it would
# name one prompt where the run measures all of them.
if [[ -n "${MLXFAST_LIVE_GOLDEN_CHOICE:-}" ]]; then
  echo "qwen38-measure-and-score.sh: MLXFAST_LIVE_GOLDEN_CHOICE='${MLXFAST_LIVE_GOLDEN_CHOICE}' is set, but every job measures every live golden ($(jq -c '.live_goldens' "${CONTRACT}")); refusing." >&2
  exit 1
fi
LIVE_GOLDEN_NAMES=()
while IFS= read -r name; do
  LIVE_GOLDEN_NAMES+=("${name}")
done < <(jq -r '(.live_goldens // [])[]' "${CONTRACT}")
if [[ ${#LIVE_GOLDEN_NAMES[@]} -eq 0 ]]; then
  echo "qwen38-measure-and-score.sh: fixture declares no live_goldens; there is no golden to score over." >&2
  exit 1
fi

# PER-DEPTH TIMED ORACLE. The timed leg replays the engine's FREE-RUN under the
# declared speculative spec, and a free-run trajectory forks from the serial one
# at near-tie argmaxes (see docs/participant-contract.md, "Correctness"). So the
# timed oracle must be authored under the SAME spec the leg replays: the serial
# golden is the launch reference (MTP-0), and each permitted MTP draft depth has
# its OWN golden per prompt. The spec label is derived through the one trusted
# deriver that ALSO boots the serve (tools/spec-declaration.sh describe ->
# serial|dflash1), so the oracle can never disagree with the draft depth
# that actually ran. This overrides ONLY the timed-oracle file and its pin; the
# correctness gate is untouched, because every per-depth golden carries the
# SAME serial cases[] oracle, so token fidelity is still measured against serial.
#
# A declared depth with no speculative_oracles.<prompt>.<label> entry is
# REFUSED: there is no timed oracle authored for it, and scoring it against the
# serial oracle would fail at step 0.
SPEC_DESC="$("${SCRIPT_DIR}/tools/spec-declaration.sh" describe)"

# For each live golden: the timed oracle the candidate leg verifies against
# (the serial tape, or the per-depth oracle), and the serial tape the
# serial-control leg verifies against (--control-golden).
GOLDEN_BASENAMES=()
GOLDEN_SHA256S=()
GOLDEN_BYTES_LIST=()
SERIAL_GOLDEN_BASENAMES=()
SERIAL_GOLDEN_SHA256S=()
SERIAL_GOLDEN_BYTES_LIST=()
for name in "${LIVE_GOLDEN_NAMES[@]}"; do
  serial_base="${name}.golden.json"
  pool_entry="$(jq -c --arg base "/${serial_base}" \
    'first(.timed_prompt_pool[] | select(.r2_path | endswith($base)))' "${CONTRACT}")"
  if [[ -z "${pool_entry}" || "${pool_entry}" == "null" ]]; then
    echo "qwen38-measure-and-score.sh: live golden '${name}' has no timed_prompt_pool entry (looked for an r2_path ending in /${serial_base})." >&2
    exit 1
  fi
  serial_sha="$(printf '%s' "${pool_entry}" | jq -r '.sha256 // empty')"
  serial_bytes="$(printf '%s' "${pool_entry}" | jq -r '.bytes // empty')"
  timed_base="${serial_base}" timed_sha="${serial_sha}" timed_bytes="${serial_bytes}"
  if [[ "${SPEC_DESC}" != "serial" ]]; then
    spec_entry="$(jq -c --arg n "${name}" --arg k "${SPEC_DESC}" '.speculative_oracles[$n][$k] // empty' "${CONTRACT}")"
    if [[ -z "${spec_entry}" || "${spec_entry}" == "null" ]]; then
      echo "qwen38-measure-and-score.sh: declared spec '${SPEC_DESC}' has no speculative_oracles['${name}'] entry in fixtures/qwen3_8_27b_tf_cuda_v1_track.json; no timed oracle is authored for that prompt at that draft depth. Refusing rather than scoring it against the serial oracle." >&2
      exit 1
    fi
    timed_base="$(printf '%s' "${spec_entry}" | jq -r '.r2_path // empty')"
    timed_base="${timed_base##*/}"
    timed_sha="$(printf '%s' "${spec_entry}" | jq -r '.sha256 // empty')"
    timed_bytes="$(printf '%s' "${spec_entry}" | jq -r '.bytes // empty')"
    if [[ -z "${timed_base}" || "${timed_base}" != *.golden.json ]]; then
      echo "qwen38-measure-and-score.sh: speculative_oracles['${name}']['${SPEC_DESC}'] names no *.golden.json (r2_path='${timed_base}')." >&2
      exit 1
    fi
    echo "qwen38-measure-and-score.sh: declared spec ${SPEC_DESC}; timed oracle for ${name} is ${timed_base} (sha256 ${timed_sha}, ${timed_bytes} bytes)" >&2
  fi
  for pin in "${serial_sha} ${serial_bytes}" "${timed_sha} ${timed_bytes}"; do
    if ! printf '%s' "${pin% *}" | grep -Eq '^[0-9a-f]{64}$' \
      || ! printf '%s' "${pin#* }" | grep -Eq '^[1-9][0-9]*$'; then
      echo "qwen38-measure-and-score.sh: live golden '${name}' is unarmed or malformed (pin '${pin}'); nothing can be pin-verified against it." >&2
      exit 1
    fi
  done
  GOLDEN_BASENAMES+=("${timed_base}")
  GOLDEN_SHA256S+=("${timed_sha}")
  GOLDEN_BYTES_LIST+=("${timed_bytes}")
  SERIAL_GOLDEN_BASENAMES+=("${serial_base}")
  SERIAL_GOLDEN_SHA256S+=("${serial_sha}")
  SERIAL_GOLDEN_BYTES_LIST+=("${serial_bytes}")
done

# The goldens are box-only, staged out of band into MLXFAST_QWEN38_GOLDEN_DIR
# and pin-verified by tools/ranked-box-preflight.sh.
GOLDEN_DIR="${MLXFAST_QWEN38_GOLDEN_DIR:-}"

# The warm-up uses one separate, never-scored golden. Fail closed when the
# fixture pin is not ready or this benchd cannot accept the warm-up flag.
WARMUP_GOLDEN_BASE=""
WARMUP_GOLDEN_SHA256=""
WARMUP_GOLDEN_BYTES=""
WARMUP_GOLDEN_PATH=""
WARMUP_ARGS=()
warmup_entry="$(jq -c '.warmup_golden // empty' "${CONTRACT}")"
if [[ -n "${warmup_entry}" && "${warmup_entry}" != "null" ]]; then
  warmup_r2_path="$(printf '%s' "${warmup_entry}" | jq -r '.r2_path // empty')"
  WARMUP_GOLDEN_BASE="${warmup_r2_path##*/}"
  WARMUP_GOLDEN_SHA256="$(printf '%s' "${warmup_entry}" | jq -r '.sha256 // empty')"
  WARMUP_GOLDEN_BYTES="$(printf '%s' "${warmup_entry}" | jq -r '.bytes // empty')"
  if [[ -z "${warmup_r2_path}" || -z "${WARMUP_GOLDEN_BASE}" || "${WARMUP_GOLDEN_BASE}" != *.golden.json ]] \
    || ! printf '%s' "${WARMUP_GOLDEN_SHA256}" | grep -Eq '^[0-9a-f]{64}$' \
    || ! printf '%s' "${WARMUP_GOLDEN_BYTES}" | grep -Eq '^[1-9][0-9]*$'; then
    echo "qwen38-measure-and-score.sh: warmup_golden is unarmed or malformed (r2_path='${warmup_r2_path}', sha256='${WARMUP_GOLDEN_SHA256}', bytes='${WARMUP_GOLDEN_BYTES}')." >&2
    exit 1
  fi
  if [[ -n "${GOLDEN_DIR}" ]]; then
    WARMUP_GOLDEN_PATH="${GOLDEN_DIR}/${WARMUP_GOLDEN_BASE}"
  fi
  if ! "${BENCHD}" iterate --help 2>&1 \
    | grep -Eq -- '(^|[[:space:]])--warmup-golden([[:space:]]|$)'; then
    echo "qwen38-measure-and-score.sh: REFUSING -- the fixture declares warmup_golden but this benchd has no --warmup-golden; this old benchd would ignore the anti-memoization guard." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# --preflight-only: a pre-GPU DRY RUN that exercises the arm gate and (when the
# golden is staged) the integrity pin, WITHOUT spawning the engine or loading the
# model. No score is written. The full run below enforces both refusals through
# benchd itself; this is the loud early gate.
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  # ARM GATE (mirror of benchd's enforce_official_scoring_enabled): an official
  # run refuses unless the fixture declares official_scoring_enabled: true. false
  # and ABSENT both refuse -- an absent arm state is not an armed one. benchd
  # enforces this itself on the real run; mirroring it here makes the pre-GPU
  # dry-run honest rather than a rubber stamp.
  armed="$(jq -r '.official_scoring_enabled // false' "${CONTRACT}")"
  if [[ "${armed}" != "true" ]]; then
    echo "qwen38-measure-and-score.sh: preflight REFUSING -- fixtures/qwen3_8_27b_tf_cuda_v1_track.json does not declare official_scoring_enabled: true (got '${armed}')." >&2
    echo "  benchd refuses to seal an official scoring artifact for an unarmed track; this dry-run mirrors that refusal." >&2
    exit 1
  fi
  echo "qwen38-measure-and-score.sh: preflight -- arm gate OK (official_scoring_enabled: true), live goldens ${LIVE_GOLDEN_NAMES[*]}" >&2

  # INTEGRITY PIN: for each staged live golden, validate-golden re-verifies its
  # raw bytes against the pin AND load-validates it (reference-model pin from
  # --contract), with NO engine spawned. When one is not staged (e.g. an
  # off-box pre-submit), the pin is still enforced by benchd on the real run.
  for i in "${!GOLDEN_BASENAMES[@]}"; do
    if [[ -n "${GOLDEN_DIR}" && -f "${GOLDEN_DIR}/${GOLDEN_BASENAMES[$i]}" ]]; then
      "${BENCHD}" validate-golden \
        --golden "${GOLDEN_DIR}/${GOLDEN_BASENAMES[$i]}" \
        --golden-sha256 "${GOLDEN_SHA256S[$i]}" \
        --golden-bytes "${GOLDEN_BYTES_LIST[$i]}" \
        --contract "${CONTRACT}"
    else
      echo "qwen38-measure-and-score.sh: preflight -- live golden not staged (MLXFAST_QWEN38_GOLDEN_DIR unset or ${GOLDEN_BASENAMES[$i]} absent); benchd re-verifies the {sha256, bytes} pin on the real run." >&2
    fi
  done
  if [[ -n "${WARMUP_GOLDEN_PATH}" && -f "${WARMUP_GOLDEN_PATH}" ]]; then
    "${BENCHD}" validate-golden \
      --golden "${WARMUP_GOLDEN_PATH}" \
      --golden-sha256 "${WARMUP_GOLDEN_SHA256}" \
      --golden-bytes "${WARMUP_GOLDEN_BYTES}" \
      --contract "${CONTRACT}"
  elif [[ -n "${WARMUP_GOLDEN_PATH}" ]]; then
    echo "qwen38-measure-and-score.sh: preflight -- warm-up golden not staged at <golden-dir>/${WARMUP_GOLDEN_BASE}; benchd re-verifies the pin on the real run." >&2
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# A REAL RUN needs BOTH halves of the paired baseline, by name. Neither has a
# default and neither may be guessed: without the reference workspace there is no
# serial-control leg to divide by, and without this box's calibration there is
# nothing to say whether that leg landed where this box lands.
BASELINE_WORKSPACE="${MLXFAST_BASELINE_WORKSPACE:-}"
BASELINE_CALIBRATION="${MLXFAST_BASELINE_CALIBRATION:-}"
if [[ -z "${BASELINE_WORKSPACE}" ]]; then
  echo "qwen38-measure-and-score.sh: MLXFAST_BASELINE_WORKSPACE is unset." >&2
  echo "  The scored path is PAIRED: the serial-control leg runs on the staged reference tree at the fixture's baseline_reference_commit. Stage it with tools/stage-baseline-workspace.sh and export the path. Refusing rather than scoring against a stored pair." >&2
  exit 1
fi
if [[ ! -d "${BASELINE_WORKSPACE}" ]]; then
  echo "qwen38-measure-and-score.sh: MLXFAST_BASELINE_WORKSPACE=<reference-workspace> is not a directory." >&2
  exit 1
fi
if [[ -z "${BASELINE_CALIBRATION}" ]]; then
  echo "qwen38-measure-and-score.sh: MLXFAST_BASELINE_CALIBRATION is unset." >&2
  echo "  Every ranked box carries its own calibration file -- the health band the serial-control leg must land inside. Calibrate this box with tools/calibrate-box.sh and export the path." >&2
  exit 1
fi
if [[ ! -f "${BASELINE_CALIBRATION}" ]]; then
  echo "qwen38-measure-and-score.sh: MLXFAST_BASELINE_CALIBRATION=<calibration-file> is not a file." >&2
  exit 1
fi

# The trusted preflight validates the required method field against this
# fixture. The pinned benchd v3 reader uses deny_unknown_fields, so give it a
# private JSON copy with only the wrapper-owned method removed.
# The measured values and benchd identity do not change.
CALIBRATION_FOR_BENCHD="$(mktemp "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/baseline-calibration-v3.XXXXXX")"
trap 'rm -f -- "${CALIBRATION_FOR_BENCHD}"' EXIT
if ! jq 'del(.method)' "${BASELINE_CALIBRATION}" > "${CALIBRATION_FOR_BENCHD}"; then
  rm -f "${CALIBRATION_FOR_BENCHD}"
  echo "qwen38-measure-and-score.sh: baseline calibration is not valid JSON: <calibration-file>" >&2
  exit 1
fi

# A BOX-PRESET SERVE SPEC IS REFUSED ON THE PAIRED PATH. Under one leg a preset
# that AGREED with the declaration was harmless. Under two legs it is not: the
# value reaches the environment of BOTH legs' serve scripts, and leg 1 must be
# serial whatever the candidate declares. So it is refused here as well as in
# tools/ranked-box-preflight.sh section 7.
if [[ -n "${SERVE_UP_SPECULATIVE:-}" ]]; then
  echo "qwen38-measure-and-score.sh: SERVE_UP_SPECULATIVE=${SERVE_UP_SPECULATIVE} is preset in the environment; on the paired path it would reach the serial-control leg's serve too, and that leg is serial regardless of the candidate's declaration." >&2
  exit 1
fi

# AN INHERITED RESIDENT SOCKET IS REFUSED. benchd boots each leg's resident
# itself and injects that leg's socket into that leg's worker spawns. A socket
# already in the environment would be a THIRD resident -- one neither leg booted
# -- and every worker that inherited it would measure whatever is on the other
# end. benchd refuses one in its own environment; this refuses it earlier, by
# name.
for var in DS4_RESIDENT_SOCKET BENCH_WORKER_RESIDENT_SOCKET; do
  eval "value=\${${var}:-}"
  if [[ -n "${value}" ]]; then
    echo "qwen38-measure-and-score.sh: ${var}=<socket-dir> is set in the environment; benchd boots each leg's resident and injects its socket, so an inherited one would point both legs' workers at a resident neither leg booted." >&2
    exit 1
  fi
done

# A REAL RUN needs every staged live golden.
if [[ -z "${GOLDEN_DIR}" || ! -d "${GOLDEN_DIR}" ]]; then
  cat >&2 <<EOF
qwen38-measure-and-score.sh: MLXFAST_QWEN38_GOLDEN_DIR is unset or missing.
  The live goldens (${GOLDEN_BASENAMES[*]}) are staged onto the box out of band
  (docs/sandbox-design.md under Operator setup) and this job holds no credential
  to fetch them. There is nothing this script can do here except refuse.
EOF
  exit 1
fi
GOLDEN_ARGS=()
for i in "${!GOLDEN_BASENAMES[@]}"; do
  if [[ ! -f "${GOLDEN_DIR}/${GOLDEN_BASENAMES[$i]}" ]]; then
    echo "qwen38-measure-and-score.sh: live golden not found at <golden-dir>/${GOLDEN_BASENAMES[$i]}" >&2
    echo "  live_goldens names '${LIVE_GOLDEN_NAMES[$i]}' (fixtures/qwen3_8_27b_tf_cuda_v1_track.json); stage ${GOLDEN_BASENAMES[$i]} into MLXFAST_QWEN38_GOLDEN_DIR." >&2
    exit 1
  fi
  GOLDEN_ARGS+=(--golden "${GOLDEN_DIR}/${GOLDEN_BASENAMES[$i]}" --golden-sha256 "${GOLDEN_SHA256S[$i]}" --golden-bytes "${GOLDEN_BYTES_LIST[$i]}")
done
if [[ -n "${WARMUP_GOLDEN_PATH}" ]]; then
  if [[ ! -f "${WARMUP_GOLDEN_PATH}" ]]; then
    echo "qwen38-measure-and-score.sh: warm-up golden not found at <golden-dir>/${WARMUP_GOLDEN_BASE}" >&2
    exit 1
  fi
  WARMUP_ARGS=(
    --warmup-golden "${WARMUP_GOLDEN_PATH}"
    --warmup-golden-sha256 "${WARMUP_GOLDEN_SHA256}"
    --warmup-golden-bytes "${WARMUP_GOLDEN_BYTES}"
  )
fi

# --engine. `iterate` spawns the Engine Protocol v1 adapter, which connects to
# the leg's TensorFold resident. Default to the fixed staged path
# tools/stage-cuda-engine.sh writes, so the documented invocation is
# self-sufficient; MLXFAST_ENGINE_BIN overrides it for engine development.
ENGINE_BIN="${MLXFAST_ENGINE_BIN:-${SCRIPT_DIR}/.build/release/bench-worker}"
if [[ ! -x "${ENGINE_BIN}" ]]; then
  echo "qwen38-measure-and-score.sh: engine binary missing or not executable: <workspace>/.build/release/bench-worker" >&2
  echo "  build + stage it (./setup.sh, or tools/stage-cuda-engine.sh), or set MLXFAST_ENGINE_BIN." >&2
  exit 1
fi

SANDBOX_WRAPPER="${SCRIPT_DIR}/tools/run-engine-sandboxed.sh"
if [[ -n "${SANDBOX_USER:-}" ]]; then
  if [[ ! -x "${SANDBOX_WRAPPER}" ]]; then
    echo "qwen38-measure-and-score.sh: SANDBOX_USER=${SANDBOX_USER} but the engine wrapper is missing or not executable: tools/run-engine-sandboxed.sh" >&2
    exit 1
  fi
  # The staged engine must be exactly the shim stage-cuda-engine.sh writes, and
  # the adapter it names a regular executable file; otherwise the candidate leg
  # would run outside the sandbox.
  # The shim bakes in this step's SANDBOX_* and RUNNER_TEMP (same job, same
  # values as at staging); stage-cuda-engine.sh --print-shim is its one source.
  real_engine="$(cd "$(dirname -- "${ENGINE_BIN}")" && pwd -P)/$(basename -- "${ENGINE_BIN}").sandboxed"
  expected_shim="$("${SCRIPT_DIR}/tools/stage-cuda-engine.sh" --print-shim)" || exit 1
  if [[ -L "${ENGINE_BIN}" || "$(cat -- "${ENGINE_BIN}" 2>/dev/null)" != "${expected_shim}" ]]; then
    echo "qwen38-measure-and-score.sh: SANDBOX_USER=${SANDBOX_USER} but <workspace>/.build/release/bench-worker is not the sandbox shim; stage it with SANDBOX_USER set (tools/engine/build.sh)." >&2
    exit 1
  fi
  if [[ -L "${real_engine}" || ! -f "${real_engine}" || ! -x "${real_engine}" ]]; then
    echo "qwen38-measure-and-score.sh: SANDBOX_USER=${SANDBOX_USER} but the sandboxed adapter is missing or not a regular executable file: <workspace>/.build/release/bench-worker.sandboxed" >&2
    exit 1
  fi
  if ! command -v sudo >/dev/null 2>&1; then
    echo "qwen38-measure-and-score.sh: SANDBOX_USER=${SANDBOX_USER} but sudo is not on PATH." >&2
    exit 1
  fi
  if ! sudo -n -u "${SANDBOX_USER}" -- true 2>/dev/null; then
    echo "qwen38-measure-and-score.sh: SANDBOX_USER=${SANDBOX_USER} but passwordless sudo to that user is unusable (see docs/sandbox-design.md, 'Operator setup')." >&2
    exit 1
  fi
  echo "qwen38-measure-and-score.sh: candidate leg engine <workspace>/.build/release/bench-worker is the sandbox shim (adapter <workspace>/.build/release/bench-worker.sandboxed as ${SANDBOX_USER})" >&2
fi

# --weights. benchd resolves the target weights from `--weights`. Default to the
# on-box target snapshot ./setup.sh verifies in place
# (MLXFAST_TARGET_SNAPSHOT_DIR); MLXFAST_WEIGHTS_PATH overrides.
WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-${MLXFAST_TARGET_SNAPSHOT_DIR:-}}"
if [[ -z "${WEIGHTS_PATH}" || ! -d "${WEIGHTS_PATH}" ]]; then
  echo "qwen38-measure-and-score.sh: the target weights directory is missing or unset: '<target-snapshot>'" >&2
  echo "  Set MLXFAST_TARGET_SNAPSHOT_DIR to the on-box target snapshot (./setup.sh verifies it), or MLXFAST_WEIGHTS_PATH to override." >&2
  exit 1
fi
# The resident serves this path. Reset-time memory checks must use it too.
export SERVE_UP_WEIGHTS_DIR="${WEIGHTS_PATH}"

SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"

# THE SOLE SCORED PATH: benchd iterate --mode official over the live golden.
# benchd measures BOTH legs -- the serial control on --baseline-workspace and the
# candidate on this checkout -- and the score is the ratio of the two. There is no
# OFFICIAL_BASELINE constant and no stored pair in the golden. benchd SEALS
# score.json (the {score, metrics} shape Yukon reads), score.json.sha256, and the
# benchmark-integrity sidecar itself; this script does no post-conversion.
#
# --golden-sha256/--golden-bytes are the INTEGRITY PIN (benchd re-verifies the
# raw bytes before parse and refuses on mismatch). --contract carries the ARM
# GATE (benchd refuses, pre-GPU, unless official_scoring_enabled: true).
# --baseline-calibration carries this box's HEALTH BAND: benchd refuses the run
# when the control leg lands outside it, and never divides by it.

# --control-golden: the serial tape for the serial-control leg, one per live
# golden, in the same order as --golden. A benchd that does not know the flag
# verifies leg 1 against the candidate's per-depth tape, which is only the same
# tape at depth 0; a speculative declaration on such a benchd is refused rather
# than measured against the wrong oracle.
CONTROL_ARGS=()
if "${BENCHD}" iterate --help 2>&1 | grep -q -- '--control-golden'; then
  for i in "${!SERIAL_GOLDEN_BASENAMES[@]}"; do
    control_path="${GOLDEN_DIR}/${SERIAL_GOLDEN_BASENAMES[$i]}"
    if [[ ! -f "${control_path}" ]]; then
      echo "qwen38-measure-and-score.sh: serial golden not found at <golden-dir>/${SERIAL_GOLDEN_BASENAMES[$i]}; the serial-control leg has no tape to verify against." >&2
      exit 1
    fi
    CONTROL_ARGS+=(--control-golden "${control_path}" --control-golden-sha256 "${SERIAL_GOLDEN_SHA256S[$i]}" --control-golden-bytes "${SERIAL_GOLDEN_BYTES_LIST[$i]}")
    echo "qwen38-measure-and-score.sh: serial-control leg for ${LIVE_GOLDEN_NAMES[$i]} verifies against ${SERIAL_GOLDEN_BASENAMES[$i]} (sha256 ${SERIAL_GOLDEN_SHA256S[$i]}, ${SERIAL_GOLDEN_BYTES_LIST[$i]} bytes)" >&2
  done
elif [[ "${SPEC_DESC}" != "serial" ]]; then
  echo "qwen38-measure-and-score.sh: REFUSING -- the declaration is ${SPEC_DESC} but this benchd has no --control-golden; the serial-control leg would be verified against the ${SPEC_DESC} tape." >&2
  exit 1
fi
iterate_official=(
  "${BENCHD}" iterate
  --engine "${ENGINE_BIN}"
  --weights "${WEIGHTS_PATH}"
  "${GOLDEN_ARGS[@]}"
  --mode official
  --score-path "${SCORE_PATH}"
  --contract "${CONTRACT}"
  --baseline-workspace "${BASELINE_WORKSPACE}"
  --baseline-calibration "${CALIBRATION_FOR_BENCHD}"
)
iterate_official+=( ${CONTROL_ARGS[@]+"${CONTROL_ARGS[@]}"} )
iterate_official+=( ${WARMUP_ARGS[@]+"${WARMUP_ARGS[@]}"} )

# --box names the runner the band is checked against. benchd reads RUNNER_NAME
# when the flag is absent, so this is the explicit form of what would happen
# anyway -- and it puts the box in the command line the log records.
if [[ -n "${RUNNER_NAME:-}" ]]; then
  iterate_official+=( --box "${RUNNER_NAME}" )
fi

# The candidate legs must REQUEST the declared spec. benchd's official path
# sends free_decode_begin with no spec unless it is told one, and the
# the worker then resolves serial, so a declared DFlash leg would be measured
# as serial while its oracle is the DFlash one. The explicit candidate spec is
# a refusal boundary, never a silent serial fallback.
if [[ "${SPEC_DESC}" != "serial" ]]; then
  iterate_official+=( --candidate-spec "$("${SCRIPT_DIR}/tools/spec-declaration.sh" json)" )
  echo "qwen38-measure-and-score.sh: benchd will request the declared ${SPEC_DESC} spec" >&2
fi

# TWO SERVES, ONE AT A TIME -- AND benchd SEQUENCES THEM.
#
# tools/serve-up.sh boots ONE TensorFold resident, which loads the target weights
# and one admitted cache and holds the GPU for as long as it runs. Under the RETIRED single-leg model this
# script wrapped the whole benchd invocation in ONE such serve. Under the paired
# model that is no longer possible, for two separate reasons:
#
#   * leg 1 must run the REFERENCE tree's engine. A candidate-tree resident
#     serving both legs would divide the candidate by itself.
#   * the two residents must not overlap. serve-up.sh's own memory plan requires
#     the target weights plus its minimum cache free BEFORE it loads. A second resident would
#     also share the GPU with the leg being measured. So leg 1's serve boots, leg 1 is measured, that serve
#     is torn down, and only then does leg 2's serve boot.
#
# So this script boots NO serve. It hands benchd both trees, and benchd boots
# each leg's resident in turn through the PER-LEG VERBS, run from that leg's own
# workspace:
#
#   tools/serve-up.sh --boot --spec serial|dflash --draft-len N --socket-out FILE
#   tools/serve-up.sh --stop --socket PATH
#
#   leg 1  ${MLXFAST_BASELINE_WORKSPACE}/tools/serve-up.sh
#            --boot --spec serial --draft-len 0
#          ALWAYS serial: the flag is the authority in --boot mode, and
#          serve-up.sh refuses a disagreeing SERVE_UP_SPECULATIVE rather than
#          honouring it. So the control leg is serial whatever the candidate
#          declares.
#   leg 2  this tree's tools/serve-up.sh --boot at the DECLARED spec, which
#          benchd derives from the candidate spec given above.
#
# benchd reads the socket back from the first line of the --socket-out file and
# injects it into that leg's worker spawns as DS4_RESIDENT_SOCKET and
# BENCH_WORKER_RESIDENT_SOCKET. It refuses a socket inherited in its OWN
# environment, so nothing else in the job may export one -- the loop above
# refuses that earlier, by name.
#
# --stop runs on success and on failure alike, and it is idempotent, so a leg
# that fails mid-measurement still frees the box for the other leg.
#
# Both serves run inside the workflow step's GPU lock; neither takes a lock of
# its own.
#
# ONE RETRY PER JOB (single budget), for either:
#   (a) a control-leg TOKEN DIVERGENCE (SERIAL-CONTROL-LEG-FAILED: benchmark
#       <prefill|free-run decode seed|free-run decode> token mismatch ...), or
#   (b) a pair-1 CANDIDATE resident boot failure before any candidate timing is
#       sealed (LEG-SERVE-BOOT-FAILED: the candidate leg's ... with
#       metrics.paired_legs empty — pair 2's candidate boot keeps pair 1 in
#       paired_legs and is never retried), or
#   (c) a pair-1 SERIAL-CONTROL resident boot failure before any pair completes
#       (LEG-SERVE-BOOT-FAILED: the serial-control leg's ... with
#       metrics.paired_legs empty — pair 2's control boot keeps pair 1 in
#       paired_legs and is never retried; a candidate could provoke it after
#       its own leg ran), or
#   (d) a pair-1 SERIAL-CONTROL calibration-band failure before any pair timing
#       is sealed (pair 1 of N: SERIAL-CONTROL-LEG-OUTSIDE-BAND: ... with
#       metrics.paired_legs empty). Pair 2 or later is never retried: its
#       control leg runs after the candidate leg, so a candidate could provoke
#       it to buy a second draw.
# Any of these re-runs the whole measurement once from scratch after the
# teardown below. The second attempt's seal is the result; the first attempt's
# timings are discarded, never compared. Every other failure is final,
# INCLUDING the other control-leg failures (spawn, session, protocol, cool
# gate, memory): pair 2's control leg runs after the candidate's leg, so a
# candidate could provoke those on purpose to buy a second draw. A second
# control-leg token divergence, pair-1 serial-control boot failure, or pair-1
# serial-control calibration-band failure exits 75: an infrastructure outcome,
# not a submission failure. The attempt count is logged as
# control_leg_attempts=N; it
# is not sealed, because score.json is pinned by its .sha256 and
# benchmark-integrity.json and only benchd writes it.
#
# The job log carries benchd's score marker exactly once, so each attempt's
# stdout (benchd's copy of score.json) and marker are
# held in a private directory and printed only for the attempt that counts.
HELD="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/measure-held.XXXXXX")"
trap 'rm -rf -- "${HELD}"; rm -f -- "${CALIBRATION_FOR_BENCHD}"' EXIT
INTEGRITY_PATH="$(dirname -- "${SCORE_PATH}")/benchmark-integrity.json"
PUBLIC_LOG_REDACT_AWK="${SCRIPT_DIR}/tools/public-log-redact.awk"
if ! PUBLIC_LOG_HOST="$(hostname 2>/dev/null)" || [[ -z "${PUBLIC_LOG_HOST}" ]]; then
  echo "qwen38-measure-and-score.sh: warning: hostname lookup failed; continuing without hostname redaction" >&2
  PUBLIC_LOG_HOST=""
fi
export PUBLIC_LOG_CALIBRATION_FILE="${CALIBRATION_FOR_BENCHD}"
export PUBLIC_LOG_HOST

redact() {
  "${SCRIPT_DIR}/tools/public-log-redact.sh"
}

run_attempt() {
  rm -f -- "${HELD}/stdout" "${HELD}/marker"
  set +e
  "${iterate_official[@]}" 2>&1 >"${HELD}/stdout" \
    | LC_ALL=C awk -f "${PUBLIC_LOG_REDACT_AWK}" -v held="${HELD}/marker" -v host="${PUBLIC_LOG_HOST}" >&2
  local rc="${PIPESTATUS[0]}"
  set -e
  return "${rc}"
}

# Only THIS attempt's seal counts: the marker must have been held from this run
# (the seal files are removed before every attempt), and the error must be the
# reference's own token divergence (benchd bench-runner's TokenMismatch labels,
# directly or inside the reference warmup leg).
control_leg_diverged() {
  [[ -f "${HELD}/marker" ]] || return 1
  jq -e '.passed == false and ((.metrics.error // "") | test(
      "^SERIAL-CONTROL-LEG-FAILED: (reference warmup leg failed: )?benchmark (prefill|free-run decode seed|free-run decode) token mismatch( |:|$)"))' \
    "${SCORE_PATH}" >/dev/null 2>&1
}

# Pair 1 only: benchd seals no paired_legs row until a pair completes; a
# candidate boot failure on pair 2 carries pair 1 in paired_legs (audit trail).
# benchd omits paired_legs when it is empty (score.rs: skip_serializing_if), so
# an ABSENT key is the pair-1 case; the pinned benchd always seals the completed
# pairs on a candidate boot refusal (official.rs, with_measured_pairs).
pair1_candidate_boot_failed() {
  [[ -f "${HELD}/marker" ]] || return 1
  jq -e '.passed == false
    and ((.metrics.error // "") | test("^LEG-SERVE-BOOT-FAILED: the candidate leg"))
    and ((.metrics.paired_legs // []) | length == 0)' \
    "${SCORE_PATH}" >/dev/null 2>&1
}

# Pair 1 only: same paired_legs empty/absent rule as pair1_candidate_boot_failed.
pair1_serial_control_boot_failed() {
  [[ -f "${HELD}/marker" ]] || return 1
  jq -e '.passed == false
    and ((.metrics.error // "") | test("^LEG-SERVE-BOOT-FAILED: the serial-control leg"))
    and ((.metrics.paired_legs // []) | length == 0)' \
    "${SCORE_PATH}" >/dev/null 2>&1
}

# benchd seals error text with every digit run replaced by "#" (score.rs seal_engine_chosen_text), so the
# sealed error reads "pair # of #: ..."; an empty or absent paired_legs is what proves pair 1.
pair1_serial_control_out_of_band() {
  [[ -f "${HELD}/marker" ]] || return 1
  jq -e '.passed == false
    and ((.metrics.error // "") | test("^pair (#|1) of (#|[0-9]+): SERIAL-CONTROL-LEG-OUTSIDE-BAND: "))
    and ((.metrics.paired_legs // []) | length == 0)' \
    "${SCORE_PATH}" >/dev/null 2>&1
}

measurement_retry_warranted() {
  control_leg_diverged || pair1_candidate_boot_failed || pair1_serial_control_boot_failed \
    || pair1_serial_control_out_of_band
}
remove_seal() { rm -f -- "${SCORE_PATH}" "${SCORE_PATH}.sha256" "${INTEGRITY_PATH}"; }

# Nothing from attempt 1 may reach attempt 2 (contract: no reuse across runs):
# its seal is deleted, every sandbox process is killed, everything the sandbox
# user left in the shared temp dirs and its home is removed, and so is all of
# this job's per-job directory the sandbox user can write -- the same five the
# workflow's cleanup step empties. None is read at run time after the native
# worker is staged. Memory
# must be free again before benchd boots the next resident. benchd has already
# stopped each leg's resident (--stop runs on failure too).
# Kill stragglers, wipe shared dirs, wait for memory (ranked sandbox only).
sandbox_reset_shared() {
  [[ -n "${SANDBOX_USER:-}" ]] || return 0
  local left=""
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sudo -n -u "${SANDBOX_USER}" -- kill -KILL -1 >/dev/null 2>&1 || true
    left="$(pgrep -u "${SANDBOX_USER}" 2>/dev/null | tr '\n' ' ' || true)"
    [[ -z "${left// /}" ]] && break
    sleep 0.5
  done
  if [[ -n "${left// /}" ]]; then
    echo "qwen38-measure-and-score.sh: ${SANDBOX_USER} processes survived kill -KILL -1: ${left}" >&2
    return 1
  fi
  "${SCRIPT_DIR}/tools/sandbox-wipe.sh" || return 1
  bash "${SCRIPT_DIR}/tools/wait-for-memory.sh"
}

clear_sandbox_job_dirs() {
  [[ -n "${SANDBOX_ROOT:-}" && -n "${SANDBOX_JOB:-}" ]] || return 1
  local job_dir="${SANDBOX_ROOT}/${SANDBOX_JOB}"
  local name sub count
  for name in zig-global-cache zig-cache out sock adapter-home; do
    sub="${job_dir}/${name}"
    [[ -d "${sub}" ]] || continue
    # Same five dirs as .github/workflows/benchmark.yml sandbox cleanup; -P never
    # follows symlinks out of the sub dir; -mindepth 1 keeps the sub dir itself.
    sudo -n -u "${SANDBOX_USER}" -- chmod -R u+rwX "${sub}" >/dev/null 2>&1 || true
    # Each user removes what it owns; only the count below decides success.
    sudo -n -u "${SANDBOX_USER}" -- find -P "${sub}" -mindepth 1 -delete >/dev/null 2>&1 || true
    find -P "${sub}" -mindepth 1 -delete >/dev/null 2>&1 || true
    count="$(find -P "${sub}" -mindepth 1 -printf . 2>/dev/null | wc -c)" || return 1
    if (( count != 0 )); then
      echo "qwen38-measure-and-score.sh: sandbox job subdirectory ${name} is not empty after cleanup ($((count)) leftover entries)" >&2
      return 1
    fi
  done
}

between_attempts() {
  remove_seal
  if [[ -n "${SANDBOX_USER:-}" ]]; then
    sandbox_reset_shared || return 1
    clear_sandbox_job_dirs || return 1
  else
    bash "${SCRIPT_DIR}/tools/wait-for-memory.sh"
  fi
}

attempts=1
rc=0
remove_seal
if [[ -n "${SANDBOX_USER:-}" ]]; then
  if ! sandbox_reset_shared; then
    echo "::error title=Pre-measurement sandbox reset failed::no score sealed: sandbox leftovers from the submission could not be removed" >&2
    exit 75
  fi
fi
run_attempt || rc=$?
if [[ "${rc}" -ne 0 ]] && measurement_retry_warranted; then
  if control_leg_diverged; then
    echo "qwen38-measure-and-score.sh: attempt 1 DISCARDED, the serial control leg diverged: $(jq -r '.metrics.error' "${SCORE_PATH}")" >&2
  elif pair1_serial_control_boot_failed; then
    printf '%s%s\n' \
      "qwen38-measure-and-score.sh: attempt 1 DISCARDED, the pair-1 serial-control resident did not boot (no pair timing sealed): " \
      "$(jq -r '.metrics.error' "${SCORE_PATH}")" | redact >&2 \
      || echo "qwen38-measure-and-score.sh: public log redaction failed; retry reason withheld" >&2
  elif pair1_serial_control_out_of_band; then
    printf '%s%s\n' \
      "qwen38-measure-and-score.sh: attempt 1 DISCARDED, the pair-1 serial-control leg was outside this box's band (no pair timing sealed): " \
      "$(jq -r '.metrics.error' "${SCORE_PATH}")" | redact >&2 \
      || echo "qwen38-measure-and-score.sh: public log redaction failed; retry reason withheld" >&2
  else
    printf '%s%s\n' \
      "qwen38-measure-and-score.sh: attempt 1 DISCARDED, the pair-1 candidate resident did not boot (no candidate timing sealed): " \
      "$(jq -r '.metrics.error' "${SCORE_PATH}")" | redact >&2 \
      || echo "qwen38-measure-and-score.sh: public log redaction failed; retry reason withheld" >&2
  fi
  if ! between_attempts; then
    echo "::error title=Measurement retry teardown failed::the box could not be reset after a retriable infrastructure failure; infrastructure outcome, not a submission failure" >&2
    exit 75
  fi
  echo "qwen38-measure-and-score.sh: re-running the whole measurement once (attempt 2 of 2)" >&2
  attempts=2
  rc=0
  run_attempt || rc=$?
fi
if ! redact < "${HELD}/stdout"; then
  echo "qwen38-measure-and-score.sh: public log redaction failed; score output withheld" >&2
fi
if [[ -f "${HELD}/marker" ]]; then cat -- "${HELD}/marker" >&2; fi
echo "qwen38-measure-and-score.sh: control_leg_attempts=${attempts}" >&2
if [[ "${rc}" -ne 0 && "${attempts}" -eq 2 ]] && control_leg_diverged; then
  echo "::error title=Serial control leg diverged twice::infrastructure outcome, not a submission failure; re-dispatch this job" >&2
  exit 75
fi
if [[ "${rc}" -ne 0 && "${attempts}" -eq 2 ]] && pair1_serial_control_boot_failed; then
  echo "::error title=Serial-control leg boot failed twice::infrastructure outcome, not a submission failure; re-dispatch this job" >&2
  exit 75
fi
if [[ "${rc}" -ne 0 && "${attempts}" -eq 2 ]] && pair1_serial_control_out_of_band; then
  echo "::error title=Serial-control leg outside band twice::infrastructure outcome, not a submission failure; re-dispatch this job" >&2
  exit 75
fi
exit "${rc}"

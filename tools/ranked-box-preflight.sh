#!/usr/bin/env bash
#
# ranked-box-preflight.sh -- the fail-closed gate the ranked job runs BEFORE
# ./setup.sh and before any measurement, for track qwen3.8-27b-tf-cuda-v1.
#
# WHAT IT IS FOR. The ranked job holds NO CREDENTIAL by design (bundles-no-keys:
# a box receives staged bundles, never keys), so the hidden material this track
# measures against cannot be fetched by the job itself -- it is staged onto the
# box out of band by the organizer. That moves the whole question from "can we
# download it" to "is what is on this box the pinned material". This script
# answers that question, and refuses when the answer is anything other than
# "yes, exactly".
#
# Every check below aborts before a score.json could exist. There is no
# degraded mode: the alternative to a verified staged asset is a non-zero exit,
# never a substituted, defaulted, or re-fetched one.
#
# THE STAGING CONVENTION IS THE EXISTING ONE, not a new one. The golden path
# comes from the runner process environment under the name
# tools/qwen38-measure-and-score.sh already reads:
#
#   MLXFAST_QWEN38_GOLDEN_DIR          directory holding the staged track goldens
#                                      (the live golden this leg scores over plus
#                                      the rotation set). Required: nothing here
#                                      is in the checkout.
#   MLXFAST_BASELINE_WORKSPACE         the built REFERENCE tree the serial-control
#                                      leg runs on (section 8)
#   MLXFAST_BASELINE_CALIBRATION       this box's calibration file, the health
#                                      band for that leg (section 8b)
#
# On a self-hosted runner they are set in the runner service environment by
# whoever stages the box; a `run:` step inherits them. Nothing here reads a
# GitHub secret, and nothing here reaches the network.
#
# PAIRED, WITH A PER-BOX BASELINE. Each job measures one pair per live golden
# (two for this track), on the SAME box in the SAME job: a SERIAL CONTROL leg on
# the organizer-staged reference tree, and the CANDIDATE leg on this checkout at
# the organizer-fixed draft depth. Each pair gives a composite score, and the run
# score is the mean of the pair scores. NO PAIR IS
# STORED ANYWHERE -- not in a benchd constant, not in the fixture, not in a
# golden -- so this preflight verifies the staged pool goldens (and the live
# golden among them), the fixture arm state, the pinned benchd against its own
# manifest, the serve identity, AND the two halves the paired path adds: the
# reference workspace is the pinned reference commit and is built, and this box
# has a calibration file that describes THIS box against THAT commit.
#
# THE CALIBRATION IS A HEALTH BAND, NEVER A DENOMINATOR. benchd divides by the
# measured control leg. The band only says whether that leg landed where this box
# has landed before; outside it, the run dies and seals nothing.
#
# WHAT THE PINS ARE. fixtures/qwen3_8_27b_tf_cuda_v1_track.json is trusted-side (not an
# editable path), and its timed_prompt_pool[] carries {r2_path, sha256, bytes}
# per golden plus hidden_correctness_golden's {sha256, bytes}. A staged file is
# accepted only when its byte count AND its sha256 equal the contract's -- byte
# count first, because a truncated stage is the common failure and naming it
# precisely is worth one `wc -c` (the same order tools/fetch-goldens.sh and
# tools/fetch-benchd.sh verify in).
#
# Usage:  tools/ranked-box-preflight.sh
# Exit:   0 every check passed
#         1 a check failed (message on stderr names which)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONTRACT="${REPO_ROOT}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json"

fail() {
  echo "ranked-box-preflight: REFUSING -- $*" >&2
  exit 1
}

ok() {
  echo "ok    $*"
}

# --- 0. tools ---------------------------------------------------------------
# jq is already a hard requirement of tools/qwen38-measure-and-score.sh
# (it reads the fixture pins and the trackId), so a box that cannot run this
# cannot finish a ranked run either.
command -v jq >/dev/null 2>&1 || fail "jq is required to read the track contract"
command -v shasum >/dev/null 2>&1 || fail "shasum is required to verify staged assets"
[[ -r "${CONTRACT}" ]] || fail "cannot read the track contract at ${CONTRACT}"
jq -e . >/dev/null 2>&1 < "${CONTRACT}" || fail "the track contract is not valid JSON: ${CONTRACT}"
ok "track contract readable and parses: fixtures/qwen3_8_27b_tf_cuda_v1_track.json"

# --- 1. the job holds no credential -----------------------------------------
# The workflow references no secret (tools/ci-workflow-egress-scan.sh is the
# static half of that). This is the runtime half: a credential reaching the job
# through the RUNNER's environment would defeat the same invariant without ever
# appearing in the workflow file. R2 keys and a signer are what would let this
# job pull hidden material itself instead of measuring the staged bundle.
for var in R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY MLXFAST_QWEN38_R2_DOWNLOADER BENCHD_DIST_TOKEN; do
  eval "value=\${${var}:-}"
  # shellcheck disable=SC2154  # assigned by the eval above
  [[ -z "${value}" ]] || fail "${var} is set in the ranked job's environment; this job must hold no credential (boxes get staged bundles, never keys)"
done
ok "no R2 credential, signer, or dist token in the job environment"

# --- 2. no measurement-weakening override -----------------------------------
# Each of these is a real, local-debugging-only switch in this tree. On a
# ranked box any of them would silently change what is measured or what is
# accepted, so their presence is a refusal rather than a warning.
#
#   BENCHD                            tools/qwen38-measure-and-score.sh honours a
#                                     caller-supplied benchd WITHOUT the
#                                     channel-manifest hash check (deliberate, for
#                                     benchd development) -- so on a ranked run
#                                     it is a way to measure against unpinned
#                                     scoring code.
#   MLXFAST_SKIP_WEIGHTS_DOWNLOAD /   setup.sh builds tools only and never
#   SKIP_MODEL_DOWNLOAD               obtains or verifies the checkpoint.
#   MLXFAST_LOCAL_COOL_GATE           disables the thermal gate.
#   MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT  publishes a timing estimate past a failed
#                                     public correctness gate.
#   SERVE_UP_LAUNCHER                 tools/serve-up.sh's test seam: `host` runs the
#                                     resident outside the pinned container, with
#                                     the host's own Python and network.
#   SANDBOX_ENGINE_PATH               test-only PATH override for the sandboxed engine shim
#   SANDBOX_WIPE_LIST /               test-only overrides of tools/sandbox-wipe.sh's
#   SANDBOX_WIPE_HOME                 directory list and the sandbox user's home
#   SERVE_UP_NO_CAP_DROP /            tools/serve-up.sh's escape hatches that drop
#   SERVE_UP_ALLOW_NEW_PRIVS          --cap-drop ALL / no-new-privileges from the
#                                     sandboxed resident container
#                                     (see docs/sandbox-design.md, "Operator setup").
#   RESIDENT_* / TENSORFOLD_* / HF_* / the engine configuration. The resident takes
#   TRANSFORMERS_*                    its configuration from the fixture's
#                                     engine_env, but an inherited TENSORFOLD_* or HF_*
#                                     value changes what TensorFold does (kernels,
#                                     caches, downloads) while every seal still
#                                     reads normal, so any of them preset in the
#                                     job is refused.
#   DS4_RESIDENT_SOCKET /             the resident each leg's worker connects to.
#   BENCH_WORKER_RESIDENT_SOCKET      benchd boots each leg's resident itself and
#                                     injects that leg's socket into that leg's
#                                     worker spawns, so a socket already in the
#                                     environment is a THIRD resident neither leg
#                                     booted -- and every worker that inherited
#                                     it would measure whatever is on the other
#                                     end, on either leg, with a normal-looking
#                                     seal. benchd refuses one in its own
#                                     environment; this refuses it before the
#                                     box is committed to anything.
#
# MLXFAST_GPU_TEMP_CMD is NOT refused: it is benchd's documented, first-class
# override for the temperature reader, and on a Linux/CUDA box the native reader
# below (nvidia-smi) is what benchd's cool gate reads anyway. It stays an
# OPTIONAL override -- never required, never set here for the job -- so a box
# needs no environment variable at all to get a temperature reading.
#
# The loop below refuses a value that is PRESET in the job environment, before
# this preflight runs. The workflow's engine-build-cache step sets
# MLXFAST_SKIP_ENGINE_BUILD=1 LATER in the same job, after this preflight, and
# only on a verified cache hit, so that write and this refusal are consistent.
for var in BENCHD MLXFAST_SKIP_WEIGHTS_DOWNLOAD SKIP_MODEL_DOWNLOAD MLXFAST_LOCAL_COOL_GATE MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT MLXFAST_SKIP_ENGINE_BUILD MLXFAST_SKIP_WEIGHTS_SHA256 SERVE_UP_LAUNCHER SERVE_UP_NO_CAP_DROP SERVE_UP_ALLOW_NEW_PRIVS SANDBOX_ENGINE_PATH SANDBOX_WIPE_LIST SANDBOX_WIPE_HOME TENSORFOLD_WORKER_SOURCE ENGINE_EOS_IDS DS4_RESIDENT_SOCKET BENCH_WORKER_RESIDENT_SOCKET; do
  eval "value=\${${var}:-}"
  [[ -z "${value}" ]] || fail "${var} is set in the ranked job's environment; it weakens or bypasses what the ranked run measures"
done
engine_overrides="$(env | cut -d= -f1 | grep -E '^(RESIDENT_|TENSORFOLD_|HF_|TRANSFORMERS_|TORCHINDUCTOR_|TRITON_)' | tr '\n' ' ' || true)"
[[ -z "${engine_overrides}" ]] \
  || fail "engine configuration preset in the ranked job's environment: ${engine_overrides}; the fixture's engine_env is the only engine configuration"
ok "no measurement-weakening override in the job environment"

# --- 2b. the GPU temperature reader resolves --------------------------------
# The participant contract requires thermal control before calibration, and a
# missing reader REFUSES rather than silently self-disabling.
#
# WHY IT MUST BE LOUD HERE. The pinned benchd's own cool gate SKIPS with a
# warning when no reader resolves -- it returns GateState::SkippedNoReader
# rather than failing. That is correct for a participant's laptop and
# catastrophic for a ranked run: every timed leg would proceed ungated and the
# seal would look normal. Refusing here makes that path unreachable on this box.
#
# THE READER IS NATIVE TO THE PLATFORM, not an environment workaround. This box
# is Linux/CUDA (GB10), so the reader is nvidia-smi, which every CUDA box has --
# no environment variable is needed to get a reading. MLXFAST_GPU_TEMP_CMD is an
# OPTIONAL override, honoured first (benchd honours it first too); it is never
# required and never set here for the job.
GPU_TEMP_SOURCE=""
if [[ -n "${MLXFAST_GPU_TEMP_CMD:-}" ]]; then
  GPU_TEMP_SOURCE="MLXFAST_GPU_TEMP_CMD override"
  read_gpu_temp() { eval "${MLXFAST_GPU_TEMP_CMD}" 2>/dev/null | head -1 | tr -d '[:space:]'; }
else
  [[ "$(uname -s)" == "Linux" ]] || fail "this track's ranked boxes are Linux/CUDA (DGX Spark); refusing on $(uname -s)"
  command -v nvidia-smi >/dev/null 2>&1 \
    || fail "nvidia-smi not found: this Linux/CUDA box has no GPU temperature reader, so benchd's cool-down gate would silently skip on every timed leg. Install the NVIDIA driver utilities (nvidia-smi) or set MLXFAST_GPU_TEMP_CMD; this run measures nothing without thermal control"
  GPU_TEMP_SOURCE="nvidia-smi"
  read_gpu_temp() {
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]'
  }
fi
ok "GPU temperature reader resolved: ${GPU_TEMP_SOURCE}"

# --- 2c. the reader gives a plausible reading -------------------------------
# A sensor that reads at or below the 5C implausibility floor is broken or
# frozen, and a broken sensor passes every cool gate on an arbitrarily hot GPU.
first_temp="$(read_gpu_temp || true)"
[[ -n "${first_temp}" ]] \
  || fail "the temperature reader (${GPU_TEMP_SOURCE}) produced no reading; a reader that cannot be read is a missing reader"
[[ "$(jq -n --argjson t "${first_temp}" '$t > 5')" == "true" ]] \
  || fail "GPU temperature reads ${first_temp}C, at or below the 5C implausibility floor; the sensor is broken or frozen, and a broken sensor passes every cool gate"

# nvidia-smi reports an INTEGER Celsius that can legitimately sit on one value
# across a long idle stretch, so no "identical reading = stuck sensor" check is
# run; the plausibility floor above is the check.
ok "temperature reader is plausible (${GPU_TEMP_SOURCE}: ${first_temp}C)"

# --- 3. the timed pool is armed ---------------------------------------------
# "Armed" is a property of the CONTRACT, checked before anything on disk is
# looked at: a sentinel entry has no digest to verify a staged file against, so
# a staged file would be accepted on its name alone.
#
# THIS TRACK'S SENTINEL IS SPELLED IN FULL. The refusal matches the substring,
# so an unarmed pool of ANY spelling refuses, and names this track's sentinel.
TRACK_SENTINEL="QWEN38-TENSORFOLD-CUDA-PENDING-ORGANIZER"

if grep -q 'PENDING-ORGANIZER' "${CONTRACT}"; then
  fail "the track contract still carries ${TRACK_SENTINEL} sentinels; the timed pool is unarmed and nothing can be pin-verified against it"
fi

# The pool must be non-empty, so there IS a live golden to pin-verify against;
# every entry's pin is then checked below, and every live golden the fixture
# names is asserted pinned and staged in section 4b.
pool_count="$(jq -r '.timed_prompt_pool | length' "${CONTRACT}")"
[[ "${pool_count}" =~ ^[1-9][0-9]*$ ]] || fail "timed_prompt_pool is empty; there is no live golden to pin-verify against"

# A pin is {sha256, bytes} together; neither half alone is one. An entry that
# fails this is unarmed no matter what it is called.
malformed="$(jq -r '
  .timed_prompt_pool
  | to_entries
  | map(select(
      (.value.r2_path | type != "string" or length == 0)
      or (.value.sha256 | type != "string" or test("^[0-9a-f]{64}$") | not)
      or (.value.bytes | type != "number" or . <= 0)
    ))
  | map("timed_prompt_pool[" + (.key | tostring) + "]")
  | join(", ")
' "${CONTRACT}")"
[[ -z "${malformed}" ]] || fail "unarmed or malformed pool pin(s): ${malformed}"

hidden_sha="$(jq -r '.hidden_correctness_golden.sha256 // ""' "${CONTRACT}")"
hidden_bytes="$(jq -r '.hidden_correctness_golden.bytes // 0' "${CONTRACT}")"
printf '%s' "${hidden_sha}" | grep -Eq '^[0-9a-f]{64}$' \
  || fail "hidden_correctness_golden.sha256 is not a 64-hex digest; the token-fidelity oracle is unarmed"
printf '%s' "${hidden_bytes}" | grep -Eq '^[1-9][0-9]*$' \
  || fail "hidden_correctness_golden.bytes is not a positive integer"
ok "timed pool armed: ${pool_count} pinned pool golden(s) + a pinned hidden correctness golden"

# --- 4. the staged tapes match the pins -------------------------------------
# THE TAPES ARE NEVER IN THE CHECKOUT. They are organizer material published in
# R2 at the contract's r2_path keys, and the ranked box stages them out of band
# into the directory its runner service exports as MLXFAST_QWEN38_GOLDEN_DIR.
# There is no in-repo default to fall back on, so an unset name is a refusal
# rather than a quiet substitution. Whatever directory is named, every golden is
# verified byte-then-sha against the contract pins below.
GOLDEN_DIR="${MLXFAST_QWEN38_GOLDEN_DIR:-}"
[[ -n "${GOLDEN_DIR}" ]] \
  || fail "MLXFAST_QWEN38_GOLDEN_DIR is unset; the track goldens are staged onto the box out of band and this job holds no credential to fetch them"
[[ -d "${GOLDEN_DIR}" ]] \
  || fail "MLXFAST_QWEN38_GOLDEN_DIR does not exist or is not a directory: ${GOLDEN_DIR}"

verify_pin() {
  # verify_pin <path> <want_sha256> <want_bytes> <label>
  local path="$1" want_sha="$2" want_bytes="$3" label="$4" got_bytes got_sha
  [[ -f "${path}" ]] || fail "${label}: staged file is missing: ${path}"
  got_bytes="$(wc -c < "${path}" | tr -d '[:space:]')"
  [[ "${got_bytes}" == "${want_bytes}" ]] \
    || fail "${label}: byte-count mismatch (staged ${got_bytes}, pinned ${want_bytes}): ${path}"
  got_sha="$(shasum -a 256 "${path}" | awk '{print $1}')"
  [[ "${got_sha}" == "${want_sha}" ]] \
    || fail "${label}: sha256 mismatch (staged ${got_sha}, pinned ${want_sha}): ${path}"
}

expected_list=""
while IFS='	' read -r r2_path want_sha want_bytes; do
  [[ -n "${r2_path}" ]] || continue
  name="${r2_path##*/}"
  verify_pin "${GOLDEN_DIR}/${name}" "${want_sha}" "${want_bytes}" "timed-pool tape ${name}"
  expected_list="${expected_list}${name}
"
done <<EOF
$(jq -r '.timed_prompt_pool[] | [.r2_path, .sha256, (.bytes | tostring)] | @tsv' "${CONTRACT}")
EOF
ok "all ${pool_count} staged pool goldens match their contract pins (bytes then sha256)"

# The warm-up golden is a separate prompt. It is never scored, but it is read
# by both paired legs before their timed prompt. Pin and stage it separately.
warmup_path="$(jq -r '.warmup_golden.r2_path // empty' "${CONTRACT}")"
warmup_sha="$(jq -r '.warmup_golden.sha256 // empty' "${CONTRACT}")"
warmup_bytes="$(jq -r '.warmup_golden.bytes // 0' "${CONTRACT}")"
warmup_name="${warmup_path##*/}"
[[ -n "${warmup_path}" && "${warmup_name}" == *.golden.json ]] \
  || fail "warmup_golden.r2_path is missing or malformed"
printf '%s' "${warmup_sha}" | grep -Eq '^[0-9a-f]{64}$' \
  || fail "warmup_golden.sha256 is not a 64-hex digest"
printf '%s' "${warmup_bytes}" | grep -Eq '^[1-9][0-9]*$' \
  || fail "warmup_golden.bytes is not a positive integer"
verify_pin "${GOLDEN_DIR}/${warmup_name}" "${warmup_sha}" "${warmup_bytes}" \
  "warm-up golden ${warmup_name}"
expected_list="${expected_list}${warmup_name}
"
ok "the staged warm-up golden matches its separate contract pin"

# PER-DEPTH TIMED ORACLES. speculative_oracles.<prompt> maps DFlash depth 1
# to that prompt's own timed-oracle golden.
# {r2_path, sha256, bytes}; tools/qwen38-measure-and-score.sh selects one by
# the drawn prompt and the declared depth.
# Each must be pin-verified here AND allowed to exist in the staging directory
# (the "only pinned *.json" check below refuses any name absent from
# expected_list). A depth may REUSE another depth's file when their trajectories
# are identical, so a name is appended to expected_list at most once. An absent
# map means a serial-only track and this section verifies nothing.
spec_malformed="$(jq -r '
  [(.speculative_oracles // {}) | to_entries[] | .key as $p | .value | to_entries[]
    | select(
      (.value.r2_path | type != "string" or length == 0)
      or (.value.sha256 | type != "string" or test("^[0-9a-f]{64}$") | not)
      or (.value.bytes | type != "number" or . <= 0)
    )
    | "speculative_oracles[" + $p + "][" + .key + "]"]
  | join(", ")
' "${CONTRACT}")"
[[ -z "${spec_malformed}" ]] || fail "unarmed or malformed per-depth oracle pin(s): ${spec_malformed}"

spec_count=0
while IFS='	' read -r r2_path want_sha want_bytes; do
  [[ -n "${r2_path}" ]] || continue
  name="${r2_path##*/}"
  verify_pin "${GOLDEN_DIR}/${name}" "${want_sha}" "${want_bytes}" "per-depth oracle ${name}"
  # A here-string, not printf | grep -q: under pipefail, grep -q exits on the
  # first match, printf can then die of SIGPIPE, and the test reads as "absent".
  if ! grep -Fxq -- "${name}" <<<"${expected_list}"; then
    expected_list="${expected_list}${name}
"
  fi
  spec_count=$((spec_count + 1))
done <<EOF
$(jq -r '(.speculative_oracles // {}) | to_entries[] | .value | to_entries[] | [.value.r2_path, .value.sha256, (.value.bytes | tostring)] | @tsv' "${CONTRACT}")
EOF
[[ "${spec_count}" -eq 0 ]] || ok "all ${spec_count} per-depth oracle pin(s) match their staged golden(s)"

# The staging directory must hold ONLY pinned goldens. The scored run reads just
# the live golden -- every leg of every pair scores over that one prompt -- but
# an unpinned *.json staged where the pool lives is still an unattributed
# golden -- a mis-staged or leftover file that has no contract pin behind it --
# so it is a refusal rather than a warning: this directory carries pinned
# material only.
#
# ONE EXCEPTION: a refresh generation staged AHEAD of its pin
# (tools/pool-refresh.sh, docs/pool-refresh-and-final-rescore.md). Step 3 stages
# g<N>-<n>[.dflash<d>].golden.json on every box before step 4 pushes the fixture
# that pins them, and jobs dispatched at the older commit must still pass
# meanwhile. Only that exact name shape is tolerated; the held-out `final-*`
# goldens are NOT (one on a box before the close is exactly what must fail loud).
unexpected=""
for staged in "${GOLDEN_DIR}"/*.json; do
  [[ -e "${staged}" ]] || continue
  staged_name="${staged##*/}"
  [[ "${staged_name}" =~ ^g[0-9]+-[0-9]+(\.dflash[0-9]+)?\.golden\.json$ ]] && continue
  if ! grep -Fxq -- "${staged_name}" <<<"${expected_list}"; then
    unexpected="${unexpected} ${staged_name}"
  fi
done
[[ -z "${unexpected}" ]] \
  || fail "MLXFAST_QWEN38_GOLDEN_DIR holds *.json file(s) that are not pinned in timed_prompt_pool:${unexpected} (an unpinned golden staged where the pool lives is unattributed material; this directory carries pinned goldens only)"
ok "no unpinned *.json in the staging directory"

# The hidden correctness oracle is pinned by digest only -- the contract gives
# it no r2_path -- and benchd resolves it itself. If the box names one
# through the existing MLXFAST_CORRECTNESS_GOLDEN_PATH convention, it must be
# the pinned bytes; if it names none, this asserts nothing about it rather than
# inventing a location.
if [[ -n "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" ]]; then
  verify_pin "${MLXFAST_CORRECTNESS_GOLDEN_PATH}" "${hidden_sha}" "${hidden_bytes}" "hidden correctness golden"
  ok "MLXFAST_CORRECTNESS_GOLDEN_PATH matches hidden_correctness_golden"
else
  ok "MLXFAST_CORRECTNESS_GOLDEN_PATH unset; benchd resolves the oracle from the contract"
fi

# --- 4b. every prompt the ranked run may score is pinned and staged --------
# A fixture that declares live_goldens scores EVERY listed prompt in every job.
# Otherwise the ranked run scores ONE prompt per job: the one the workflow
# draws from live_golden_rotation.pool (tools/draw-live-golden.sh), or the
# fixture's live_golden when it declares no rotation. Every prompt a job can
# measure must carry its serial golden (a timed_prompt_pool entry) and one
# per-depth oracle for every permitted draft depth, all pinned and staged. The
# loops above already pin-verified what is pinned; this refuses, by name, a
# prompt with either required tape missing, pre-GPU rather than in the one job
# that happens to measure it.
if [[ "$(jq -r 'has("live_goldens")' "${CONTRACT}")" == "true" ]]; then
  [[ "$(jq -r 'has("live_golden_rotation")' "${CONTRACT}")" != "true" ]] \
    || fail "the fixture declares both live_goldens and live_golden_rotation; a job either measures every live golden or draws one prompt"
  mapfile -t ROTATION_POOL < <(jq -r '.live_goldens // [] | .[]' "${CONTRACT}")
  [[ "${#ROTATION_POOL[@]}" -ge 1 ]] || fail "live_goldens is empty; there is nothing to score"
  LIVE_GOLDEN_NAME="${ROTATION_POOL[0]}"
elif ! LIVE_GOLDEN_NAME="$(jq -r '.live_golden // ""' "${CONTRACT}")" || [[ -z "${LIVE_GOLDEN_NAME}" ]]; then
  fail "the fixture declares neither live_goldens nor live_golden; there is no golden for the ranked run to score over"
elif [[ "$(jq -r 'has("live_golden_rotation")' "${CONTRACT}")" == "true" ]]; then
  rot_mode="$(jq -r '.live_golden_rotation.mode // ""' "${CONTRACT}")"
  [[ "${rot_mode}" == "per_job_random" ]] \
    || fail "live_golden_rotation.mode is '${rot_mode}'; only per_job_random is defined"
  mapfile -t ROTATION_POOL < <(jq -r '.live_golden_rotation.pool // [] | .[]' "${CONTRACT}")
  [[ "${#ROTATION_POOL[@]}" -ge 1 ]] || fail "live_golden_rotation.pool is empty; there is nothing to draw"
else
  ROTATION_POOL=("${LIVE_GOLDEN_NAME}")
fi
mapfile -t PERMITTED_DEPTHS < <(jq -r '.permitted_speculative_depths // [] | .[]' "${CONTRACT}")
for prompt in "${ROTATION_POOL[@]}"; do
  [[ "${prompt}" =~ ^[a-z0-9_-]+$ ]] || fail "rotation pool name '${prompt}' is not [a-z0-9_-]+"
  base="${prompt}.golden.json"
  jq -e --arg f "/${base}" 'any(.timed_prompt_pool[]; .r2_path | endswith($f))' "${CONTRACT}" >/dev/null \
    || fail "rotation pool prompt '${prompt}' names no timed_prompt_pool entry (looked for ${base}); its serial golden carries no pin"
  [[ -f "${GOLDEN_DIR}/${base}" ]] \
    || fail "rotation pool prompt '${prompt}': its serial golden ${base} is not staged in ${GOLDEN_DIR}"
  for d in "${PERMITTED_DEPTHS[@]}"; do
    oracle="$(jq -r --arg p "${prompt}" --arg k "dflash${d}" '.speculative_oracles[$p][$k].r2_path // ""' "${CONTRACT}")"
    [[ -n "${oracle}" ]] \
      || fail "rotation pool prompt '${prompt}' has no pinned DFlash oracle speculative_oracles[${prompt}][dflash${d}]"
    [[ "${oracle##*/}" == "${prompt}.dflash${d}.golden.json" ]] \
      || fail "speculative_oracles[${prompt}][dflash${d}] points at ${oracle##*/}, not ${prompt}.dflash${d}.golden.json"
    [[ -f "${GOLDEN_DIR}/${oracle##*/}" ]] \
      || fail "rotation pool prompt '${prompt}': its per-depth oracle ${oracle##*/} is not staged in ${GOLDEN_DIR}"
  done
done
ok "every rotation pool prompt (${ROTATION_POOL[*]}) has its serial golden and ${#PERMITTED_DEPTHS[@]} per-depth oracle(s) pinned and staged"

# --- 5. the fixture is armed for official scoring ---------------------------
# benchd refuses, pre-GPU, to seal an official artifact unless the fixture
# declares official_scoring_enabled: true (enforce_official_scoring_enabled).
# false AND absent both refuse -- an absent arm state is not an armed one. The
# measure wrapper's --preflight-only mirrors this; refusing here makes an
# unarmed ranked dispatch fail before setup rather than after the GPU window.
armed="$(jq -r '.official_scoring_enabled // false' "${CONTRACT}")"
[[ "${armed}" == "true" ]] \
  || fail "the track contract does not declare official_scoring_enabled: true (got '${armed}'); benchd refuses to seal an official score for an unarmed track, and so does this gate"
ok "fixture is armed for official scoring (official_scoring_enabled: true)"

# --- 6. the pinned benchd is present and matches its own manifest ---------
# The scored composite is SEALED by benchd, so an unattributable or mismatched
# benchd is a refusal here, before setup. This is the fail-fast, provenance-
# logging half of tools/fetch-benchd.sh's OFFLINE path (its step 1): the
# organizer-staged PAIR benchd-bin/{benchd, benchd.manifest.json} is placed
# on the box out of band (bundles-no-keys: this job holds no dist token and
# reaches no network -- section 1 refuses BENCHD_DIST_TOKEN), and fetch-benchd.sh
# re-verifies it authoritatively in the next workflow step (adding the channel
# branch and the target_triple / ELF-container checks). A binary with no manifest
# is unattributable and must never seal a score. BENCHD_BIN_DIR is resolved
# exactly as fetch-benchd.sh resolves it, so a box that stages the pair outside
# the workspace is checked at the same path the harness will read.
BENCHD_DIR="${BENCHD_BIN_DIR:-${REPO_ROOT}/benchd-bin}"
BENCHD_BIN="${BENCHD_DIR}/benchd"
BENCHD_MANIFEST="${BENCHD_DIR}/benchd.manifest.json"
[[ -f "${BENCHD_BIN}" ]] \
  || fail "the pinned benchd is not staged at ${BENCHD_BIN}; this job holds no dist token and does not fetch benchd over the network. Stage benchd and its manifest on the box (tools/fetch-benchd.sh)"
[[ -f "${BENCHD_MANIFEST}" ]] \
  || fail "benchd.manifest.json is missing beside ${BENCHD_BIN}; a binary with no manifest is unattributable and will not seal a score"
jq -e . >/dev/null 2>&1 < "${BENCHD_MANIFEST}" \
  || fail "the benchd manifest is not valid JSON: ${BENCHD_MANIFEST}"
benchd_sha="$(jq -r '.sha256 // ""' "${BENCHD_MANIFEST}")"
benchd_bytes="$(jq -r '.bytes // 0' "${BENCHD_MANIFEST}")"
benchd_commit="$(jq -r '.source_commit // ""' "${BENCHD_MANIFEST}")"
printf '%s' "${benchd_sha}" | grep -Eq '^[0-9a-f]{64}$' \
  || fail "the benchd manifest sha256 is not a 64-hex digest ('${benchd_sha}'); the harness identity is unverifiable"
printf '%s' "${benchd_bytes}" | grep -Eq '^[1-9][0-9]*$' \
  || fail "the benchd manifest bytes is not a positive integer: '${benchd_bytes}'"
verify_pin "${BENCHD_BIN}" "${benchd_sha}" "${benchd_bytes}" "pinned benchd"
[[ -n "${benchd_commit}" ]] \
  || echo "ranked-box-preflight: WARNING -- benchd manifest records no source_commit; provenance is unattributed" >&2
ok "pinned benchd matches its manifest (source_commit ${benchd_commit:-unrecorded}, sha256 ${benchd_sha})"

# --- 7. the serve spec matches the participant declaration (the ARM GATE) ----
# The candidate leg's serve spec IS part of the identity: a resident SERIAL
# serve and a scored DFlash serve are different runs. The serve spec is DERIVED
# from the participant declaration (mtp-head.manifest.json `spec`) by the single
# trusted source tools/spec-declaration.sh, which validates it fail-closed. This
# gate resolves the SAME derivation and:
#   * REFUSES an invalid declaration (the helper exits non-zero and names why);
#   * when the declaration ENABLES speculation, requires the fixture to be ARMED
#     for official scoring (official_scoring_enabled: true) -- a valid declaration
#     alone does not open the gate;
#   * REFUSES any box-preset SERVE_UP_SPECULATIVE. Under one leg a preset that
#     agreed with the declaration was harmless. Under the paired path it is not:
#     the value reaches the environment of BOTH legs' serve scripts, and the
#     serial-control leg is serial whatever the candidate declares. So the serve
#     spec comes from the declaration and from nowhere else.
#
# A disabled declaration derives 0 (SERIAL), while an absent, malformed or
# incomplete declaration is refused by tools/spec-declaration.sh. An enabled
# declaration opens this gate only when its spec is valid and the fixture is
# armed for official scoring.
if ! decl_spec="$("${REPO_ROOT}/tools/spec-declaration.sh" speculative)"; then
  fail "the speculative-decode declaration (mtp-head.manifest.json) is invalid; the derivation above names why. An undeclared or disabled declaration is valid and derives serial"
fi
decl_draft="$("${REPO_ROOT}/tools/spec-declaration.sh" draft-len)"
[[ -z "${SERVE_UP_SPECULATIVE:-}" ]] \
  || fail "a box pre-set SERVE_UP_SPECULATIVE=${SERVE_UP_SPECULATIVE}; on the paired path that value reaches BOTH legs' serve scripts and the serial-control leg must be serial whatever the candidate declares. The serve spec comes from the declaration only"
if [[ "${decl_spec}" == "1" ]]; then
  armed="$(jq -r '.official_scoring_enabled // false' "${CONTRACT}")"
  [[ "${armed}" == "true" ]] \
    || fail "the declaration enables DFlash depth ${decl_draft} but fixtures/qwen3_8_27b_tf_cuda_v1_track.json is not armed"
  ok "serve spec is the declared DFlash config (tokens=${decl_draft}) and the fixture is armed"
else
  ok "serve spec is the serial launch reference (SERVE_UP_SPECULATIVE=0; declaration absent or disabled)"
fi

# NATIVE TENSORFOLD PINS. The source stays participant-editable. The fixture
# pins the fork base and the native toolchain.
want_engine_pin="$(jq -r '.tensorfold_revision // empty' "${CONTRACT}")"
[[ "${want_engine_pin}" == "eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab" ]] \
  || fail "tensorfold_revision is not the track fork pin"
grep -qF '.version = "1.1.6"' "${REPO_ROOT}/Vendor/TensorFold/build.zig.zon" \
  || fail "Vendor/TensorFold is not version 1.1.6"
ok "TensorFold v1.1.6 fork pin ${want_engine_pin:0:12}"

command -v nvcc >/dev/null 2>&1 || fail "nvcc is not on PATH"
nvcc --version >/dev/null 2>&1 || fail "nvcc is present but does not run"
[[ "$(jq -r '.serve_configuration.toolchain.cuda_arch // empty' "${CONTRACT}")" == "sm_121" ]] \
  || fail "serve_configuration.toolchain.cuda_arch must be sm_121"
ok "CUDA compiler is available for sm_121"

want_driver_min="$(jq -r '.serve_configuration.toolchain.driver_min // empty' "${CONTRACT}")"
[[ -n "${want_driver_min}" ]] || fail "serve_configuration.toolchain.driver_min is unarmed"
have_driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')"
[[ -n "${have_driver}" ]] || fail "nvidia-smi reports no driver version"
if [[ "$(printf '%s\n%s\n' "${want_driver_min}" "${have_driver}" | sort -V | head -1)" != "${want_driver_min}" ]]; then
  fail "driver ${have_driver} is below the fixture floor ${want_driver_min}"
fi
ok "driver ${have_driver} (floor ${want_driver_min})"

want_owner="$(jq -r '.serve_configuration.weight_owner // empty' "${CONTRACT}")"
[[ "${want_owner}" == "tensorfold-resident" ]] \
  || fail "serve_configuration.weight_owner must be tensorfold-resident"
grep -qF 'args=(resident --weights' "${REPO_ROOT}/tools/serve-up.sh" \
  || fail "tools/serve-up.sh no longer starts the native resident with --weights"
ok "weight owner tensorfold-resident loads once per leg"

want_depths="$(jq -c '.serve_configuration.depths' "${CONTRACT}")"
[[ "${want_depths}" == '{"serial":{"draft_depth":0,"timed_oracle":"timed_prompt_pool[<live prompt>]"},"dflash1":{"draft_depth":1,"timed_oracle":"speculative_oracles[<live prompt>].dflash1"}}' ]] \
  || fail "serve_configuration.depths must pin serial and dflash1 only"
ok "serve configuration pins serial and DFlash depth 1"

# --- 8. the paired, per-box baseline ----------------------------------------
# PAIRED WITH A PER-BOX BASELINE. The ranked job measures TWO
# legs on THIS box, in THIS job, over the ONE live golden:
#
#   leg 1  the SERIAL control, on the organizer-staged REFERENCE tree
#          (MLXFAST_BASELINE_WORKSPACE), speculation off, always;
#   leg 2  the candidate, on this checkout, at its declared draft depth.
#
# The score is the ratio of the two legs. NOTHING stores the pair: not a
# constant, not the fixture, not a golden (the staged tapes above are checked for
# that at rest by tools/lint-benchmark-manifest.py). So the two things this
# section must establish are that the REFERENCE TREE IS THE RIGHT TREE and that
# THIS BOX HAS A HEALTH BAND for its leg 1.
#
# THE CALIBRATION IS A HEALTH BAND, NEVER A DENOMINATOR. It says what leg 1 has
# measured on this box before. If leg 1 lands outside the band the run dies and
# seals nothing, because a control leg that moved means the box moved -- and a
# ratio taken on a moved box is not this track's number. It is never divided by.
for var in MLXFAST_BASELINE_WORKSPACE MLXFAST_BASELINE_CALIBRATION; do
  eval "value=\${${var}:-}"
  [[ -n "${value}" ]] \
    || fail "${var} is not set in the ranked job's environment; the paired path measures its serial control on the staged reference tree against this box's band, and neither can be resolved without it"
done
command -v git >/dev/null 2>&1 || fail "git is required to verify the reference workspace HEAD"

BASELINE_WORKSPACE="${MLXFAST_BASELINE_WORKSPACE}"
BASELINE_CALIBRATION="${MLXFAST_BASELINE_CALIBRATION}"
[[ -d "${BASELINE_WORKSPACE}" ]] \
  || fail "MLXFAST_BASELINE_WORKSPACE=${BASELINE_WORKSPACE} is not a directory on this box; stage the reference tree with tools/stage-baseline-workspace.sh"

# The reference tree and the candidate tree are TWO trees. One tree cannot be
# both legs: leg 1 would then measure the submission and the score would be the
# candidate divided by itself.
baseline_real="$(cd "${BASELINE_WORKSPACE}" && pwd -P)"
[[ "${baseline_real}" != "${REPO_ROOT}" ]] \
  || fail "MLXFAST_BASELINE_WORKSPACE points at this checkout (${REPO_ROOT}); the serial control must run on the reference tree, not on the submission"

# THE REFERENCE COMMIT IS THE FIXTURE'S. baseline_reference_commit names the
# promoted baseline every box's leg 1 runs, so one board compares one control.
WANT_REF_COMMIT="$(jq -r '.baseline_reference_commit // empty' "${CONTRACT}")"
[[ "${WANT_REF_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "fixtures/qwen3_8_27b_tf_cuda_v1_track.json carries no 40-hex baseline_reference_commit (got '${WANT_REF_COMMIT}'); the serial-control leg has no reference tree to run on"
have_ref_commit="$(git -C "${BASELINE_WORKSPACE}" rev-parse HEAD 2>/dev/null || true)"
[[ "${have_ref_commit}" == "${WANT_REF_COMMIT}" ]] \
  || fail "the reference workspace is at ${have_ref_commit:-no git HEAD} but the fixture pins ${WANT_REF_COMMIT}; re-stage it with tools/stage-baseline-workspace.sh"
git -C "${BASELINE_WORKSPACE}" diff --quiet HEAD 2>/dev/null \
  || fail "the reference workspace has uncommitted changes; leg 1 must run the pinned reference tree, not an edited one"
ok "reference workspace at ${WANT_REF_COMMIT:0:12}, clean: ${BASELINE_WORKSPACE}"

# THE REFERENCE TREE MUST BE BUILT. A staged-but-unbuilt reference tree fails
# only when serve-up.sh boots it -- after the candidate has already paid for the
# build. These are exactly the outputs tools/engine/build.sh stages.
rel=.build/release/bench-worker
[[ -f "${BASELINE_WORKSPACE}/${rel}" ]] \
  || fail "the reference workspace has no ${rel}; its engine build is not staged (run tools/engine/build.sh in ${BASELINE_WORKSPACE}, or re-stage with tools/stage-baseline-workspace.sh)"
[[ -x "${BASELINE_WORKSPACE}/tools/serve-up.sh" ]] \
  || fail "the reference workspace has no executable tools/serve-up.sh; leg 1 boots the REFERENCE tree's own serve script, not this checkout's"
# WHICH serve-up.sh IS IN THERE IS SETTLED BY THE COMMIT PIN ABOVE, not by a
# grep here: the workspace is at baseline_reference_commit and is clean, so its
# serve script is exactly that commit's. What has to be true of that COMMIT --
# that it speaks the per-leg --boot/--stop verbs benchd boots the control leg
# with -- is a property of the fixture's choice of reference, and
# tools/lint-benchmark-manifest.py checks it at rest, where it can be fixed.
ok "reference workspace carries its staged native worker"

# --- 8b. this box's calibration file ----------------------------------------
[[ -f "${BASELINE_CALIBRATION}" ]] \
  || fail "MLXFAST_BASELINE_CALIBRATION=${BASELINE_CALIBRATION} is not a file; calibrate this box with tools/calibrate-box.sh"
jq -e . >/dev/null 2>&1 < "${BASELINE_CALIBRATION}" \
  || fail "the baseline calibration is not valid JSON: ${BASELINE_CALIBRATION}"

method_payload="$(jq -ceS '
  def positive_integer: type == "number" and . > 0 and . == floor;
  if ((.official_prefill_timed_runs | positive_integer) and
      (.official_prefill_warmup_runs | positive_integer) and
      (.warmup_golden.r2_path | type == "string" and length > 0) and
      (.decode_golden_offset | positive_integer) and
      (.decode_window_includes_seed | type == "boolean") and
      (.seed_tokens | positive_integer) and
      (.benchmark_decode_steps | positive_integer))
  then {
    official_prefill_timed_runs,
    official_prefill_warmup_runs,
    warmup_golden: {r2_path: .warmup_golden.r2_path, sha256: .warmup_golden.sha256, bytes: .warmup_golden.bytes},
    timed_prompt_pool: [.timed_prompt_pool[] | {sha256, bytes}],
    decode_golden_offset,
    decode_window_includes_seed,
    seed_tokens,
    benchmark_decode_steps
  }
  else error("missing or malformed timing-method field")
  end
' "${CONTRACT}" 2>/dev/null)" \
  || fail "the track contract has a missing or malformed calibration timing-method field"
want_method="$(printf '%s' "${method_payload}" | shasum -a 256 | awk '{print $1}')"
cal_method="$(jq -r '.method // empty' "${BASELINE_CALIBRATION}")"
[[ "${cal_method}" == "${want_method}" ]] \
  || fail "the baseline calibration method is '${cal_method:-missing}' but this fixture requires '${want_method}'; re-calibrate this box after any timing-method change"
ok "baseline calibration method matches the timing-relevant fixture fields"

# Version 3 holds one entry for each prompt in prompts[]. For this track its
# decode mean is (seed-prefill time + decode time) / N because the fixture sets
# decode_window_includes_seed to true. The ranked paired legs use the same rule.
cal_version="$(jq -r '.version // empty' "${BASELINE_CALIBRATION}")"
[[ "${cal_version}" != "1" && "${cal_version}" != "2" ]] \
  || fail "the baseline calibration declares version '${cal_version}': this preflight reads version 3 only, whose decode timing follows the fixture. Recalibrate this box with tools/calibrate-box.sh and the pinned benchd, which writes version 3"
[[ "${cal_version}" == "3" ]] \
  || fail "the baseline calibration declares version '${cal_version}'; this preflight reads version 3 only"

# THE CALIBRATION PROMPTS. Under live_goldens every listed prompt carries its
# own band, and benchd gates each control leg with the band of the prompt it
# measured. Otherwise the band is captured on the fixture's calibration_prompt
# (default: live_golden) and gates the serial-control leg on whichever pool
# prompt the job drew. Version 3 stores per-prompt bands in prompts[]; flatten
# one entry and the file header into one object before the per-prompt checks.
if [[ "$(jq -r 'has("live_goldens")' "${CONTRACT}")" == "true" ]]; then
  mapfile -t CALIBRATION_PROMPTS < <(jq -r '.live_goldens[]' "${CONTRACT}")
else
  CALIBRATION_PROMPTS=("$(jq -r '.calibration_prompt // .live_golden // ""' "${CONTRACT}")")
fi
cal_view_tmp=""

TRACK_ID="$(jq -r '.track_id // empty' "${CONTRACT}")"
cal_track="$(jq -r '.track_id // empty' "${BASELINE_CALIBRATION}")"
[[ "${cal_track}" == "${TRACK_ID}" ]] \
  || fail "the baseline calibration names track '${cal_track}' but this track is '${TRACK_ID}'; a band from another track does not describe this leg"

# THE BOX NAMES ITSELF. A band is a property of ONE machine, so it is attributed
# to the runner it was captured on. An unnamed runner is refused rather than
# waved through: a band that cannot be attributed is a band from anywhere.
cal_box="$(jq -r '.box // empty' "${BASELINE_CALIBRATION}")"
[[ -n "${cal_box}" ]] || fail "the baseline calibration names no box; a per-box band must say which box"
[[ -n "${RUNNER_NAME:-}" ]] \
  || fail "the baseline calibration names box '${cal_box}' but this job has no RUNNER_NAME; a per-box band cannot be attributed to a runner that does not identify itself"
[[ "${cal_box}" == "${RUNNER_NAME}" ]] \
  || fail "the baseline calibration was captured on box '${cal_box}' but this job runs on '${RUNNER_NAME}'; another box's band does not describe this box"

cal_ref="$(jq -r '.reference_commit // empty' "${BASELINE_CALIBRATION}")"
[[ "${cal_ref}" == "${WANT_REF_COMMIT}" ]] \
  || fail "the baseline calibration was captured against reference commit '${cal_ref}' but the fixture pins ${WANT_REF_COMMIT}; re-calibrate this box against the promoted reference"

check_band() { # check_band PROMPT -- the band checks for one calibration prompt
CALIBRATION_PROMPT="$1"
CAL_FOR_CHECKS="${BASELINE_CALIBRATION}"
if [[ "${cal_version}" == "3" ]]; then
  cal_view_tmp="$(mktemp "${TMPDIR:-/tmp}/baseline-cal.XXXXXX")"
  trap 'rm -f "${cal_view_tmp:-}"' EXIT  # removed on every refusal path too
  cal_view_err="$(mktemp "${TMPDIR:-/tmp}/baseline-cal-err.XXXXXX")"
  if ! "${REPO_ROOT}/tools/baseline-calibration-v1-view.sh" \
      "${BASELINE_CALIBRATION}" "${CALIBRATION_PROMPT}" > "${cal_view_tmp}" 2> "${cal_view_err}"; then
    cal_view_msg="$(tr -d '\n' < "${cal_view_err}")"
    rm -f "${cal_view_tmp}" "${cal_view_err}"
    case "${cal_view_msg}" in
      *"no prompts[] entry"*)
        fail "the baseline calibration has no prompts[] entry for prompt '${CALIBRATION_PROMPT}'"
        ;;
      *"more than one prompts[] entry"*)
        fail "the baseline calibration has more than one prompts[] entry for prompt '${CALIBRATION_PROMPT}'"
        ;;
      *)
        fail "the baseline calibration could not be read: ${cal_view_msg}"
        ;;
    esac
  fi
  rm -f "${cal_view_err}"
  CAL_FOR_CHECKS="${cal_view_tmp}"
fi

LIVE_GOLDEN_NAME_CAL="$(jq -r '.prompt // empty' "${CAL_FOR_CHECKS}")"
[[ "${LIVE_GOLDEN_NAME_CAL}" == "${CALIBRATION_PROMPT}" ]] \
  || fail "the baseline calibration was captured on prompt '${LIVE_GOLDEN_NAME_CAL}' but the fixture's calibration_prompt is '${CALIBRATION_PROMPT}'; the band must be captured on the calibration prompt"

cal_passes="$(jq -r '.passes // empty' "${CAL_FOR_CHECKS}")"
[[ "${cal_passes}" =~ ^[2-9][0-9]*$ ]] \
  || fail "the baseline calibration records ${cal_passes:-no} pass(es); a band needs at least two, because the CV is undefined below two"

# THE NUMBERS. A mean that is zero, negative or absent is not a band, and a CV
# above the contract's calibration_max_cv_percent means the box was not stable
# when it was calibrated -- the same gate the calibrator applies, re-applied
# here so a hand-edited file cannot widen it.
for key in prefill_seconds_per_token_mean decode_seconds_per_token_mean; do
  value="$(jq -r --arg k "${key}" '.[$k] // empty' "${CAL_FOR_CHECKS}")"
  [[ -n "${value}" ]] || fail "the baseline calibration carries no ${key}; there is no band centre on that axis"
  [[ "$(jq -n --argjson v "${value}" '$v > 0 and $v < 60')" == "true" ]] \
    || fail "the baseline calibration's ${key} is ${value} s/tok, which is not a plausible per-token time; the band centre is unusable"
done
# The stability gate is the contract's calibration_max_cv_percent. Read the
# same value here as benchd calibrate-baseline. Absent, the historical 2 % holds.
cv_max_percent="$(jq -r 'if has("calibration_max_cv_percent") and .calibration_max_cv_percent != null then .calibration_max_cv_percent else 2 end' "${CONTRACT}")"
[[ "$(jq -r 'if has("calibration_max_cv_percent") and .calibration_max_cv_percent != null then .calibration_max_cv_percent else 2 end | type == "number" and . > 0 and . <= 100' "${CONTRACT}")" == "true" ]] \
  || fail "the track contract's calibration_max_cv_percent is '${cv_max_percent}', not a usable percentage"
cv_max="$(jq -n --argjson p "${cv_max_percent}" '$p / 100')"
for key in prefill_cv decode_cv; do
  value="$(jq -r --arg k "${key}" '.[$k] // empty' "${CAL_FOR_CHECKS}")"
  [[ -n "${value}" ]] || fail "the baseline calibration carries no ${key}; there is no evidence the mean describes this box"
  [[ "$(jq -n --argjson v "${value}" --argjson m "${cv_max}" '$v >= 0 and $v <= $m')" == "true" ]] \
    || fail "the baseline calibration's ${key} is ${value}, above the ${cv_max_percent} % stability gate (the track contract's calibration_max_cv_percent); this box was not stable when it was calibrated, so re-calibrate it"
done

# THE BANDS MUST BE ABLE TO FAIL. A band that does not bracket 1 rejects a
# healthy box; a band wide enough to admit a box running at half speed accepts a
# broken one. Both are refused by name.
for axis in prefill decode; do
  low="$(jq -r --arg k "${axis}_band_low" '.[$k] // empty' "${CAL_FOR_CHECKS}")"
  high="$(jq -r --arg k "${axis}_band_high" '.[$k] // empty' "${CAL_FOR_CHECKS}")"
  [[ -n "${low}" && -n "${high}" ]] \
    || fail "the baseline calibration carries no ${axis}_band_low/${axis}_band_high; leg 1 would be gated by nothing on that axis"
  [[ "$(jq -n --argjson l "${low}" --argjson h "${high}" '$l > 0.5 and $l <= 1 and $h >= 1 and $h < 2 and $l < $h')" == "true" ]] \
    || fail "the ${axis} band [${low}, ${high}] is not a usable band: it must bracket 1 and stay inside (0.5, 2), or it either rejects a healthy box or admits one running at half speed"
done

[[ -z "${cal_view_tmp}" ]] || rm -f "${cal_view_tmp}"
cal_view_tmp=""
}
for prompt in "${CALIBRATION_PROMPTS[@]}"; do check_band "${prompt}"; done

cal_benchd="$(jq -r '.benchd_source_commit // empty' "${BASELINE_CALIBRATION}")"
[[ "${cal_benchd}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "the baseline calibration records benchd_source_commit '${cal_benchd}'; a band captured by an unattributable benchd cannot be traced to the code that measured it"

# TIMESTAMPS. The band must be YOUNGER than the reference commit it describes --
# a band captured before the reference tree existed describes a different engine
# -- and it may not be dated in the future, which is a clock fault on the box.
cal_at="$(jq -r '.captured_at // empty' "${BASELINE_CALIBRATION}")"
cal_epoch="$(jq -r '(.captured_at // "") | try fromdateiso8601 catch empty' "${BASELINE_CALIBRATION}")"
[[ -n "${cal_epoch}" ]] \
  || fail "the baseline calibration's captured_at ('${cal_at}') is not an RFC 3339 UTC timestamp; the band cannot be dated"
ref_epoch="$(git -C "${BASELINE_WORKSPACE}" show -s --format=%ct HEAD 2>/dev/null || true)"
[[ "${ref_epoch}" =~ ^[0-9]+$ ]] \
  || fail "cannot read the reference commit's date from ${BASELINE_WORKSPACE}; the band's age cannot be checked"
now_epoch="$(date -u +%s)"
(( cal_epoch >= ref_epoch )) \
  || fail "the baseline calibration is dated ${cal_at}, BEFORE reference commit ${WANT_REF_COMMIT:0:12}; a band captured before the reference tree existed describes a different engine"
(( cal_epoch <= now_epoch + 300 )) \
  || fail "the baseline calibration is dated ${cal_at}, in the future; the box clock is wrong and every age check here is meaningless"
cal_sha="$(shasum -a 256 "${BASELINE_CALIBRATION}" | awk '{print $1}')"
ok "box band for ${cal_box}: ${cal_passes} passes on ${CALIBRATION_PROMPTS[*]}, captured ${cal_at} against ${WANT_REF_COMMIT:0:12} (sha256 ${cal_sha:0:12})"

# --- 9. participant code has a sandbox to run in ----------------------------
# docs/sandbox-design.md: every participant-built execution point runs as the
# unprivileged qwen38sub, never as runner (docker, bench, the checkout). A box
# where that user, its sudo drop, its toolchain or its egress block is missing
# would run participant code with runner's reach, so it is refused. The check
# script is trusted (tools/ is not an editable path) and changes nothing.
if ! sandbox_out="$("${REPO_ROOT}/tools/box-runner/setup-sandbox-user.sh" --check 2>&1)"; then
  fail "the sandbox user is not set up on this box (tools/box-runner/setup-sandbox-user.sh --check): $(grep -E '^FAIL|^setup-sandbox-user' <<<"${sandbox_out}" | paste -sd ';' -); run 'sudo tools/box-runner/setup-sandbox-user.sh' once per box"
fi
ok "sandbox user ready: setup-sandbox-user.sh --check passed"

echo "ranked-box-preflight: all checks passed"

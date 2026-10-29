#!/usr/bin/env bash
# Build and stage the native TensorFold worker in the pinned CUDA container.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
FIXTURE="${ROOT}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json"
ZIG_REQUEST="${TENSORFOLD_ZIG:-zig}"
SANDBOX_USER="${SANDBOX_USER:-}"
CUDA_SM="${TENSORFOLD_CUDA_SM:-121}"
NVCC_BIN="${TENSORFOLD_NVCC:-/usr/local/cuda/bin/nvcc}"

die() { printf 'engine/build.sh: %s\n' "$*" >&2; exit 1; }
log() { printf 'engine/build.sh: %s\n' "$*"; }
file_owner() { stat -c %U -- "$1" 2>/dev/null || stat -f %Su -- "$1"; }

for tool in docker jq; do command -v "${tool}" >/dev/null 2>&1 || die "${tool} is required"; done
ZIG_BIN="$(command -v -- "${ZIG_REQUEST}" 2>/dev/null || true)"
[[ "${ZIG_BIN}" == /* ]] || die "Zig must resolve to an absolute executable path: <zig-toolchain>"
[[ -x "${ZIG_BIN}" ]] || die "Zig is not executable: <zig-toolchain>"
[[ "$("${ZIG_BIN}" version)" == "0.17.0" ]] || die "Zig 0.17.0 is required"
ZIG_TOOLCHAIN_DIR="$(cd -- "$(dirname -- "${ZIG_BIN}")" && pwd -P)"
[[ -d "${ZIG_TOOLCHAIN_DIR}/lib" ]] || die "Zig library directory is missing: <zig-toolchain>/lib"
[[ -x "${NVCC_BIN}" ]] || die "nvcc is not executable: ${NVCC_BIN}"

IMAGE="$(jq -r '.serve_configuration.container_image // empty' "${FIXTURE}")"
DIGEST="$(jq -r '.serve_configuration.container_digest // empty' "${FIXTURE}")"
[[ -n "${IMAGE}" && "${DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || die "the fixture pins no CUDA container image and digest"
IMAGE_REF="${IMAGE%%:*}@${DIGEST}"
docker image inspect "${IMAGE_REF}" >/dev/null 2>&1 \
  || die "the pinned CUDA image is not present: ${IMAGE_REF}"

sandbox_quiet() {
  sudo -n -u "${SANDBOX_USER}" -- kill -KILL -1 >/dev/null 2>&1 || true
  local rc=0
  pgrep -u "${SANDBOX_USER}" >/dev/null 2>&1 || rc=$?
  [[ "${rc}" -eq 1 ]] || die "sandbox user ${SANDBOX_USER} still has processes after the build"
}

container_uid="$(id -u):$(id -g)"
global_cache="${ROOT}/.zig-global-cache"
local_cache="${ROOT}/.zig-cache"
output="${ROOT}/Vendor/TensorFold/zig-out"
prefix_args=()

if [[ -n "${SANDBOX_USER}" ]]; then
  [[ -n "${SANDBOX_ROOT:-}" && -n "${SANDBOX_JOB:-}" ]] \
    || die "SANDBOX_USER is set but SANDBOX_ROOT or SANDBOX_JOB is not"
  sudo -n -u "${SANDBOX_USER}" true \
    || die "SANDBOX_USER=${SANDBOX_USER} cannot run with passwordless sudo"
  JOB="${SANDBOX_ROOT}/${SANDBOX_JOB}"
  [[ ! -e "${JOB}" ]] || die "sandbox job directory ${JOB} already exists"
  mkdir -m 2750 "${JOB}" "${JOB}/sock"
  mkdir -m 2770 "${JOB}/zig-global-cache" "${JOB}/zig-cache" "${JOB}/out" "${JOB}/adapter-home"
  global_cache="${JOB}/zig-global-cache"
  local_cache="${JOB}/zig-cache"
  output="${JOB}/out"
  prefix_args=(--prefix "${output}")
  share_gid="$(getent group "${SHARE_GROUP:-qwen38share}" | cut -d: -f3)"
  [[ -n "${share_gid}" ]] || die "group ${SHARE_GROUP:-qwen38share} does not exist"
  container_uid="$(id -u "${SANDBOX_USER}"):${share_gid}"
else
  mkdir -p "${global_cache}" "${local_cache}" "${output}"
fi

log "building the native worker in ${IMAGE_REF}"
set +e
docker run --rm --init --gpus all --ipc host --network none \
  --user "${container_uid}" --cap-drop ALL --security-opt no-new-privileges \
  --cpus "${KERNELS_CPUS:-16}" --memory "${KERNELS_MEMORY:-64g}" \
  --pids-limit "${KERNELS_PIDS_LIMIT:-4096}" \
  --read-only --tmpfs /tmp:rw,exec,size=8g \
  --mount "type=bind,src=${ROOT},dst=${ROOT},readonly" \
  --mount "type=bind,src=${global_cache},dst=${global_cache}" \
  --mount "type=bind,src=${local_cache},dst=${local_cache}" \
  --mount "type=bind,src=${output},dst=${output}" \
  --mount "type=bind,src=${ZIG_TOOLCHAIN_DIR},dst=/opt/tensorfold-tools/zig,readonly" \
  --env HOME=/tmp --env "ZIG_GLOBAL_CACHE_DIR=${global_cache}" --env "ZIG_LOCAL_CACHE_DIR=${local_cache}" \
  --workdir "${ROOT}/Vendor/TensorFold" --entrypoint /opt/tensorfold-tools/zig/zig \
  "${IMAGE_REF}" build bench-worker -Doptimize=ReleaseFast \
  -Dsm="${CUDA_SM}" -Dnvcc="${NVCC_BIN}" ${prefix_args[@]+"${prefix_args[@]}"} \
  2>&1 | "${ROOT}/tools/public-log-redact.sh"
build_status=("${PIPESTATUS[@]}")
set -e
if (( build_status[0] != 0 )); then
  exit "${build_status[0]}"
fi
# a failed log filter must not turn a successful build into a refusal
(( build_status[1] == 0 )) || echo "build.sh: public log redaction failed; build output withheld" >&2

if [[ -n "${SANDBOX_USER}" ]]; then
  sandbox_quiet
  worker="${output}/bin/bench-worker"
  [[ ! -L "${worker}" && -f "${worker}" && -x "${worker}" ]] \
    || die "sandbox build did not create a regular executable worker"
  [[ "$(file_owner "${worker}")" == "${SANDBOX_USER}" ]] \
    || die "sandbox worker is not owned by ${SANDBOX_USER}"
else
  worker="${output}/bin/bench-worker"
  [[ -x "${worker}" ]] || die "build did not create ${worker}"
fi

TENSORFOLD_WORKER_SOURCE="${worker}" "${ROOT}/tools/stage-cuda-engine.sh"
log "native worker built and staged"

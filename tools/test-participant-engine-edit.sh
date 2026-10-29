#!/usr/bin/env bash
# Prove the editable TensorFold source reaches review and worker identity.
set -euo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
jq -e '.editablePaths == ["Vendor/TensorFold"]' "${SOURCE_ROOT}/benchmark.json" >/dev/null

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
ROOT="${WORK}/repo"
mkdir -p "${ROOT}/tools" "${ROOT}/Vendor/TensorFold/zig-out/bin" "${ROOT}/.github/scripts"
cp "${SOURCE_ROOT}/benchmark.json" "${ROOT}/"
cp "${SOURCE_ROOT}/tools/stage-cuda-engine.sh" "${ROOT}/tools/"
cp "${SOURCE_ROOT}/.github/scripts/submission-static-review-checks.sh" "${ROOT}/.github/scripts/"
printf 'pub const tuned = 1;\n' > "${ROOT}/Vendor/TensorFold/probe.zig"
printf '#!/usr/bin/env bash\nexit 0\n' > "${ROOT}/Vendor/TensorFold/zig-out/bin/bench-worker"
chmod +x "${ROOT}/Vendor/TensorFold/zig-out/bin/bench-worker"
"${ROOT}/tools/stage-cuda-engine.sh" >/dev/null
IDENTITY="${ROOT}/.build/release/bench-worker.source.sha256"
before="$(<"${IDENTITY}")"
mkdir -p "${ROOT}/Vendor/TensorFold/zig-out/generated"
printf 'generated\n' > "${ROOT}/Vendor/TensorFold/zig-out/generated/output.txt"
"${ROOT}/tools/stage-cuda-engine.sh" >/dev/null
generated="$(<"${IDENTITY}")"
[[ "${before}" == "${generated}" ]]
printf 'pub const tuned = 2;\n' > "${ROOT}/Vendor/TensorFold/probe.zig"
"${ROOT}/tools/stage-cuda-engine.sh" >/dev/null
after="$(<"${IDENTITY}")"
[[ "${before}" =~ ^[0-9a-f]{64}$ && "${after}" =~ ^[0-9a-f]{64}$ && "${before}" != "${after}" ]]

review_out="${WORK}/review"
(
  cd "${ROOT}"
  MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
  MLXFAST_SUBMISSION_STATIC_REVIEW_OUT_DIR="${review_out}" \
    .github/scripts/submission-static-review-checks.sh >/dev/null
)
jq -e 'select(.path == "Vendor/TensorFold/probe.zig")' "${review_out}/files.ndjson" >/dev/null

echo "test-participant-engine-edit: PASS"

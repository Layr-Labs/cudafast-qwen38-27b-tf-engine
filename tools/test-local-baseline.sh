#!/usr/bin/env bash
# Check local baseline path selection with no model load.
set -euo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
ROOT="${WORK}/repo"
mkdir -p "${ROOT}/tools" "${ROOT}/fixtures" "${ROOT}/correctness_prompts" "${ROOT}/weights" "${ROOT}/.build/release"
cp "${SOURCE_ROOT}/tools/local-baseline.sh" "${SOURCE_ROOT}/tools/spec-declaration.sh" "${ROOT}/tools/"
cp "${SOURCE_ROOT}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json" "${ROOT}/fixtures/"
printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$@" > "${CAPTURE}.args"\nprintf "%%s\\n" "${MLXFAST_CORRECTNESS_GOLDEN_PATH}" > "${CAPTURE}.golden"\n' > "${ROOT}/benchmark.sh"
printf '#!/usr/bin/env bash\nexit 0\n' > "${ROOT}/.build/release/bench-worker"
chmod +x "${ROOT}/benchmark.sh" "${ROOT}/.build/release/bench-worker"
: > "${ROOT}/weights/model.safetensors"
printf '{"spec":{"enabled":false,"tokens":0}}\n' > "${ROOT}/mtp-head.manifest.json"
if CAPTURE="${WORK}/absent" MLXFAST_TARGET_SNAPSHOT_DIR="${ROOT}/weights" \
    "${ROOT}/tools/local-baseline.sh" >"${WORK}/absent.out" 2>&1; then
  echo "test-local-baseline.sh: absent public golden was accepted" >&2
  exit 1
fi
grep -Fq 'public golden not recorded: correctness_prompts/public-longcopy-gate-english-496.golden.json; run tools/record-goldens.sh' "${WORK}/absent.out"
[[ ! -e "${WORK}/absent.args" ]]

: > "${ROOT}/correctness_prompts/public-longcopy-gate-english-496.golden.json"
: > "${ROOT}/correctness_prompts/public-longcopy-gate-english-496.dflash1.golden.json"
CAPTURE="${WORK}/serial" MLXFAST_TARGET_SNAPSHOT_DIR="${ROOT}/weights" \
  "${ROOT}/tools/local-baseline.sh" >/dev/null
grep -Fxq -- '--local-iterate' "${WORK}/serial.args"
grep -Fxq -- "correctness_prompts/public-longcopy-gate-english-496.golden.json" "${WORK}/serial.golden"

printf '{"spec":{"enabled":true,"tokens":1}}\n' > "${ROOT}/mtp-head.manifest.json"
CAPTURE="${WORK}/dflash" MLXFAST_TARGET_SNAPSHOT_DIR="${ROOT}/weights" \
  "${ROOT}/tools/local-baseline.sh" >/dev/null
grep -Fxq -- "correctness_prompts/public-longcopy-gate-english-496.dflash1.golden.json" "${WORK}/dflash.golden"

echo "test-local-baseline.sh: PASS"

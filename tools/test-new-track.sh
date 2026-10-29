#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TMP="$(mktemp -d)"
trap 'find "${TMP}" -type f -delete; find "${TMP}" -depth -type d -empty -delete' EXIT
mkdir -p "${TMP}/tools" "${TMP}/fixtures" "${TMP}/snapshot"
cp "${ROOT}/tools/new-track.sh" "${TMP}/tools/new-track.sh"
printf '%s\n' '{"trackId":"old-1b-cuda-v1","contractPath":"fixtures/old_track.json","leaderboard":{}}' >"${TMP}/benchmark.json"
printf '%s\n' '{"target":{"files":[]}}' >"${TMP}/fixtures/old_track.json"
printf '%s\n' '#!/usr/bin/env bash' 'PINNED_SOURCE_COMMIT="old"' >"${TMP}/tools/fetch-benchd.sh"
printf 'test snapshot\n' >"${TMP}/snapshot/config.json"
(cd "${TMP}" && tools/new-track.sh \
  --track-id qwen3.8-1b-cuda-v1 \
  --fork-sha eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab \
  --checkpoint Example/Model@22d8d538154e0e5b6f8dcb1fcc74a60a0798104c \
  --bench-commit 8ce0ca7821e9c882f2ff1e1cbd1e2ce561040729 \
  --os Linux --hash-non-lfs --snapshot-dir "${TMP}/snapshot") >/dev/null
jq -e '.target.files | length == 1' "${TMP}/fixtures/qwen3_8_1b_cuda_v1_track.json" >/dev/null
grep -F '8ce0ca7821e9c882f2ff1e1cbd1e2ce561040729' "${TMP}/tools/fetch-benchd.sh" >/dev/null
printf 'test-new-track: PASS\n'

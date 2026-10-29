#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TMP="$(mktemp -d)"
trap 'find "${TMP}" -depth -delete' EXIT

printf '\177ELF' >"${TMP}/benchd"
chmod +x "${TMP}/benchd"
sha="$(sha256sum "${TMP}/benchd" | awk '{print $1}')"
bytes="$(wc -c <"${TMP}/benchd")"
jq -n --arg source '3101418f95eb074cc9c100f8dc61cc85cc924155' --arg sha "${sha}" --argjson bytes "${bytes}" \
  '{branch:"main",source_commit:$source,target_triple:"aarch64-unknown-linux-gnu",sha256:$sha,bytes:$bytes}' >"${TMP}/benchd.manifest.json"
resolved="$(BENCHD_DIST_LOCAL="${TMP}" BENCHD_BIN_DIR="${TMP}/installed" \
  BENCHD_EXPECT_TARGET_TRIPLE=aarch64-unknown-linux-gnu "${ROOT}/tools/fetch-benchd.sh")"
[[ "${resolved}" == "${TMP}/installed/benchd" ]]
printf 'test-fetch-benchd: PASS\n'

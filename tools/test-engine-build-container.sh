#!/usr/bin/env bash
# Test the native build container's Zig mount and resource limits.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

TREE="${TMP}/repo"
mkdir -p "${TREE}/tools/engine" "${TREE}/fixtures" "${TREE}/Vendor/TensorFold" \
  "${TMP}/bin" "${TMP}/zig/lib"
cp "${ROOT}/tools/engine/build.sh" "${TREE}/tools/engine/build.sh"
cp "${ROOT}/tools/public-log-redact.awk" "${ROOT}/tools/public-log-redact.sh" "${TREE}/tools/"
cp "${ROOT}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json" "${TREE}/fixtures/"

cat > "${TMP}/zig/zig" <<'EOF'
#!/usr/bin/env bash
[[ "${1:-}" == version ]] && { echo 0.17.0; exit 0; }
exit 1
EOF
cat > "${TMP}/nvcc" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
cat > "${TREE}/tools/stage-cuda-engine.sh" <<'EOF'
#!/usr/bin/env bash
[[ -x "${TENSORFOLD_WORKER_SOURCE}" ]]
EOF
cat > "${TMP}/bin/docker" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" >> "${DOCKER_ARGV}"
if [[ "${1:-} ${2:-}" == "image inspect" ]]; then
  exit 0
fi
[[ "${1:-}" == run ]] || exit 1
mkdir -p "$(dirname -- "${TEST_WORKER}")"
printf '#!/usr/bin/env bash\nexit 0\n' > "${TEST_WORKER}"
chmod +x "${TEST_WORKER}"
EOF
chmod +x "${TMP}/zig/zig" "${TMP}/nvcc" "${TREE}/tools/stage-cuda-engine.sh" "${TMP}/bin/docker"

: > "${TMP}/docker.argv"
ZIG_DIR="$(cd -- "${TMP}/zig" && pwd -P)"
PATH="${TMP}/bin:${PATH}" DOCKER_ARGV="${TMP}/docker.argv" \
  TEST_WORKER="${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker" \
  TENSORFOLD_ZIG="${TMP}/zig/zig" TENSORFOLD_NVCC="${TMP}/nvcc" \
  "${TREE}/tools/engine/build.sh" >/dev/null

grep -Fxq -- "type=bind,src=${ZIG_DIR},dst=/opt/tensorfold-tools/zig,readonly" "${TMP}/docker.argv"
grep -Fxq -- '/opt/tensorfold-tools/zig/zig' "${TMP}/docker.argv"
[[ -d "${TMP}/zig/lib" ]]

for pair in '--cpus 16' '--memory 64g' '--pids-limit 4096'; do
  flag="${pair% *}"
  value="${pair#* }"
  awk -v flag="${flag}" -v value="${value}" '
    previous == flag && $0 == value { found = 1 }
    { previous = $0 }
    END { exit !found }
  ' "${TMP}/docker.argv"
done

echo 'test-engine-build-container: PASS'

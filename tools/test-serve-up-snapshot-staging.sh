#!/usr/bin/env bash
# Test dereferenced snapshot staging and the Docker mount boundary.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TMP="$(mktemp -d)"
SOCKET_TMP="$(mktemp -d /tmp/snapshot-staging.XXXXXX)"
cleanup() {
  [[ ! -f "${TMP}/docker.pid" ]] || kill "$(cat "${TMP}/docker.pid")" 2>/dev/null || true
  rm -rf "${TMP}" "${SOCKET_TMP}"
}
trap cleanup EXIT

TREE="${TMP}/tree"
SOURCE="${TMP}/cache/snapshots/revision"
DRAFTER="${TMP}/drafter"
BIN="${TMP}/bin"
mkdir -p "${TREE}/tools" "${TREE}/fixtures" "${TMP}/cache/blobs" "$SOURCE" "$DRAFTER" "$BIN"
cp "${ROOT}/tools/serve-up.sh" "${TREE}/tools/serve-up.sh"
chmod +x "${TREE}/tools/serve-up.sh"
printf 'target\n' > "${TMP}/cache/blobs/model.safetensors"
ln -s ../../blobs/model.safetensors "${SOURCE}/model.safetensors"
printf 'drafter\n' > "${DRAFTER}/draft.safetensors"
target_sha="$(shasum -a 256 "${TMP}/cache/blobs/model.safetensors" | awk '{print $1}')"
target_bytes="$(wc -c < "${TMP}/cache/blobs/model.safetensors" | tr -d '[:space:]')"
jq -n --arg sha "$target_sha" --argjson bytes "$target_bytes" \
  '{target:{files:[{path:"model.safetensors",bytes:$bytes,sha256:$sha}],drafter:{}},
    serve_configuration:{container_image:"example.invalid/tensorfold:test",
      container_digest:("sha256:" + ("0" * 64))}}' > "${TREE}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json"
printf x > "${TMP}/worker"
chmod +x "${TMP}/worker"
printf 'MemTotal: 20000000 kB\nMemAvailable: 20000000 kB\n' > "${TMP}/meminfo"

cat > "${BIN}/docker" <<'EOF'
#!/usr/bin/env bash
printf 'docker' >> "${DOCKER_LOG}"
printf ' <%s>' "$@" >> "${DOCKER_LOG}"
printf '\n' >> "${DOCKER_LOG}"
[[ "${1:-} ${2:-}" == "image inspect" ]] && exit 0
if [[ "${1:-}" == exec ]]; then
  pid="$(cat "${DOCKER_PID_FILE}")"
  printf '{"id":0,"ok":true,"protocol_version":1,"backend":"cuda","device":"mock","resident":{"pid":%s,"load_epoch":1}}\n' "$pid"
  exit 0
fi
if [[ "${1:-}" == stop || "${1:-}" == rm ]]; then
  [[ ! -f "${DOCKER_PID_FILE}" ]] || kill "$(cat "${DOCKER_PID_FILE}")" 2>/dev/null || true
  exit 0
fi
[[ "${1:-}" == run ]] || exit 0
socket=""; previous=""
for argument in "$@"; do
  [[ "$previous" == --socket ]] && socket="$argument"
  previous="$argument"
done
printf '%s\n' "$$" > "${DOCKER_PID_FILE}"
touch "$socket"
exec sleep 60
EOF
chmod +x "${BIN}/docker"

common=(PATH="${BIN}:${PATH}" DOCKER_LOG="${TMP}/docker.log" DOCKER_PID_FILE="${TMP}/docker.pid"
  TENSORFOLD_BENCH_WORKER="${TMP}/worker" SERVE_UP_MEMINFO="${TMP}/meminfo"
  SERVE_UP_SOCKET_DIR="${SOCKET_TMP}" SERVE_UP_LOG_DIR="${TMP}/logs"
  SERVE_UP_HEALTH_TIMEOUT_S=5 SERVE_UP_TEST_READY_FILE=1 SERVE_UP_TEST_REQUIRE_PROBE=1)

: > "${TMP}/docker.log"
if env "${common[@]}" MLXFAST_TARGET_SNAPSHOT_DIR="$SOURCE" \
    "${TREE}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
    --socket-out "${TMP}/symlink.socket" > "${TMP}/symlink.out" 2>&1; then
  echo 'test-serve-up-snapshot-staging: symlink snapshot was accepted' >&2
  exit 1
fi
grep -Fq 'run the ranked workflow snapshot staging step' "${TMP}/symlink.out"
[[ ! -s "${TMP}/docker.log" ]]

env_out="${TMP}/staged.env"
MLXFAST_TARGET_SNAPSHOT_DIR="$SOURCE" MLXFAST_DRAFTER_SNAPSHOT_DIR="$DRAFTER" \
  "${TREE}/tools/serve-up.sh" --stage-snapshots --dest-root "${TMP}/staged" \
  --group "$(id -gn)" --env-out "$env_out" > "${TMP}/stage.out" 2>&1
[[ -f "${TMP}/staged/target/model.safetensors" ]]
[[ ! -L "${TMP}/staged/target/model.safetensors" ]]
grep -Fq 'recording staged drafter sha256 values' "${TMP}/stage.out"
grep -Fq './draft.safetensors' "${TMP}/stage.out"
# shellcheck disable=SC1090
. "$env_out"
export MLXFAST_TARGET_SNAPSHOT_DIR MLXFAST_DRAFTER_SNAPSHOT_DIR MLXFAST_WEIGHTS_PATH SERVE_UP_WEIGHTS_DIR
[[ "$MLXFAST_WEIGHTS_PATH" == "$MLXFAST_TARGET_SNAPSHOT_DIR" ]]
[[ "$SERVE_UP_WEIGHTS_DIR" == "$MLXFAST_TARGET_SNAPSHOT_DIR" ]]
while IFS= read -r path; do
  [[ "$(stat -c %a "$path" 2>/dev/null || stat -f %Lp "$path")" == 2750 ]]
  [[ "$(stat -c %U "$path" 2>/dev/null || stat -f %Su "$path")" == "$(id -un)" ]]
  [[ "$(stat -c %G "$path" 2>/dev/null || stat -f %Sg "$path")" == "$(id -gn)" ]]
done < <(find "${TMP}/staged" -type d)
while IFS= read -r path; do
  [[ "$(stat -c %a "$path" 2>/dev/null || stat -f %Lp "$path")" == 640 ]]
  [[ "$(stat -c %U "$path" 2>/dev/null || stat -f %Su "$path")" == "$(id -un)" ]]
  [[ "$(stat -c %G "$path" 2>/dev/null || stat -f %Sg "$path")" == "$(id -gn)" ]]
done < <(find "${TMP}/staged" -type f)

: > "${TMP}/docker.log"
env "${common[@]}" MLXFAST_TARGET_SNAPSHOT_DIR="$MLXFAST_TARGET_SNAPSHOT_DIR" \
  MLXFAST_DRAFTER_SNAPSHOT_DIR="$MLXFAST_DRAFTER_SNAPSHOT_DIR" \
  "${TREE}/tools/serve-up.sh" --boot --spec dflash --draft-len 1 \
  --socket-out "${TMP}/mounted.socket" > "${TMP}/mounted.out" 2>&1
run="$(grep '^docker <run>' "${TMP}/docker.log")"
grep -Fq "<type=bind,src=${MLXFAST_TARGET_SNAPSHOT_DIR},dst=${MLXFAST_TARGET_SNAPSHOT_DIR},readonly>" <<< "$run"
grep -Fq "<type=bind,src=${MLXFAST_DRAFTER_SNAPSHOT_DIR},dst=${MLXFAST_DRAFTER_SNAPSHOT_DIR},readonly>" <<< "$run"
env "${common[@]}" SERVE_UP_LOG_DIR="${TMP}/logs" \
  "${TREE}/tools/serve-up.sh" --stop --socket "$(sed -n '1p' "${TMP}/mounted.socket")" >/dev/null 2>&1

printf 'change\n' > "${TMP}/cache/blobs/model.safetensors"
if MLXFAST_TARGET_SNAPSHOT_DIR="$SOURCE" MLXFAST_DRAFTER_SNAPSHOT_DIR="$DRAFTER" \
    "${TREE}/tools/serve-up.sh" --stage-snapshots --dest-root "${TMP}/bad-stage" \
    --group "$(id -gn)" --env-out "${TMP}/bad.env" > "${TMP}/bad.out" 2>&1; then
  echo 'test-serve-up-snapshot-staging: sha mismatch was accepted' >&2
  exit 1
fi
grep -Fq 'sha256 mismatch for target model.safetensors' "${TMP}/bad.out"

workflow="${ROOT}/.github/workflows/benchmark.yml"
setup_line="$(grep -nF -- '- name: Setup (engine + pinned checkpoint)' "$workflow" | cut -d: -f1)"
stage_line="$(grep -nF -- '- name: Stage model snapshots (dereferenced and verified)' "$workflow" | cut -d: -f1)"
measure_line="$(grep -nF -- '- name: GPU-exclusive measurement' "$workflow" | cut -d: -f1)"
(( setup_line < stage_line && stage_line < measure_line ))
grep -Fq -- '--dest-root "${SANDBOX_ROOT}/${SANDBOX_JOB}/weights"' "$workflow"
grep -Fq -- '--group "${SHARE_GROUP:-qwen38share}"' "$workflow"
grep -Fq 'setfacl -m "u:${SANDBOX_USER}:---" "${RUNNER_TEMP}"' "$workflow"
grep -Fq 'resident_log_dir="$(mktemp -d "${RUNNER_TEMP}/resident-logs.XXXXXX")"' "$workflow"
grep -Fq 'setfacl -b -k -- "${resident_log_dir}"' "$workflow"
grep -Fq "printf 'SERVE_UP_LOG_DIR=%s\\n' \"\${resident_log_dir}\" >> \"\${GITHUB_ENV}\"" "$workflow"
grep -Fq 'rm -rf "${job_dir}" || rc=1' "$workflow"
! grep -Fq -- '--dest-root "${RUNNER_TEMP}/qwen38-model-snapshots"' "$workflow"
grep -Fq 'export SERVE_UP_WEIGHTS_DIR="${WEIGHTS_PATH}"' \
  "${ROOT}/tools/qwen38-measure-and-score.sh"

printf 'test-serve-up-snapshot-staging: PASS\n'

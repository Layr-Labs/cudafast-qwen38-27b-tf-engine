#!/usr/bin/env bash
# test-record-goldens.sh -- public-only recording with stubbed engine tools.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok() { echo "ok    $*"; }

TEST_ROOT="${WORK}/repo"
PROMPTS="${WORK}/prompts"
OUT="${WORK}/out"
SNAP="${WORK}/snapshot"
BIN="${WORK}/bin"
mkdir -p "${TEST_ROOT}/tools" "${TEST_ROOT}/fixtures" "${PROMPTS}" "${OUT}" "${SNAP}" "${BIN}"
cp "${ROOT}/tools/record-goldens.sh" "${TEST_ROOT}/tools/record-goldens.sh"
cp "${ROOT}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json" "${TEST_ROOT}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json"
cp "${ROOT}/benchmark.json" "${TEST_ROOT}/benchmark.json"

printf '{"eos_token_id":[42]}\n' > "${SNAP}/generation_config.json"
GEN_SHA="$(sha256sum "${SNAP}/generation_config.json" | cut -d' ' -f1)"
jq --arg sha "${GEN_SHA}" '(.target.files[] | select(.path == "generation_config.json")).sha256 = $sha' \
  "${TEST_ROOT}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json" > "${WORK}/fixture"
mv "${WORK}/fixture" "${TEST_ROOT}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json"

cat > "${TEST_ROOT}/tools/serve-up.sh" <<'EOF'
#!/usr/bin/env bash
set -e
if [[ "$1" == --stop ]]; then exit 0; fi
while [[ $# -gt 0 ]]; do
  case "$1" in --socket-out) socket_out="$2"; shift 2 ;; *) shift ;; esac
done
printf 'stub.socket\n' > "${socket_out}"
EOF
cat > "${BIN}/record-correctness-golden" <<'EOF'
#!/usr/bin/env bash
set -e
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    --case-name) name="$2"; shift 2 ;;
    --benchmark-spec) spec="$2"; shift 2 ;;
    *) shift ;;
  esac
done
if [[ "${spec}" == serial ]]; then cases='[{"tokens":[1,2,3]}]'; else cases='[{"tokens":[9]}]'; fi
jq -n --arg name "${name}" --arg spec "${spec}" --argjson cases "${cases}" \
  '{case_name:$name, cases:$cases, benchmark:{spec:$spec}}' > "${out}"
EOF
cat > "${BIN}/benchd" <<'EOF'
#!/usr/bin/env bash
set -e
[[ "$1" == validate-golden ]]
while [[ $# -gt 0 ]]; do
  case "$1" in --golden) golden="$2"; shift 2 ;; *) shift ;; esac
done
jq -e '.cases | type == "array"' "${golden}" >/dev/null
EOF
chmod +x "${TEST_ROOT}/tools/record-goldens.sh" "${TEST_ROOT}/tools/serve-up.sh" "${BIN}/record-correctness-golden" "${BIN}/benchd"

run() {
  MLXFAST_TARGET_SNAPSHOT_DIR="${SNAP}" BENCHD_BIN_DIR="${BIN}" \
    "${TEST_ROOT}/tools/record-goldens.sh" --prompts "${PROMPTS}" --out "${OUT}" "$@" 2>&1
}
refused() {
  local label="$1" want="$2"
  shift 2
  local output
  if output="$(run "$@")"; then
    fail "${label} was accepted"
  elif grep -Fq -- "${want}" <<<"${output}"; then
    ok "${label} is refused"
  else
    fail "${label} refusal is wrong: ${output}"
  fi
}

refused "--public-only with --only" "--public-only takes no --only" --public-only --only pool-1
refused "--public-only with --oracles-for" "--public-only takes no --oracles-for" --public-only --oracles-for pool-1
refused "--public-only with --apply" "--public-only takes no --apply" --public-only --apply
refused "--public-only without public.tokens.json" "--public-only requires ${PROMPTS}/public.tokens.json" --public-only

printf '[101,102]\n' > "${PROMPTS}/public.tokens.json"
if output="$(run --public-only)"; then
  serial="${OUT}/public/public-longcopy-gate-english-496.golden.json"
  dflash="${OUT}/public/public-longcopy-gate-english-496.dflash1.golden.json"
  if [[ -f "${serial}" && -f "${dflash}" ]] \
    && [[ "$(find "${OUT}" -name '*.golden.json' | wc -l | tr -d ' ')" == 2 ]] \
    && cmp -s <(jq -c '.cases' "${serial}") <(jq -c '.cases' "${dflash}") \
    && [[ "$(jq -r '.benchmark.spec' "${dflash}")" == dflash:1 ]]; then
    ok "--public-only writes only the serial and DFlash depth-1 public goldens with serial cases[]"
  else
    fail "public-only output is wrong: ${output}"
  fi
else
  fail "--public-only stubbed run failed: ${output}"
fi

if [[ "${failures}" -eq 0 ]]; then
  echo "test-record-goldens.sh: all cases passed"
  exit 0
fi
echo "test-record-goldens.sh: ${failures} case(s) failed" >&2
exit 1

#!/usr/bin/env bash
# Test the contained TensorFold build and staged-copy guards.
set -euo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TMP="$(cd "$(mktemp -d)" && pwd -P)"
ROOT="${TMP}/repo"
mkdir -p "${ROOT}/tools/engine" "${ROOT}/fixtures" "${ROOT}/Vendor/TensorFold"
cp "${SOURCE_ROOT}/tools/engine/build.sh" "${ROOT}/tools/engine/build.sh"
cp "${SOURCE_ROOT}/tools/stage-cuda-engine.sh" "${ROOT}/tools/stage-cuda-engine.sh"
cp "${SOURCE_ROOT}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json" "${ROOT}/fixtures/"
OUT="${ROOT}/Vendor/TensorFold/zig-out"
STAGED="${ROOT}/.build/release/bench-worker"
cleanup() {
  rm -rf "${TMP}"
}
trap cleanup EXIT

BIN="${TMP}/bin"
# build.sh mounts the whole toolchain dir, so the fake Zig needs an adjacent lib/.
mkdir -p "${BIN}/lib"
LOG="${TMP}/calls.log"
export LOG
cat > "${BIN}/zig" <<'EOF'
#!/usr/bin/env bash
[[ "${1:-}" == version ]] && { echo 0.17.0; exit 0; }
exit 2
EOF
cat > "${BIN}/nvcc" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
cat > "${BIN}/docker" <<'EOF'
#!/usr/bin/env bash
printf 'docker' >> "${LOG}"
printf ' <%s>' "$@" >> "${LOG}"
printf '\n' >> "${LOG}"
[[ "${1:-} ${2:-}" == "image inspect" ]] && exit 0
[[ "${1:-}" == run ]] || exit 0
out=""
for arg in "$@"; do
  case "${arg}" in
    type=bind,src=*,dst=*)
      src="${arg#type=bind,src=}"; src="${src%%,dst=*}"
      case "${src}" in */zig-out|*/out) out="${src}" ;; esac
      ;;
  esac
done
[[ -n "${out}" ]] || exit 3
mkdir -p "${out}/bin"
printf '#!/usr/bin/env bash\nexit 0\n' > "${out}/bin/bench-worker"
chmod +x "${out}/bin/bench-worker"
EOF
cat > "${BIN}/sudo" <<'EOF'
#!/usr/bin/env bash
printf 'sudo' >> "${LOG}"
printf ' <%s>' "$@" >> "${LOG}"
printf '\n' >> "${LOG}"
[[ "${STUB_SUDO_FAIL:-0}" == 1 ]] && exit 1
exit 0
EOF
cat > "${BIN}/pgrep" <<'EOF'
#!/usr/bin/env bash
exit 1
EOF
cat > "${BIN}/getent" <<EOF
#!/usr/bin/env bash
printf 'qwen38share:x:%s:\n' "$(id -g)"
EOF
chmod +x "${BIN}"/*

: > "${LOG}"
PATH="${BIN}:${PATH}" TENSORFOLD_ZIG=zig TENSORFOLD_NVCC="${BIN}/nvcc" \
  "${ROOT}/tools/engine/build.sh" >/dev/null
run="$(grep '^docker <run>' "${LOG}")"
for flag in '<--network> <none>' '<--read-only>' '<--cap-drop> <ALL>' \
  '<--security-opt> <no-new-privileges>' '<--gpus> <all>' '<--ipc> <host>'; do
  grep -Fq -- "${flag}" <<<"${run}"
done
grep -Fq -- "<type=bind,src=${ROOT},dst=${ROOT},readonly>" <<<"${run}"
grep -Fq -- "<type=bind,src=${BIN},dst=/opt/tensorfold-tools/zig,readonly>" <<<"${run}"
grep -Fq -- '<build> <bench-worker> <-Doptimize=ReleaseFast> <-Dsm=121>' <<<"${run}"
[[ -x "${STAGED}" ]]

rm -rf "${OUT}" "${ROOT}/.zig-global-cache" "${ROOT}/.zig-cache"
rm -f "${STAGED}"
SANDBOX_ROOT="${TMP}/sandbox"
SANDBOX_JOB=42-1
RUNNER_TEMP="${TMP}/runner-temp"
mkdir -p "${SANDBOX_ROOT}" "${RUNNER_TEMP}"
me="$(id -un)"
: > "${LOG}"
PATH="${BIN}:${PATH}" TENSORFOLD_ZIG=zig TENSORFOLD_NVCC="${BIN}/nvcc" \
  SANDBOX_USER="${me}" SANDBOX_ROOT="${SANDBOX_ROOT}" SANDBOX_JOB="${SANDBOX_JOB}" \
  RUNNER_TEMP="${RUNNER_TEMP}" "${ROOT}/tools/engine/build.sh" >/dev/null
job="${SANDBOX_ROOT}/${SANDBOX_JOB}"
grep -Fq -- "<--user> <$(id -u):$(id -g)>" "${LOG}"
for entry in sock:2750 zig-global-cache:2770 zig-cache:2770 out:2770 adapter-home:2770; do
  [[ "$(stat -c %a "${job}/${entry%%:*}" 2>/dev/null || stat -f %Lp "${job}/${entry%%:*}")" == "${entry##*:}" ]]
done
[[ -f "${STAGED}.sandboxed" && -x "${STAGED}" ]]
grep -Fq 'run-engine-sandboxed.sh' "${STAGED}"

# A failed sudo handoff must stop before the build or job directory.
: > "${LOG}"
if err="$(PATH="${BIN}:${PATH}" TENSORFOLD_ZIG=zig TENSORFOLD_NVCC="${BIN}/nvcc" \
    STUB_SUDO_FAIL=1 SANDBOX_USER="${me}" SANDBOX_ROOT="${SANDBOX_ROOT}" SANDBOX_JOB=43-1 \
    RUNNER_TEMP="${RUNNER_TEMP}" "${ROOT}/tools/engine/build.sh" 2>&1)"; then
  echo 'test-engine-build: failed sudo handoff was accepted' >&2
  exit 1
fi
grep -Fq "SANDBOX_USER=${me} cannot run with passwordless sudo" <<<"${err}"
[[ ! -e "${SANDBOX_ROOT}/43-1" ]]
! grep -Fq 'docker <run>' "${LOG}"

real="${TMP}/real-worker"
printf '#!/usr/bin/env bash\nexit 0\n' > "${real}"
chmod +x "${real}"
ln -s "${real}" "${TMP}/linked-worker"
if SANDBOX_USER="${me}" SANDBOX_ROOT="${SANDBOX_ROOT}" SANDBOX_JOB=x RUNNER_TEMP="${RUNNER_TEMP}" \
    TENSORFOLD_WORKER_SOURCE="${TMP}/linked-worker" "${ROOT}/tools/stage-cuda-engine.sh" >/dev/null 2>&1; then
  echo 'test-engine-build: symlink source was accepted' >&2
  exit 1
fi
if SANDBOX_USER=definitely-not-owner SANDBOX_ROOT="${SANDBOX_ROOT}" SANDBOX_JOB=x RUNNER_TEMP="${RUNNER_TEMP}" \
    TENSORFOLD_WORKER_SOURCE="${real}" "${ROOT}/tools/stage-cuda-engine.sh" >/dev/null 2>&1; then
  echo 'test-engine-build: foreign-owned source was accepted' >&2
  exit 1
fi

# Values baked into the trusted shim must be plain, owned paths.
shim_refused() {
  local label="$1" needle="$2"
  shift 2
  rm -f "${STAGED}" "${STAGED}.sandboxed" "${STAGED}.source.sha256"
  if out="$(env SANDBOX_USER="${me}" SANDBOX_ROOT="${SANDBOX_ROOT}" SANDBOX_JOB=x \
      RUNNER_TEMP="${RUNNER_TEMP}" "$@" TENSORFOLD_WORKER_SOURCE="${real}" \
      "${ROOT}/tools/stage-cuda-engine.sh" 2>&1)"; then
    echo "test-engine-build: ${label} was accepted" >&2
    exit 1
  fi
  grep -Fq -- "${needle}" <<<"${out}"
  [[ ! -e "${STAGED}" && ! -e "${STAGED}.sandboxed" ]]
}
shim_refused 'invalid SANDBOX_JOB' 'SANDBOX_JOB' 'SANDBOX_JOB=1;rm'
if out="$(SANDBOX_USER='a$(id)' "${ROOT}/tools/stage-cuda-engine.sh" --print-shim 2>&1)"; then
  echo 'test-engine-build: invalid SANDBOX_USER was accepted' >&2
  exit 1
fi
grep -Fq 'is not a plain user name' <<<"${out}"
shim_refused 'relative SANDBOX_ROOT' 'SANDBOX_ROOT' SANDBOX_ROOT=relative
shim_refused 'unset RUNNER_TEMP' 'RUNNER_TEMP' RUNNER_TEMP=
ln -s "${RUNNER_TEMP}" "${TMP}/runner-temp-link"
shim_refused 'symlinked RUNNER_TEMP' 'RUNNER_TEMP' RUNNER_TEMP="${TMP}/runner-temp-link"
shim_refused 'foreign-owned RUNNER_TEMP' 'RUNNER_TEMP' RUNNER_TEMP=/
expected_shim="$(SANDBOX_USER="${me}" SANDBOX_ROOT="${SANDBOX_ROOT}" SANDBOX_JOB=x \
  RUNNER_TEMP="${RUNNER_TEMP}" "${ROOT}/tools/stage-cuda-engine.sh" --print-shim)"
SANDBOX_USER="${me}" SANDBOX_ROOT="${SANDBOX_ROOT}" SANDBOX_JOB=x \
  RUNNER_TEMP="${RUNNER_TEMP}" TENSORFOLD_WORKER_SOURCE="${real}" \
  "${ROOT}/tools/stage-cuda-engine.sh" >/dev/null
[[ "$(<"${STAGED}")" == "${expected_shim}" ]]

CPBIN="${TMP}/cpbin"
mkdir -p "${CPBIN}"
REAL_CP="$(command -v cp)"
cat > "${CPBIN}/cp" <<EOF
#!/usr/bin/env bash
src="\${*: -2:1}"
rm -f "\${src}"
ln -s /etc/passwd "\${src}"
exec "${REAL_CP}" "\$@"
EOF
chmod +x "${CPBIN}/cp"
swap="${TMP}/swap-worker"
printf '#!/usr/bin/env bash\nexit 0\n' > "${swap}"
chmod +x "${swap}"
if PATH="${CPBIN}:${PATH}" SANDBOX_USER="${me}" SANDBOX_ROOT="${SANDBOX_ROOT}" SANDBOX_JOB=x \
    RUNNER_TEMP="${RUNNER_TEMP}" TENSORFOLD_WORKER_SOURCE="${swap}" \
    "${ROOT}/tools/stage-cuda-engine.sh" >/dev/null 2>&1; then
  echo 'test-engine-build: swapped staged copy was accepted' >&2
  exit 1
fi
[[ ! -e "${STAGED}.sandboxed" && ! -L "${STAGED}.sandboxed" ]]

printf 'test-engine-build: PASS\n'

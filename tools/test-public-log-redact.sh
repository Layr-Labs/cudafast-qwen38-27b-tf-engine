#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
FILTER="${ROOT}/tools/public-log-redact.sh"
AWK_PROGRAM="${ROOT}/tools/public-log-redact.awk"
AWK=/usr/bin/awk
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
failures=0

fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

sample_out="$(
  printf '%s\n' \
    'runner environment ok: snapshot=/home/fake-op/weights/target benchd=/home/fake-op/runner/benchd-bin' \
    'reference workspace ok: /home/fake-op/prod/reference at deadbeef' \
    'paired baseline: calibration=/home/fake-op/calibration/spark.json private=/tmp/measure/calibration.json goldens=/home/fake-op/goldens' \
    'paths: /home/fake-op/weights/drafter /home/fake-op/actions-runner/jobs /home/fake-op/run/sockets /home/fake-op/sandbox /home/fake-op/work/repo' \
    'paired official run on box fake-runner-42; host fake-host-77; extra /home/someone/other' \
    '{"baseline_box":"fake-runner-42"}' \
    '::error title=failed::bad path /home/fake-op/weights/target' \
    '::notice::runner fake-runner-42' \
    '::add-mask::/home/fake-op/private' \
  | HOME=/home/fake-op \
    BENCHD_BIN_DIR=/home/fake-op/runner/benchd-bin \
    MLXFAST_TARGET_SNAPSHOT_DIR=/home/fake-op/weights/target \
    MLXFAST_DRAFTER_SNAPSHOT_DIR=/home/fake-op/weights/drafter \
    MLXFAST_BASELINE_WORKSPACE=/home/fake-op/prod/reference \
    MLXFAST_BASELINE_CALIBRATION=/home/fake-op/calibration/spark.json \
    PUBLIC_LOG_CALIBRATION_FILE=/tmp/measure/calibration.json \
    MLXFAST_QWEN38_GOLDEN_DIR=/home/fake-op/goldens \
    RUNNER_DIR=/home/fake-op/actions-runner/ \
    SERVE_UP_SOCKET_DIR=/home/fake-op/run/sockets \
    SANDBOX_ROOT=/home/fake-op/sandbox \
    GITHUB_WORKSPACE=/home/fake-op/work/repo \
    RUNNER_NAME=fake-runner-42 \
    PUBLIC_LOG_HOST=fake-host-77 \
    "${FILTER}"
)"

[[ "${sample_out}" != *"/home/"* ]] || fail "sample output still contains /home/"
[[ "${sample_out}" != *"fake-runner-42"* ]] || fail "sample output still contains the runner name"
[[ "${sample_out}" != *"fake-host-77"* ]] || fail "sample output still contains the hostname"
for role in '<target-snapshot>' '<drafter-snapshot>' '<benchd-dir>' \
  '<reference-workspace>' '<calibration-file>' '<golden-dir>' '<runner-dir>' '<socket-dir>' \
  '<sandbox-root>' '<workspace>' '<ranked-box>' '<home>/'; do
  [[ "${sample_out}" == *"${role}"* ]] || fail "sample output lacks ${role}"
done
grep -q '^::error title=failed::' <<< "${sample_out}" || fail "::error:: line did not survive"
grep -q '^::notice::' <<< "${sample_out}" || fail "::notice:: line did not survive"
grep -q '^::add-mask::' <<< "${sample_out}" || fail "::add-mask:: line did not survive"

path_out="$(printf '%s\n' '/dir /dir/child' \
  | HOME=/ BENCHD_BIN_DIR=/dir/ LC_ALL=C "${AWK}" -f "${AWK_PROGRAM}")"
[[ "${path_out}" == '<benchd-dir> <benchd-dir>/child' ]] \
  || fail "trailing-slash path did not redact dir and child: ${path_out}"

literal_path='/tmp/a.b+[x](y){z}|q?^$&\tail'
literal_out="$(printf '%s\n' "${literal_path}" \
  | HOME=/ BENCHD_BIN_DIR="${literal_path}" LC_ALL=C "${AWK}" -f "${AWK_PROGRAM}")"
[[ "${literal_out}" == '<benchd-dir>' ]] \
  || fail "ERE metacharacters were not treated literally: ${literal_out}"

word_out="$(printf '%s\n' 'gpu gpus xgpu gpu-x (gpu) abc abcd xabc abc_x [abc]' \
  | HOME=/ RUNNER_NAME=abc LC_ALL=C "${AWK}" -f "${AWK_PROGRAM}" -v host=gpu)"
[[ "${word_out}" == '<ranked-box> gpus xgpu gpu-x (<ranked-box>) <ranked-box> abcd xabc abc_x [<ranked-box>]' ]] \
  || fail "name word boundaries were wrong: ${word_out}"

number_out="$(printf '%s\n' '1234 01234 12345 x01234' \
  | HOME=/ RUNNER_NAME=1234 LC_ALL=C "${AWK}" -f "${AWK_PROGRAM}" -v host=01234)"
[[ "${number_out}" == '<ranked-box> <ranked-box> 12345 x01234' ]] \
  || fail "numeric-looking names were deduplicated as numbers: ${number_out}"

for unsafe_home in '' '/'; do
  safe_out="$(printf '%s\n' '/only/root/path' \
    | HOME="${unsafe_home}" BENCHD_BIN_DIR=relative LC_ALL=C "${AWK}" -f "${AWK_PROGRAM}")"
  [[ "${safe_out}" == '/only/root/path' ]] \
    || fail "HOME='${unsafe_home}' or relative path was accepted: ${safe_out}"
done
bare_home="$(printf '%s\n' '/home/alice /home/alice/private' \
  | HOME='' LC_ALL=C "${AWK}" -f "${AWK_PROGRAM}")"
[[ "${bare_home}" == '<home> <home>/private' ]] \
  || fail "/home/name forms were not redacted: ${bare_home}"

printf '\377raw-byte\n' > "${TMP}/non-utf8.in"
HOME=/ LC_ALL=C "${AWK}" -f "${AWK_PROGRAM}" < "${TMP}/non-utf8.in" > "${TMP}/non-utf8.out" \
  || fail "non-UTF-8 input made the filter fail"
cmp -s "${TMP}/non-utf8.in" "${TMP}/non-utf8.out" \
  || fail "non-UTF-8 bytes did not pass through unchanged"

"${AWK}" 'BEGIN {
  for (i = 0; i < 4000; i++) printf "/p "
  for (i = 0; i < 4000; i++) printf "xabcx "
  for (i = 36000; i < 65536; i++) printf "z"
  print ""
}' > "${TMP}/linear.in"
start_ns="$(python3 -c 'import time; print(time.monotonic_ns())')"
HOME=/ BENCHD_BIN_DIR=/p RUNNER_NAME=abc LC_ALL=C "${AWK}" -f "${AWK_PROGRAM}" \
  < "${TMP}/linear.in" > "${TMP}/linear.out" \
  || fail "64 KiB linear-cost case failed"
end_ns="$(python3 -c 'import time; print(time.monotonic_ns())')"
elapsed_ns=$((end_ns - start_ns))
(( elapsed_ns < 1000000000 )) \
  || fail "64 KiB linear-cost case took $((elapsed_ns / 1000000)) ms"
linear_counts="$("${AWK}" '{ paths += gsub(/<benchd-dir>/, ""); near += gsub(/xabcx/, "") }
  END { print paths, near }' "${TMP}/linear.out")"
[[ "${linear_counts}" == '4000 4000' ]] \
  || fail "linear-cost case did not contain 4000 redactions and 4000 rejected near-matches: ${linear_counts}"

"${AWK}" 'BEGIN { for (i = 0; i < 3 * 1024 * 1024; i++) printf "x"; print "" }' \
  > "${TMP}/long.in"
HOME=/ LC_ALL=C "${AWK}" -f "${AWK_PROGRAM}" < "${TMP}/long.in" > "${TMP}/long.out" \
  || fail "3 MiB line made the filter fail"
"${AWK}" 'BEGIN { note = " [3080192 more bytes not shown]" }
  length($0) != 65536 + length(note) || substr($0, length($0) - length(note) + 1) != note { exit 1 }' \
  "${TMP}/long.out" || fail "3 MiB line was not cut at 64 KiB with the byte-count note"

held="${TMP}/marker"
HOME=/home/fake-op RUNNER_NAME=score bash -c \
  'printf "%s\n" "benchd iterate: wrote score.json" >&2; printf "%s\n" "/home/fake-op/private" >&2; exit 42' \
  2>&1 > "${TMP}/producer.stdout" \
  | HOME=/home/fake-op RUNNER_NAME=score LC_ALL=C "${AWK}" -f "${AWK_PROGRAM}" \
      -v held="${held}" -v host=score > "${TMP}/attempt.stderr"
producer_status="${PIPESTATUS[0]}"
[[ "${producer_status}" == "42" ]] || fail "run_attempt producer status changed: ${producer_status}"
[[ "$(cat "${held}" 2>/dev/null)" == 'benchd iterate: wrote score.json' ]] \
  || fail "raw score marker was not captured unchanged"
[[ "$(cat "${TMP}/attempt.stderr")" == '<home>/private' ]] \
  || fail "run_attempt stderr was not redacted"

stream_out="${TMP}/stream.out"
(
  printf 'first-line\n'
  sleep 1
  printf 'last-line\n'
) | PUBLIC_LOG_HOST=fake-host-77 "${FILTER}" > "${stream_out}" &
stream_pid=$!
first_seen=0
for _ in {1..40}; do
  if grep -q '^first-line$' "${stream_out}" 2>/dev/null; then
    first_seen=1
    break
  fi
  sleep 0.02
done
[[ "${first_seen}" -eq 1 ]] || fail "first line was buffered until producer exit"
if [[ "${first_seen}" -eq 1 ]]; then
  kill -0 "${stream_pid}" 2>/dev/null || fail "producer ended before the first line was observed"
  ! grep -q '^last-line$' "${stream_out}" 2>/dev/null || fail "producer had already emitted its last line"
fi
wait "${stream_pid}" || fail "streaming filter failed"

mkdir -p "${TMP}/bin"
printf '%s\n' '#!/usr/bin/env bash' 'exit 1' > "${TMP}/bin/hostname"
chmod +x "${TMP}/bin/hostname"
unset PUBLIC_LOG_HOST
PATH="${TMP}/bin:${PATH}" "${FILTER}" </dev/null > "${TMP}/host.out" 2> "${TMP}/host.err" \
  || fail "failed hostname lookup made the filter fail"
warning_count="$(grep -c 'hostname lookup failed' "${TMP}/host.err" || true)"
[[ "${warning_count}" == "1" ]] || fail "hostname failure printed ${warning_count} warnings"

if [[ "${failures}" -ne 0 ]]; then
  echo "test-public-log-redact: ${failures} failure(s)" >&2
  exit 1
fi
echo "test-public-log-redact: PASS ($((elapsed_ns / 1000000)) ms linear-cost case)"

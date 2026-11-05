#!/usr/bin/env bash
# Redact box-local identities from stdin and write the result to stdout.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${PUBLIC_LOG_HOST+x}" ]]; then
  host_name="${PUBLIC_LOG_HOST}"
elif ! host_name="$(hostname 2>/dev/null)" || [[ -z "${host_name}" ]]; then
  echo "public-log-redact.sh: warning: hostname lookup failed; continuing without hostname redaction" >&2
  host_name=""
fi

# The filter must pass each line on as it arrives. gawk does that: it reads
# what is in the pipe and the program calls fflush() after each line. mawk
# reads its input in blocks, so it holds the first line until the block fills
# or the producer exits. `-W interactive` makes mawk read stdin one line at a
# time and write stdout unbuffered. The program sets no RS, so the records are
# the same. Other awks get no extra option, so gawk runs exactly as before.
awk_options=()
awk_version="$(awk -W version </dev/null 2>/dev/null || true)"
if [[ "${awk_version}" == mawk* ]]; then
  awk_options=(-W interactive)
fi

LC_ALL=C exec awk ${awk_options[@]+"${awk_options[@]}"} -f "${SCRIPT_DIR}/public-log-redact.awk" -v host="${host_name}"

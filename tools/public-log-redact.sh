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

LC_ALL=C exec awk -f "${SCRIPT_DIR}/public-log-redact.awk" -v host="${host_name}"

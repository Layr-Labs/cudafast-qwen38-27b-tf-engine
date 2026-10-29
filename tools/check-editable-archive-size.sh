#!/usr/bin/env bash
# Keep the compressed editable-surface archive below 20 MiB (5 MiB under Yukon's
# 25 MiB submission cap). Like `yukon submit`, it archives every file under each
# benchmark.json editablePaths entry as it is on disk -- tracked, untracked and
# ignored alike, .git included -- so a build or a stray file left inside the
# editable surface counts too.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
cd "${ROOT}"

MAX_BYTES=20971520
WORK="$(mktemp -d "${TMPDIR:-/tmp}/editable-archive.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT
PATHS="${WORK}/paths.list"
LIST="${WORK}/files.list"
ARCHIVE="${WORK}/editable.tar.gz"
fail() { echo "check-editable-archive-size: FAIL: $1" >&2; exit 1; }

command -v jq >/dev/null 2>&1 || fail "jq is required"
jq -e '.editablePaths | type == "array" and length > 0 and all(type == "string")' benchmark.json >/dev/null \
  || fail "benchmark.json has no editablePaths list"
jq -r '.editablePaths[]' benchmark.json > "${PATHS}" || fail "cannot read editablePaths"

: > "${LIST}"
editables=()
while IFS= read -r editable; do
  if [[ -z "${editable}" || "${editable}" == /* || "${editable}" == :* \
        || "/${editable}/" == *"/../"* || "/${editable}/" == *"/./"* ]]; then
    fail "invalid editable path '${editable}'"
  fi
  [[ -e "${editable}" || -L "${editable}" ]] || fail "editable path '${editable}' does not exist"
  find "${editable}" \( -type f -o -type l \) -print0 >> "${LIST}" \
    || fail "cannot list '${editable}'"
  editables+=("${editable}")
done < "${PATHS}"

file_count="$(tr -cd '\0' < "${LIST}" | wc -c | tr -d ' ')"
(( file_count > 0 )) || fail "the editable surface has no files"

# Archive the editable paths themselves (directory entries and any .git included),
# as `yukon submit` does. GNU tar (CI) also gets portable, mtime-free headers.
tar_flags=()
if tar --version 2>/dev/null | grep -q 'GNU tar'; then
  tar_flags+=(--format=ustar --owner=0 --group=0 --numeric-owner --mtime=@0 --sort=name)
fi
tar -czf "${ARCHIVE}" ${tar_flags[@]+"${tar_flags[@]}"} -- "${editables[@]}" || fail "tar failed"
bytes="$(wc -c < "${ARCHIVE}" | tr -d ' ')"
echo "editable archive: ${bytes} bytes (${file_count} files; limit ${MAX_BYTES})"
(( bytes <= MAX_BYTES )) || fail "archive exceeds ${MAX_BYTES} bytes"
echo "check-editable-archive-size: PASS"

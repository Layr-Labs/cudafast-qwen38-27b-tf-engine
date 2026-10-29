#!/usr/bin/env bash
set -euo pipefail

die() { printf 'new-track.sh: %s\n' "$*" >&2; exit 1; }

TRACK_ID=""
FORK_SHA=""
CHECKPOINT=""
BENCH_COMMIT=""
OS_TOKEN=""
SNAPSHOT_DIR=""
HASH_NON_LFS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track-id) TRACK_ID="${2:-}"; shift 2 ;;
    --fork-sha) FORK_SHA="${2:-}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:-}"; shift 2 ;;
    --bench-commit) BENCH_COMMIT="${2:-}"; shift 2 ;;
    --os) OS_TOKEN="${2:-}"; shift 2 ;;
    --snapshot-dir) SNAPSHOT_DIR="${2:-}"; shift 2 ;;
    --hash-non-lfs) HASH_NON_LFS=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "${TRACK_ID}" =~ ^[a-z0-9.]+-[a-z0-9-]+-(mlx|cuda)-v[0-9]+$ ]] \
  || die "--track-id is invalid"
[[ "${FORK_SHA}" =~ ^[0-9a-f]{40}$ ]] || die "--fork-sha must be 40 lowercase hex"
[[ "${CHECKPOINT}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+@[0-9a-f]{40}$ ]] \
  || die "--checkpoint must be repo@40-hex-revision"
[[ "${BENCH_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || die "--bench-commit must be 40 lowercase hex"
[[ "${OS_TOKEN}" == "macOS" || "${OS_TOKEN}" == "Linux" ]] \
  || die "--os must be macOS or Linux"
[[ -n "${SNAPSHOT_DIR}" ]] || die "--snapshot-dir is required for offline stamping"
[[ -d "${SNAPSHOT_DIR}" ]] || die "snapshot directory is missing: ${SNAPSHOT_DIR}"
[[ "${HASH_NON_LFS}" == "1" ]] || die "--hash-non-lfs is required with --snapshot-dir"

command -v python3 >/dev/null 2>&1 || die "python3 is required"
command -v jq >/dev/null 2>&1 || die "jq is required"

OLD_CONTRACT="$(jq -er '.contractPath' benchmark.json)"
NEW_STEM="$(printf '%s' "${TRACK_ID}" | tr '.-' '__')_track"
NEW_CONTRACT="fixtures/${NEW_STEM}.json"
MODEL_PART="$(printf '%s' "${TRACK_ID}" | sed -E 's/-(mlx|cuda)-v[0-9]+$//' | tr -d '.')"
PLATFORM="$(printf '%s' "${TRACK_ID}" | sed -E 's/^.*-(mlx|cuda)-v[0-9]+$/\1/')"
if [[ "${PLATFORM}" == "cuda" ]]; then
  NEW_NAME="cudafast-${MODEL_PART}"
else
  NEW_NAME="mlxfast-${MODEL_PART}"
fi

python3 - "${OLD_CONTRACT}" "${NEW_CONTRACT}" "${TRACK_ID}" "${NEW_NAME}" \
  "${FORK_SHA}" "${CHECKPOINT}" "${SNAPSHOT_DIR}" "${BENCH_COMMIT}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

old_contract, new_contract, track_id, name, fork_sha, checkpoint, snapshot, bench_commit = sys.argv[1:]
repo, revision = checkpoint.rsplit("@", 1)

with open("benchmark.json", encoding="utf-8") as file:
    benchmark = json.load(file)
benchmark["name"] = name
benchmark["trackId"] = track_id
benchmark["staticReviewTrackId"] = track_id
benchmark["contractPath"] = new_contract
benchmark.setdefault("leaderboard", {})["namespace"] = track_id
with open("benchmark.json", "w", encoding="utf-8") as file:
    json.dump(benchmark, file, indent=2)
    file.write("\n")

with open(old_contract, encoding="utf-8") as file:
    contract = json.load(file)
contract["track_id"] = track_id
contract["track_name"] = f"{track_id} -- composite paired speedup"
contract["benchmark_name"] = name
contract["official_scoring_enabled"] = False
contract["tensorfold_revision"] = fork_sha
contract["baseline_reference_commit"] = "UNRECORDED"

target = contract.setdefault("target", {})
target["upstream_model_id"] = repo
target["upstream_revision"] = revision

omit_names = {".gitattributes", "license", "license.md", "license.txt", "notice"}
omit_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
records = []
for path in sorted(pathlib.Path(snapshot).iterdir(), key=lambda item: item.name):
    name_part = path.name.lower()
    if name_part in omit_names or path.suffix.lower() in omit_suffixes or not path.is_file():
        continue
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    records.append({"path": path.name, "bytes": size, "sha256": digest.hexdigest()})
if not records:
    raise SystemExit("new-track.sh: local snapshot has no files to pin")

manifest_path = target.get("manifest_path")
has_files = "files" in target
if not has_files and not manifest_path:
    raise SystemExit("new-track.sh: target must contain files or manifest_path")
if has_files:
    target["files"] = records
if manifest_path:
    manifest = pathlib.Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(f'{record["sha256"]} {record["bytes"]} {record["path"]}\n')

with open(new_contract, "w", encoding="utf-8") as file:
    json.dump(contract, file, indent=2)
    file.write("\n")
if old_contract != new_contract:
    os.unlink(old_contract)

fetcher = pathlib.Path("tools/fetch-benchd.sh")
if fetcher.exists():
    text = fetcher.read_text(encoding="utf-8")
    lines = []
    changed = False
    for line in text.splitlines():
        if line.startswith("PINNED_SOURCE_COMMIT="):
            line = f'PINNED_SOURCE_COMMIT="{bench_commit}"'
            changed = True
        lines.append(line)
    if not changed:
        lines.insert(2, f'PINNED_SOURCE_COMMIT="{bench_commit}"')
    fetcher.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

printf 'new-track.sh: stamped %s from local snapshot %s\n' "${TRACK_ID}" "${SNAPSHOT_DIR}"

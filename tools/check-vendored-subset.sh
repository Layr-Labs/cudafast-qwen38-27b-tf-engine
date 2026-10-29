#!/usr/bin/env bash
# check-vendored-subset.sh -- ON-DEMAND (not a CI step: Vendor/TensorFold is the
# participant editable surface, so later submissions may change it). One-time proof
# for the prune commit that the vendored TensorFold tree is an ENGINE SUBSET of the
# reference tree, never a modified copy.
#
# Against the fixture's baseline_reference_commit (the organizer's reference tree):
#   1. Vendor/TensorFold only LOSES files: no file is added, copied, modified,
#      renamed or changed in type (git diff --diff-filter=ACMRT is empty);
#   2. Vendor/TensorFold/zig, build.zig and build.zig.zon are byte-equal, so the
#      engine sources and the build graph are the reference's.
# HEAD is the commit under test. Needs the reference commit in the checkout
# (ci.yml checks out with full history).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
FIXTURE="${ROOT}/fixtures/qwen3_8_27b_tf_cuda_v1_track.json"
REF="${VENDORED_SUBSET_REF:-$(jq -r '.baseline_reference_commit // empty' "${FIXTURE}")}"
fail() { printf 'check-vendored-subset: FAIL: %s\n' "$1" >&2; exit 1; }

[[ "${REF}" =~ ^[0-9a-f]{40}$ ]] || fail "the fixture names no baseline_reference_commit"
git -C "${ROOT}" cat-file -e "${REF}^{commit}" 2>/dev/null \
  || fail "reference commit ${REF} is not in this checkout (fetch full history)"

changed="$(git -C "${ROOT}" diff --diff-filter=ACMRT --name-only "${REF}" HEAD -- Vendor/TensorFold)"
[[ -z "${changed}" ]] \
  || fail "Vendor/TensorFold adds or changes files against ${REF:0:12}: $(printf '%s\n' "${changed}" | head -5 | tr '\n' ' ')"

for path in Vendor/TensorFold/zig Vendor/TensorFold/build.zig Vendor/TensorFold/build.zig.zon; do
  git -C "${ROOT}" diff --quiet "${REF}" HEAD -- "${path}" \
    || fail "${path} differs from ${REF:0:12}"
done

removed="$(git -C "${ROOT}" diff --diff-filter=D --name-only "${REF}" HEAD -- Vendor/TensorFold | wc -l | tr -d ' ')"
echo "check-vendored-subset: PASS: Vendor/TensorFold is a deletion-only subset of ${REF:0:12} (${removed} files left out); zig/, build.zig, build.zig.zon are byte-equal"

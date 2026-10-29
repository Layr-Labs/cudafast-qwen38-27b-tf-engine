# Agent guide

Track id: `qwen3.8-27b-tf-cuda-v1`.

Read `README.md` first. Read `TASK.md` second.

## Goal

Make Qwen 3.8 27B TensorFold faster on one DGX Spark GB10.

Use the native TensorFold CUDA engine.

Keep output within the track correctness limit.

## Edit limit

- Participants may edit only `Vendor/TensorFold`.

- `benchmark.json` is the authority.

- Control-plane files are trusted.

- Keep the target weight format unchanged.

- Keep one request active at a time.

- Do not reuse state across benchmark phases.

- The organizer fixes the candidate at DFlash depth 1 with the pinned drafter
  in `mtp-head.manifest.json`; serial is the control leg.

- Entrants change only `Vendor/TensorFold` and keep both legs reproducible.

## Build

```bash
export TENSORFOLD_ZIG=/path/to/zig-0.17.0/zig
export MLXFAST_TARGET_SNAPSHOT_DIR=/path/to/target
./setup.sh
```

The supported build runs in the pinned container and stages the worker under
`.build/release/`.

## Checks

```bash
set -e
for t in tools/test-*.sh; do bash "$t"; done
bash tools/box-runner/test/test-box-runner.sh
python3 tools/lint-benchmark-manifest.py
python3 tools/check-doc-consistency.py
jq . fixtures/qwen3_8_27b_tf_cuda_v1_track.json
```

GPU tests run only on the ranked Linux box.

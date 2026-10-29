# Third-party notices

## Model artifacts

The repository stores model identifiers and pinned revisions. It does not
distribute model weights.

| Artifact | Model id | Revision | Base model | Source | License |
|---|---|---|---|---|---|
| Target, by TensorFold | `TensorFold/Qwen3.8-27B-MLX-4bit` | `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c` | `Qwen/Qwen3.8-27B` | [Pinned Hugging Face tree](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit/tree/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c) | Apache-2.0 |
| DFlash drafter, by z-lab | `z-lab/Qwen3.8-27B-DFlash2` | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | `Qwen/Qwen3.8-27B` | [Pinned Hugging Face tree](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) | Apache-2.0 |

## TensorFold

This repository includes the vendored ENGINE SUBSET of
[`ashhart/TensorFold`](https://github.com/ashhart/TensorFold) @
[`eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab`](https://github.com/ashhart/TensorFold/commit/eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab)
(v1.1.6), Apache-2.0. Upstream files that no kept build, test or tool input
reads (most of `app/`, `assets/` and `docs/`) are not included because the
engine build does not use them; they remain in the upstream repository at that
revision.

### Changes to the vendored subset

Compared with upstream revision
`eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab`, this repository changes 17
upstream files and adds 17 files. Every other included file is unchanged from
that revision. Each changed file starts with a one-line notice that it was
modified.

The changes add the Engine Protocol v1 bench worker and its build steps, a
bench session in the engine API and lanes, a DFlash audit journal, page and
pinned-memory release at phase drains in the Qwen CUDA engine, per-request
draft control in the native Qwen 27B lanes, and process-memory reporting on
Linux.

Changed files (paths under `Vendor/TensorFold/`):

- `build.zig`
- `zig/src/core/engine_api.zig`
- `zig/src/core/lanes/engine.zig`
- `zig/src/core/lanes/fake.zig`
- `zig/src/core/lanes/logprob.zig`
- `zig/src/cuda/copy_source.zig`
- `zig/src/cuda/copy_source_spare_test.zig`
- `zig/src/families/qwen27/dflash/generation.zig`
- `zig/src/families/qwen27/dflash/runtime_frame.zig`
- `zig/src/families/qwen3_5/cuda_engine.zig`
- `zig/src/families/qwen3_5/cuda_native.zig`
- `zig/src/families/qwen3_5/cuda_pages.zig`
- `zig/src/families/qwen3_5/cuda_pages_lifetime_test.zig`
- `zig/src/native/cuda_drafted.zig`
- `zig/src/native/qwen27_lanes.zig`
- `zig/src/native/qwen27_slots.zig`
- `zig/src/server/process_memory.zig`

Added files:

- `zig/build/bench_session_host.zig`
- `zig/build/dflash_audit_host.zig`
- `zig/src/bench/main.zig`
- `zig/src/bench/protocol.zig`
- `zig/src/bench/resident.zig`
- `zig/src/bench/resident_client.zig`
- `zig/src/bench/resident_test.zig`
- `zig/src/bench/self_test_test.zig`
- `zig/src/bench/worker.zig`
- `zig/src/bench/worker_dflash_test.zig`
- `zig/src/bench/worker_test.zig`
- `zig/src/core/bench_session.zig`
- `zig/src/core/bench_session_engine.zig`
- `zig/src/core/bench_session_test.zig`
- `zig/src/families/qwen27/dflash/audit.zig`
- `zig/src/families/qwen27/dflash/audit_test.zig`
- `zig/src/families/qwen3_5/cuda_native_phase_test.zig`

TensorFold is Apache-2.0. Its license, notice, and third-party notices remain
in `Vendor/TensorFold`:

- [`Vendor/TensorFold/LICENSE`](Vendor/TensorFold/LICENSE)
- [`Vendor/TensorFold/NOTICE`](Vendor/TensorFold/NOTICE)
- [`Vendor/TensorFold/THIRD_PARTY_NOTICES.md`](Vendor/TensorFold/THIRD_PARTY_NOTICES.md)

The root `LICENSE` applies to this track repository.

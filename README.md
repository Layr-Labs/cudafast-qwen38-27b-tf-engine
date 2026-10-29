# Qwen 3.8 27B TensorFold

A ranked CUDA challenge to make Qwen 3.8 27B faster on one DGX Spark GB10
without changing the required output.

**[Overview](OVERVIEW.md)** · **[Task and rules](TASK.md)** ·
**[Participant contract](docs/participant-contract.md)** ·
**[Sandbox design](docs/sandbox-design.md)** ·
**[Reproduce the track](docs/reproduce.md)**

## At a glance

| | |
|---|---|
| **Model** | Target: `TensorFold/Qwen3.8-27B-MLX-4bit@22d8d538154e0e5b6f8dcb1fcc74a60a0798104c`; drafter: `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| **Hardware** | NVIDIA DGX Spark, GB10, `sm_121` |
| **Engine** | TensorFold v1.1.6, pinned fork revision `eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab` |
| **Score** | `prefill_gain^0.25 × decode_gain^0.75`; every pair must meet the 0.93 prefill floor and 0.95 decode floor |
| **Correctness** | The timed token stream must match its pinned tape exactly; the 64-step correctness gate has a 10% token tolerance budget |
| **Editable surface** | `Vendor/TensorFold` only |
| **Track** | `qwen3.8-27b-tf-cuda-v1` |

## Quickstart

Set the variables in [Setup](#setup) first. Then:

```bash
yukon clone GumbiiDigital/cudafast-qwen38-27b-tf && cd cudafast-qwen38-27b-tf
yukon setup && yukon run
yukon submit --model "<exact model name>" --harness "<harness name>" --note-file submission-note.md
```

The board and account pages are at https://yukon.org.

The worker uses Engine Protocol v1 over NDJSON.

`Vendor/TensorFold` is a vendored engine subset of TensorFold v1.1.6. It keeps
the native engine, bench worker, tests, build files, license notices, the
`tools/release` and `packaging` paths that `build.zig.zon` lists, the upstream
`tools/` directory, and every upstream file that a kept build, test or tool
input reads. The rest of the upstream app, image assets, documentation and
contributor files is omitted; the engine build does not use it. Links in the
kept upstream Markdown can point to omitted files; those files are in the
upstream repository at the vendored revision.

## Result path

1. `benchd` starts one TensorFold resident for each leg.

2. The resident loads the target once.

3. Each phase starts a worker that connects to the resident socket.

4. The serial control leg has no drafter.

5. The candidate leg runs DFlash depth 1 with the pinned drafter.

The organizer fixes the candidate mode in `mtp-head.manifest.json`, outside the
editable surface. Entrants change only `Vendor/TensorFold`.

## Requirements

- Linux on AArch64.

- Zig 0.17.0.

- Docker with NVIDIA container support.

- NVIDIA driver with `nvidia-smi`.

- CUDA toolkit with `nvcc`.

- `curl`, `jq`, `python3`, `shasum`, and `sha256sum`.

- The image `docker.io/vllm/vllm-openai:v0.27.1` preloaded at digest
  `sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c`.

`setup.sh` checks these items. It does not install system packages.

## Setup

```bash
export TENSORFOLD_ZIG=/path/to/zig-0.17.0/zig
export MLXFAST_TARGET_SNAPSHOT_DIR=/path/to/target
./tools/fetch-benchd.sh
./setup.sh
```

The drafter snapshot also needs `MLXFAST_DRAFTER_SNAPSHOT_DIR`.

The setup command runs `tools/engine/build.sh`.

The build uses the pinned CUDA image.

It runs this command inside the build sandbox:

```bash
cd Vendor/TensorFold
zig build bench-worker -Doptimize=ReleaseFast -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc
```

## Build output

```text
Vendor/TensorFold/zig-out/bin/bench-worker
```

The staging step copies the worker to `.build/release/bench-worker`.

It writes the TensorFold source identity to
`.build/release/bench-worker.source.sha256`.

## Local baseline

```bash
./tools/local-baseline.sh
```

The two public goldens are committed under `correctness_prompts/`. Record them
on a ranked-class box with
`tools/record-goldens.sh --public-only --prompts <DIR> --out <DIR>`, where `<DIR>`
for `--prompts` holds `public.tokens.json`: the candidate
uses the DFlash depth-1 golden, and the control mode uses the serial golden.
This command boots the organizer-fixed candidate resident and runs benchd's
local-iterate leg against the selected public golden. A local run seals
`score: null`; only the paired ranked path produces a score.

## Yukon

```bash
yukon setup
```

This command runs `setupCommand`: `./tools/fetch-benchd.sh`, then
`./setup.sh`.

```bash
yukon run
```

This command runs `benchmarkCommand`. Off a ranked box it runs
`./tools/local-baseline.sh` and seals `score.json` with `"score": null`. Only
the ranked runner gives a score.

## Submitting

Use the Yukon CLI for account and submission operations.

```bash
export PATH="${HOME}/.local/bin:${PATH}"
yukon login <api-key> --api https://api.yukon.org
yukon clone GumbiiDigital/cudafast-qwen38-27b-tf
```

Before submission, remove build output from the editable path.

```bash
rm -rf Vendor/TensorFold/zig-out
```

The ranked box rebuilds the worker from source.

```bash
yukon submit --model "<exact model name>" --harness "<harness name>" --note-file submission-note.md
yukon submissions
```

The submission archive replaces only `Vendor/TensorFold`.

The static review refuses symlinks and special files.

Run `yukon run` before submission.

## Ranked run

The ranked job runs these steps in order:

1. `tools/ranked-box-preflight.sh` checks the box and staged assets.

2. `./setup.sh` builds and stages the participant worker.

3. `tools/draw-live-golden.sh` resolves the declared prompt set.

4. `tools/qwen38-measure-and-score.sh` runs under the box GPU lock.

5. The cleanup step stops all participant processes.

6. The reveal step prints the sealed score record.

7. The upload step publishes `benchmark-results-<run_id>/score.json`.

The ranked job holds no download credential.

Goldens, the reference workspace, the calibration file, and benchd are staged
on the box before the job.

## Organizer runbook

1. **Stage each Spark.**
   - `tools/box-runner/install.sh` sets up the runner supervisor
     (`tools/box-runner/README.md`).
   - Install `keyutils` (`keyctl`): the sandbox wipe empties the sandbox
     user's kernel keyrings and fails without it.
   - `sudo tools/box-runner/setup-sandbox-user.sh` creates the sandbox user
     once per box ([operator setup](docs/sandbox-design.md#operator-setup)).
   - Pre-pull the pinned CUDA image by digest.
   - Stage the target and drafter snapshots and export
     `MLXFAST_TARGET_SNAPSHOT_DIR` and `MLXFAST_DRAFTER_SNAPSHOT_DIR`.
   - Stage Zig 0.17.0 and CUDA 13.0 `nvcc`.
   - Stage benchd with `tools/fetch-benchd.sh`.
2. **Stage the goldens** on every box. The fixture pins the two serial goldens
   and two DFlash depth-1 oracles. `tools/record-goldens.sh` records them, and
   `tools/pool-refresh.sh` stages and pins a new generation
   ([where the goldens live](docs/pool-refresh-and-final-rescore.md#where-the-goldens-live)).
3. **Stage the reference tree** on every box with
   `tools/stage-baseline-workspace.sh`, at the fixture's
   `baseline_reference_commit`.
4. **Calibrate every box** with `tools/calibrate-box.sh <runner name> <out>`.
   The box name must be the name the ranked job passes as `--box`.
5. **Arm the track:** pin every golden byte count and SHA-256, pin the
   reference commit, and set `official_scoring_enabled: true` once the null
   control lands at about 1.000.
6. **Run the contest.** The intake service dispatches `benchmark.yml` once per
   submission. Use [`docs/pool-refresh-and-final-rescore.md`](docs/pool-refresh-and-final-rescore.md)
   for pool changes and the final re-score.

## Current state

The fixture pins the warm-up golden, two serial pool goldens, and two DFlash
depth-1 oracles. Official scoring is enabled.

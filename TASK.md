# Task

Improve the native TensorFold CUDA engine for Qwen 3.8 27B TensorFold.

## Fixed facts

- Track: `qwen3.8-27b-tf-cuda-v1`.

- Hardware: one DGX Spark GB10.

- Batch size: 1.

- Official pairs: 2.

- Decode floor: 0.95.

- Prefill floor: 0.93.

- Candidate draft mode: DFlash depth 1 with the pinned drafter.

- Control mode: serial.

- Target revision: `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c`.

- Drafter revision: `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.

## Scoring

```text
composite = prefill_gain^0.25 * decode_gain^0.75
```

Each candidate leg runs DFlash depth 1 beside a serial control leg.

The organizer sets the candidate mode in `mtp-head.manifest.json`. That file is
outside the editable surface; entrants change only `Vendor/TensorFold`.

Both legs use the same prompt.

Each decode gain must be at least 0.95. Each prefill gain must be at least 0.93.

## Rules

- Edit only `Vendor/TensorFold` for a submission.

- Keep the checkpoint format unchanged.

- Do not cache work across phases.

- Do not move prefill work outside its measured phase.

- Return raw counters only.

- Match the pinned token tape in the timed window.

Read the full [participant contract](docs/participant-contract.md) before a
submission.

## Where to start

| Goal | Path |
|---|---|
| Understand the run | [`README.md`](README.md) and [`OVERVIEW.md`](OVERVIEW.md) |
| Read the engine guide | [`Vendor/TensorFold/README.md`](Vendor/TensorFold/README.md) |
| Find the Qwen implementation | `Vendor/TensorFold/zig/src/families/qwen3_5/` |
| Build the supported path | `./setup.sh` |
| Run offline checks | [`docs/reproduce.md`](docs/reproduce.md) |

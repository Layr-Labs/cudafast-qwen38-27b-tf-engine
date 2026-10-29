# Qwen 3.8 27B TensorFold overview

Track id: `qwen3.8-27b-tf-cuda-v1`.

The target is a 64-layer Qwen3.8 text model.

It uses 48 linear-attention layers and 16 full-attention layers.

The target uses affine 4-bit weights in groups of 64.

TensorFold runs the model with native CUDA kernels.

The benchmark measures prefill and a 128-token decode window.

One warm-up prefill runs before one timed prefill.

Each pair scores:

```text
prefill_gain^0.25 * decode_gain^0.75
```

Each pair must meet a 0.93 prefill floor and a 0.95 decode floor. The run score
is the mean of two paired results.

The candidate leg runs DFlash depth 1 with the pinned drafter. The organizer
sets this mode in `mtp-head.manifest.json`, outside the editable surface. The
serial mode is the control leg.

The timed token stream must match its pinned tape exactly. The wider 64-step
correctness gate has a 10% token tolerance budget.

Participants may edit only `Vendor/TensorFold`.

The fixture holds all model facts and artifact pins.

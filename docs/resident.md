# Resident wire

The resident command is:

```bash
bench-worker resident --weights TARGET [--drafter DRAFTER] --socket PATH
```

The attached worker command is:

```bash
BENCH_WORKER_RESIDENT_SOCKET=PATH bench-worker runtime-worker --weights TARGET
```

The worker uses NDJSON on standard input and standard output.

The socket carries calls to the loaded resident.

The resident identity must stay fixed for all phases in one leg.

The serial control startup omits `--drafter`.

The organizer-fixed DFlash depth-1 candidate startup includes the pinned
drafter directory. `mtp-head.manifest.json` sets this mode outside the editable
surface; entrants change only `Vendor/TensorFold`.

The harness hash covers the trusted roots. The worker manifest digest records
the editable TensorFold engine identity.

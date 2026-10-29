# Reproduce the track

## Offline checks

```bash
set -e
for t in tools/test-*.sh; do bash "$t"; done
bash tools/box-runner/test/test-box-runner.sh
python3 tools/check-doc-consistency.py
python3 tools/lint-benchmark-manifest.py --gitlink-targets report
```

These checks need no GPU, network, or model weights.

## CUDA build

```bash
export TENSORFOLD_ZIG=/path/to/zig-0.17.0/zig
export MLXFAST_TARGET_SNAPSHOT_DIR=/path/to/target
./setup.sh
```

## Record and stage goldens

The organizer fixes the candidate at DFlash depth 1 with the pinned drafter in
`mtp-head.manifest.json`. Serial is the control leg. The manifest is outside
the editable surface; entrants change only `Vendor/TensorFold`.

The two public goldens are committed under `correctness_prompts/`. Record them
on a ranked-class box with:

```bash
tools/record-goldens.sh --prompts DIR --out DIR --public-only
```

The candidate uses `public-longcopy-gate-english-496.dflash1.golden.json`.
The control mode uses `public-longcopy-gate-english-496.golden.json`.

1. Stage two tokenized prompt files as `pool-1.tokens.json` and `pool-2.tokens.json`.

2. Stage the pinned target and drafter directories.

3. Run `tools/record-goldens.sh --prompts DIR --out DIR --apply` on the GB10.

4. Record the separate warm-up prompt through the same Engine Protocol path.

5. Upload and pin the warm-up file, both serial files, and both
   `.dflash1.golden.json` files by byte count and SHA-256.

6. Use `tools/fetch-goldens.sh --all --out DIR` to stage every pinned file on
   each ranked box.

7. Run `tools/ranked-box-preflight.sh`. It verifies the warm-up file and every
   live serial and DFlash depth-1 file against the fixture.

The current fixture contains all five pins and enables official scoring.

## Recalibrate a box

After a prompt or timing-method change, run:

```bash
tools/calibrate-box.sh <runner-name> <output-file> --passes 3
```

Install the new file at `MLXFAST_BASELINE_CALIBRATION` only after the command
passes. The calibration belongs to one runner name and one reference commit.

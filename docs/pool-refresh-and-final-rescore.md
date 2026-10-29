# Hidden-pool refresh and final re-score

This repository ships the pool-refresh tool. It does not ship a final
re-score dispatcher. A final re-score needs a separate, reviewed dispatch
implementation before an operator can run it.

## Current scoring shape

The fixture is the authority:

- two live prompts: `pool-1` and `pool-2`;
- two official control/candidate pairs;
- one serial golden and one DFlash depth-1 oracle per prompt;
- one separate warm-up golden;
- the candidate runs DFlash depth 1 with the pinned drafter;
- the organizer sets that mode in `mtp-head.manifest.json`, outside the
  editable surface; entrants change only `Vendor/TensorFold`;
- serial is the control leg.

Each ranked job measures both prompts. It scores the arithmetic mean of the two
pair composites.

## Why refresh the pool

A candidate sees the scored prompt while it runs. Rotating the hidden prompts
limits the value of tuning to one exposed prompt. A refresh keeps the old and
new generations separate until the switch, so an in-flight job can finish with
the fixture revision it started with.

## 1. Author the next generation

`tools/pool-refresh.sh` accepts organizer-written chat prompts through
`--texts`, or stitched prompts through `--corpus`. The default count comes from
the current fixture. For this track it is two. Set the required box list before
any command, including an author-only step.

```bash
export REFRESH_BOXES="<box-a> <box-b>"
tools/pool-refresh.sh --label g2 --dry-run --texts <text-dir> --snapshot <snapshot>
tools/pool-refresh.sh --label g2 --step author --texts <text-dir> --snapshot <snapshot>
```

The private working state stays under `.mlxfast-private/refresh/<label>/`.
That directory is ignored by Git.

## 2. Record the hidden goldens

`REFRESH_BOXES` must contain at least two SSH targets. The tool records every
golden twice and requires the two copies to be byte-identical.

```bash
export REFRESH_BOX_REPO="<checkout-relative-to-home>"
export REFRESH_BOX_GOLDENS="<golden-directory>"
export REFRESH_BOX_SNAPSHOT="<snapshot-directory>"
tools/pool-refresh.sh --label g2 --step record
```

The record step produces, for each prompt, one serial file and one
`.dflash1.golden.json` file. It uses the shared GPU lock
`/tmp/mtplx-gpu-exclusive.lock` unless `REFRESH_GPU_LOCK` overrides it.

The separate warm-up golden is not part of a refresh generation. If the
warm-up prompt or timing method changes, record that golden separately, update
its byte count and SHA-256 in the fixture, and stage it with the pool files.

## 3. Stage, calibrate, and switch

Stage the new generation without removing the old one:

```bash
tools/pool-refresh.sh --label g2 --step stage
```

Create a proposed fixture and calibrate every ranked box against it:

```bash
cp fixtures/qwen3_8_27b_tf_cuda_v1_track.json /tmp/next-fixture.json
tools/pool-refresh.sh pin-fixture /tmp/next-fixture.json g2 \
  .mlxfast-private/refresh/g2/pins.json
tools/calibrate-box.sh <runner-name> <new-file> --fixture /tmp/next-fixture.json --passes 3
```

After each box has a passing calibration, pin the generation:

```bash
tools/pool-refresh.sh --label g2 --step pin
```

The command edits the trusted fixture and prints the review commands. It does
not commit or push. Review the fixture diff, drain old jobs, install the new
calibration files, and publish the fixture through the repository's normal
review process.

Verify every box after the published fixture is available:

```bash
tools/pool-refresh.sh --label g2 --step verify
```

The ranked preflight checks the staged warm-up golden, every live serial
golden, every DFlash depth-1 oracle, the reference workspace, and the
box-specific calibration.

## 4. Retire the old generation

First unpin the old label. After every job dispatched with the old fixture has
finished, run the same command again to remove the old files from the boxes:

```bash
tools/pool-refresh.sh --label g1 --step retire
```

Keep the private archived copy for audit and recovery.

## Where the goldens live

| Copy | Holds | Purpose |
|---|---|---|
| Ranked box, `REFRESH_BOX_GOLDENS` | Warm-up file plus current serial and DFlash depth-1 files | The only copy read by scoring |
| Organizer private state, `REFRESH_PRIVATE` | Prompts, duplicate recordings, pins, and progress | Resume, audit, and recovery |
| Private archive | Current and retired generations | Backup and dispute evidence |

The hidden pool and warm-up goldens are not committed to this repository; only
the two public goldens under `correctness_prompts/` are. Verify every staged
file by byte count and SHA-256 before a ranked run.

## Final re-score boundary

`tools/pool-refresh.sh` can author and record a label named `final`. Those
files must stay off ranked boxes before the contest freeze.

This tree has no command that freezes submissions, selects finalists,
dispatches held-out runs, retries infrastructure failures, or produces a final
ranking. Do not present the `final` recording as a completed re-score. Add and
review that control-plane implementation before using this section as an
operator runbook.

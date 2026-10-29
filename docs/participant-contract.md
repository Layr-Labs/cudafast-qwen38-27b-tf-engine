# Participant contract

Track id: `qwen3.8-27b-tf-cuda-v1`.

## Editable code

Only `Vendor/TensorFold` is editable in a submission.

The static review rejects any other submitted path.

## Engine state

One resident loads weights once for each benchmark leg.

Each phase uses a new worker connection.

The resident clears model state at each phase boundary.

It must not keep prompt or decode work from an earlier phase.

The engine must not carry state from one job to a later job through any side
channel, such as the page cache or file access times of shared host files. The
sandbox cleanup does not clear those, so promoted entries are reviewed. An
entry that carries state across jobs is removed.

## Correctness

The timed token stream must match its pinned tape.

The wider correctness gate has a 10 percent token budget.

The worker reports raw work counters.

The benchmark process owns all timing.

## Draft mode

The serial control leg starts without a drafter directory.

The candidate leg runs DFlash depth 1 with the pinned drafter.

No other draft depth is allowed.

The engine must really run DFlash depth 1 with the pinned drafter in the
candidate leg. It must not use another depth and must not bypass the drafter.
The token stream cannot show the depth, so promoted entries are reviewed. An
entry that uses another depth or bypasses the drafter is removed.

The organizer sets this mode in `mtp-head.manifest.json`. The manifest is
outside the editable surface. Participants change only `Vendor/TensorFold`.

## Score

Each run has 2 control and candidate pairs.

The decode speed floor is 0.95.
The prefill speed floor is 0.93.
The candidate uses DFlash mode at depth 1.

The composite weights prefill by 0.25 and decode by 0.75.

## How the score is measured

Higher is better.

The score is a unitless speedup against the live serial control leg.

Each ranked run has 2 pairs.

Each pair has one control leg and one candidate leg on the same prompt.

Each leg first runs one unmeasured warm-up window.

The warm-up window uses the pinned warm-up golden and 128 decode steps.

The measured window has 1 unmeasured prefill pass.

It then has 1 timed prefill pass and one 128-step decode window.

The prefill value is the median timed wall time divided by the prompt token count.

With 1 timed pass, the median is that pass.

The decode value includes the seed-prefill time and the decode time.

Benchd divides this time by the 128 committed decode tokens.

For each pair:

```text
prefill_gain = control prefill s/token / candidate prefill s/token
decode_gain = control decode s/token / candidate decode s/token
pair_score = prefill_gain^0.25 * decode_gain^0.75
```

The run score is the arithmetic mean of the 2 pair scores.

It is not the composite of the mean gains.

Each pair must pass both speed floors (decode 0.95, prefill 0.93).

The box calibration sample CV must be at most 2.0% on each axis.

Each control prefill can be at most 5% slower than its prompt calibration mean.

Each control decode can be at most 3% slower than its prompt calibration mean.

A faster control leg has no lower-band refusal.

Over-budget calibration variance or a control-band failure writes no score.

The timed prefill and decode tokens must match the pinned tapes exactly.

The full correctness gate checks 64 steps with a 10% token tolerance budget.

Any timed mismatch or correctness failure writes no score.

Benchd applies no fixed decimal rounding to the score.

It reports the score at full `f64` precision.

Diagnostic values can be coarsened, but the score is not coarsened.

A new submission must beat the promoted score by 200 basis points (200 bips).

An equal score does not promote.

<!--
Pinned sources for this section:
- fixtures/qwen3_8_27b_tf_cuda_v1_track.json:7-8,15-18,23,27-39,45-49
- Layr-Labs/benchd@3101418f95eb074cc9c100f8dc61cc85cc924155:crates/bench-core/src/score.rs:22-50,53-81
- Layr-Labs/benchd@3101418f95eb074cc9c100f8dc61cc85cc924155:crates/benchd/src/official.rs:393-410,512-589,1110-1131,1389-1515,1573-1578,1936-1954,2017-2055,2112-2149,2435-2665
- Layr-Labs/benchd@3101418f95eb074cc9c100f8dc61cc85cc924155:crates/benchd/src/main.rs:643-653,674-680,688-691,732-745,6936-6971
-->

## Guards that can refuse a run

The decode speedup ceiling is 5.0. The prefill speedup ceiling is 2.0.
A pair above a ceiling refuses the run.

The decode time includes the seed step.

The seed window guard is 1.7.
For each leg, benchd divides the seed time per token by the prefill time per token of the same leg.
A ratio above 1.7 refuses the run.
An engine that makes prefill much faster but keeps the seed step slow can fail this guard.
Make the seed step as fast as prefill.

All participant files are cleared between legs and between pairs.
Do not expect a file or a cache to survive from one leg to the next.

The serial control leg must stay inside the box's calibration band.
A background OS job on a box can slow one measurement; this was seen in 1 of 17 prod runs.
A pair-1 case is retried once. Two in a row end the job as an infrastructure outcome.
A pair-2 case fails that submission, and the submitter can submit again.

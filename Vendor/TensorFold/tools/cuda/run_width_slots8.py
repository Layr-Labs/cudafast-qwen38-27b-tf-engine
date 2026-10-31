#!/usr/bin/env python3
"""Measure wider exact recurrence geometry and eliminate redundant slot reloads."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
out = Path(os.environ['Q38_OUT'])
# Keep the served family's own flags, independently of other families' glue presets.
assert '.{ .name = "qwen35", .flags = &.{"-O3"} }' in (root / 'zig/build/cuda_kernels.zig').read_text()
source = (root / 'zig/kernels/cuda/qwen35_slots.cuh').read_text()
source = source[source.index('template <int SLOTS, int R, bool REUSE'):source.index('#define TF_Q35_SLOTS')]
source = source.replace('bool REUSE = false>', 'bool REUSE = false, int WARPS = 4>')
source = source.replace('void q35_slots(', 'void q35_warp_slots(').replace('kept[4]', 'kept[WARPS]').replace('blockIdx.y * 4 + threadIdx.y', 'blockIdx.y * WARPS + threadIdx.y')
with tempfile.TemporaryDirectory(prefix='width-slots8-') as tmp:
    tmp = Path(tmp)
    (tmp / "width_warp_slots.cuh").write_text(source)
    command = ['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_121', '-Xptxas=-v',
               '-I' + str(tmp), '-I' + str(root / 'zig/kernels/cuda'),
               str(root / 'tools/cuda/width_slots8.cu'), '-o', str(tmp / 'slots')]
    with (out / 'slots8-compile.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((out / 'slots8-compile.log').read_text(), flush=True)
        raise SystemExit(result.returncode)
    with (out / 'slots8-samples.jsonl').open('w') as log:
        result = subprocess.run([str(tmp / 'slots')], stdout=log, stderr=subprocess.STDOUT)
    text = (out / 'slots8-samples.jsonl').read_text()
    print(text, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    samples = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    assert len(samples) == 1560 and all(s['mismatches'] == 0 for s in samples)
    summary = []
    keys = sorted(set((s['plan'], s['slots'], s['rows'], s['pn'], s['case']) for s in samples))
    for plan, slots, rows, pn, case in keys:
        selected = [s for s in samples if (s['plan'], s['slots'], s['rows'], s['pn'], s['case']) == (plan, slots, rows, pn, case)]
        summary.append(dict(plan=plan, slots=slots, rows=rows, pn=pn, case=case,
                            median_ms=statistics.median(s['ms'] for s in selected)))
    (out / 'slots8-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)

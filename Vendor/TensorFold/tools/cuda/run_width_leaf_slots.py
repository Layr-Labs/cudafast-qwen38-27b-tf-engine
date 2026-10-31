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
with tempfile.TemporaryDirectory(prefix='width-leaf-slots-') as tmp:
    tmp = Path(tmp)
    command = ['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_121', '-Xptxas=-v',
               '-I' + str(tmp), '-I' + str(root / 'zig/kernels/cuda'),
               str(root / 'tools/cuda/width_leaf_slots.cu'), '-o', str(tmp / 'slots')]
    with (out / 'leaf-slots-compile.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((out / 'leaf-slots-compile.log').read_text(), flush=True)
        raise SystemExit(result.returncode)
    with (out / 'leaf-slots-samples.jsonl').open('w') as log:
        result = subprocess.run([str(tmp / 'slots')], stdout=log, stderr=subprocess.STDOUT)
    text = (out / 'leaf-slots-samples.jsonl').read_text()
    print(text, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    samples = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    assert len(samples) == 624 and all(s['mismatches'] == 0 for s in samples)
    summary = []
    keys = sorted(set((s['plan'], s['slots'], s['leaf_slots'], s['pn'], s['case']) for s in samples))
    for plan, slots, leaf_slots, pn, case in keys:
        selected = [s for s in samples if (s['plan'], s['slots'], s['leaf_slots'], s['pn'], s['case']) == (plan, slots, leaf_slots, pn, case)]
        summary.append(dict(plan=plan, slots=slots, leaf_slots=leaf_slots, pn=pn, case=case,
                            median_ms=statistics.median(s['ms'] for s in selected)))
    (out / 'leaf-slots-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)

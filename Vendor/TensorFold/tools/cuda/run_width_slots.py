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
with tempfile.TemporaryDirectory(prefix='width-slots-') as tmp:
    tmp = Path(tmp)
    command = ['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_121', '-Xptxas=-v',
               '-I' + str(tmp), '-I' + str(root / 'zig/kernels/cuda'),
               str(root / 'tools/cuda/width_slots.cu'), '-o', str(tmp / 'slots')]
    with (out / 'slots-compile.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((out / 'slots-compile.log').read_text(), flush=True)
        raise SystemExit(result.returncode)
    with (out / 'slots-samples.jsonl').open('w') as log:
        result = subprocess.run([str(tmp / 'slots')], stdout=log, stderr=subprocess.STDOUT)
    text = (out / 'slots-samples.jsonl').read_text()
    print(text, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    samples = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    assert len(samples) == 780 and all(s['mismatches'] == 0 for s in samples)
    summary = []
    for slots in [2, 3, 4]:
        for rows in [16, 32]:
            for pn in [0, 2]:
                for case in ['parent', 'r1', 'r4', 'reuse2', 'reuse4']:
                    rows_ = [s for s in samples if (s['slots'], s['rows'], s['pn'], s['case']) == (slots, rows, pn, case)]
                    summary.append(dict(slots=slots, rows=rows, pn=pn, case=case,
                                        median_ms=statistics.median(s['ms'] for s in rows_)))
    (out / 'slots-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)

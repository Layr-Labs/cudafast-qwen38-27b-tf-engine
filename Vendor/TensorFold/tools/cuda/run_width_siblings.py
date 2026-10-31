#!/usr/bin/env python3
"""Measure exact leaf-sibling parallel recurrence with the native slot geometry."""
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
source = source.replace('void q35_slots(', 'void q35_slots_nested(')
with tempfile.TemporaryDirectory(prefix='width-siblings-') as tmp:
    tmp = Path(tmp)
    (tmp / "width_siblings_nested.cuh").write_text(source)
    command = ['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_121', '-Xptxas=-v',
               '-I' + str(tmp), '-I' + str(root / 'zig/kernels/cuda'),
               str(root / 'tools/cuda/width_siblings.cu'), '-o', str(tmp / 'slots')]
    with (out / 'siblings-compile.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((out / 'siblings-compile.log').read_text(), flush=True)
        raise SystemExit(result.returncode)
    with (out / 'siblings-samples.jsonl').open('w') as log:
        result = subprocess.run([str(tmp / 'slots')], stdout=log, stderr=subprocess.STDOUT)
    text = (out / 'siblings-samples.jsonl').read_text()
    print(text, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    samples = []
    pattern = None
    for line in text.splitlines():
        if not line.startswith('{'):
            continue
        row = json.loads(line)
        if 'pattern' in row:
            pattern = row['pattern']
        else:
            samples.append(dict(row, pattern=pattern))
    assert len(samples) == 1872 and all(s['mismatches'] == 0 for s in samples)
    summary = []
    for pattern in [0, 1, 2]:
        for slots in [3, 4]:
            for rows in [16, 32]:
                for pn in [0, 2]:
                    for case in ['parent', 'sibling2', 'nativebest', 'sibling1', 'nestedbest', 'hazard2']:
                        selected = [s for s in samples if (s['pattern'], s['slots'], s['rows'], s['pn'], s['case']) == (pattern, slots, rows, pn, case)]
                        summary.append(dict(pattern=pattern, slots=slots, rows=rows, pn=pn, case=case,
                                            median_ms=statistics.median(s['ms'] for s in selected)))
    (out / 'siblings-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)

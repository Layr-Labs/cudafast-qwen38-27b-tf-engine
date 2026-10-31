#!/usr/bin/env python3
"""Check grouped short-stream attention with the original tensor-core arithmetic."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
out = Path(os.environ['Q38_OUT'])
assert '.{ .name = "qwen35", .flags = &.{"-O3"} }' in (root / 'zig/build/cuda_kernels.zig').read_text()
with tempfile.TemporaryDirectory(prefix='width-attention-') as tmp:
    tmp = Path(tmp)
    command = ['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_121', '-Xptxas=-v',
               '-I' + str(tmp), '-I' + str(root / 'zig/kernels/cuda'),
               str(root / 'tools/cuda/width_attention.cu'), '-o', str(tmp / 'attention')]
    with (out / 'attention-compile.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((out / 'attention-compile.log').read_text(), flush=True)
        raise SystemExit(result.returncode)
    result = subprocess.run([str(tmp / 'attention')], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
    (out / 'attention-samples.jsonl').write_text(result.stdout)
    print(result.stdout, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    samples = [json.loads(s) for s in result.stdout.splitlines() if s.startswith('{')]
    assert samples and all(s['exact'] for s in samples)
    summary = []
    for key in sorted(set((s['rows'], s['length'], s['pattern'], s['mode']) for s in samples)):
        selected = [s for s in samples if (s['rows'], s['length'], s['pattern'], s['mode']) == key]
        summary.append(dict(zip(['rows', 'length', 'pattern', 'mode'], key),
                            median_ms=statistics.median(s['ms'] for s in selected)))
    (out / 'attention-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)

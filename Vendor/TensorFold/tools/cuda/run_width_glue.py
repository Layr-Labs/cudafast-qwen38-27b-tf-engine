#!/usr/bin/env python3
"""Check independent GDN warps grouped into larger blocks, preserving native arithmetic."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
out = Path(os.environ['Q38_OUT'])
assert '.{ .name = "qwen35", .flags = &.{"-O3"} }' in (root / 'zig/build/cuda_kernels.zig').read_text()
native = (root / 'zig/kernels/cuda/qwen35.cu').read_text()
pre = native[native.index('extern "C" __global__ void __launch_bounds__(32) tf_q35_gdn_pre'):native.index('// A segment\'s conv tail:')]
post = native[native.index('extern "C" __global__ void __launch_bounds__(32) tf_q35_gdn_post'):native.index('// Per-head RMSNorm of attention rows')]
functions = []
for warps in [4, 8]:
    for heads in [False, True]:
        label = f'{warps}{"h" if heads else "r"}'
        for name, body in [('pre', pre), ('post', post)]:
            body = body.replace(f'tf_q35_gdn_{name}', f'glue_{name}_{label}').replace('__launch_bounds__(32)', f'__launch_bounds__({32 * warps})')
            if heads:
                body = body.replace('head = blockIdx.y', f'head = blockIdx.y * {warps} + threadIdx.y')
                body = body.replace('hv = blockIdx.y', f'hv = blockIdx.y * {warps} + threadIdx.y')
            else:
                body = body.replace('w = blockIdx.z;', f'w = blockIdx.z * {warps} + threadIdx.y;')
                body = body.replace('past_rows(P, blockIdx.z)', f'past_rows(P, blockIdx.z * {warps} + threadIdx.y)')
                body = body.replace('m = blockIdx.z;', f'm = blockIdx.z * {warps} + threadIdx.y;')
            if name == 'post':
                body = body.replace('done[DK]', f'done[{warps}][DK]').replace('done[d]', 'done[threadIdx.y][d]').replace('group_sum(done +', 'group_sum(done[threadIdx.y] +')
            functions.append(body)
with tempfile.TemporaryDirectory(prefix='width-glue-') as tmp:
    tmp = Path(tmp)
    (tmp / 'width_glue.cuh').write_text('\n'.join(functions))
    command = ['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_121', '-Xptxas=-v',
               '-I' + str(tmp), '-I' + str(root / 'zig/kernels/cuda'),
               str(root / 'tools/cuda/width_glue.cu'), '-o', str(tmp / 'glue')]
    with (out / 'glue-compile.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((out / 'glue-compile.log').read_text(), flush=True)
        raise SystemExit(result.returncode)
    result = subprocess.run([str(tmp / 'glue')], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
    (out / 'glue-samples.jsonl').write_text(result.stdout)
    print(result.stdout, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    samples = [json.loads(s) for s in result.stdout.splitlines() if s.startswith('{')]
    assert len(samples) == 567 and all(s['exact'] for s in samples)
    summary = []
    for key in sorted(set((s['rows'], s['pattern'], s['mode']) for s in samples)):
        selected = [s for s in samples if (s['rows'], s['pattern'], s['mode']) == key]
        summary.append(dict(zip(['rows', 'pattern', 'mode'], key), median_ms=statistics.median(s['ms'] for s in selected)))
    (out / 'glue-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)

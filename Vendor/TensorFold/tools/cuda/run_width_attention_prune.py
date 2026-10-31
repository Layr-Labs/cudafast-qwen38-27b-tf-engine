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
source = (root / 'zig/kernels/cuda/qwen35_attn_tc.cuh').read_text()
source = source[source.index('// RB rows a block'):source.index('}  // namespace tf_q35_tc')]
source = source.replace('void attn_rows(', 'void attn_pruned(')
source = source.replace('auto scores = [&](float (&s)[2][4]) {', 'auto scores = [&](float (&s)[2][4], int only = -1) {\n            if (only >= 0 && (16 * mt / G > only || (16 * mt + 15) / G < only)) {\n#pragma unroll\n                for (int a = 0; a < 2; ++a)\n#pragma unroll\n                    for (int e = 0; e < 4; ++e) s[a][e] = 0.0f;\n                return;\n            }')
source = source.replace('scores(x);', 'scores(x, j);')
anchor = 'for (int a = 0; a < MT; a++)\n#pragma unroll\n                    for (int j = 0; j < TK / 16; j++) {'
assert source.count(anchor) == 1
source = source.replace(anchor, 'for (int a = 0; a < MT; a++) {\n                    if (only >= 0 && (16 * a / G > only || (16 * a + 15) / G < only)) continue;\n#pragma unroll\n                    for (int j = 0; j < TK / 16; j++) {')
anchor = 'qmm_frag::mma(o[a][n], pa4, b[0], b[1]);\n                            qmm_frag::mma(o[a][n + 1], pa4, b[2], b[3]);'
assert source.count(anchor) == 1
source = source.replace(anchor, 'float u[4], v[4];\n#pragma unroll\n                            for (int e = 0; e < 4; ++e) { u[e] = o[a][n][e]; v[e] = o[a][n+1][e]; }\n                            qmm_frag::mma(u, pa4, b[0], b[1]);\n                            qmm_frag::mma(v, pa4, b[2], b[3]);\n#pragma unroll\n                            for (int e = 0; e < 4; ++e) if (only < 0 || (16*a+g+(e/2)*8)/G == only) {\n                                o[a][n][e] = u[e]; o[a][n+1][e] = v[e];\n                            }')
anchor = '            };\n            if (shared) values(-1);'
assert source.count(anchor) == 1
source = source.replace(anchor, '                }\n' + anchor)
with tempfile.TemporaryDirectory(prefix='width-attn-prune-') as tmp:
    tmp = Path(tmp)
    (tmp / "width_attention_pruned.cuh").write_text("namespace tf_q35_tc {\n" + source + "\n}\n")
    command = ['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_121', '-Xptxas=-v',
               '-I' + str(tmp), '-I' + str(root / 'zig/kernels/cuda'),
               str(root / 'tools/cuda/width_attention_prune.cu'), '-o', str(tmp / 'attention')]
    with (out / 'attention-prune-compile.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((out / 'attention-prune-compile.log').read_text(), flush=True)
        raise SystemExit(result.returncode)
    result = subprocess.run([str(tmp / 'attention')], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
    (out / 'attention-prune-samples.jsonl').write_text(result.stdout)
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
    (out / 'attention-prune-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)

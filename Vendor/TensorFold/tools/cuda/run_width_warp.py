#!/usr/bin/env python3
"""Screen exact native tile widths with cold real checkpoint weights."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
out = Path(os.environ['Q38_OUT'])
snapshot = Path(os.environ['Q38_TARGET_DIR'])
index = json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map']
families = {
    'delta': ['linear_attn.in_proj_qkv', 'linear_attn.in_proj_z', 'linear_attn.in_proj_a', 'linear_attn.in_proj_b'],
    'attention': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
    'deltaout': ['linear_attn.out_proj'],
    'down': ['mlp.down_proj'],

}

def split(n, k):
    sk = 1
    while sk < 8 and ((n + 63) // 64) * sk < 192 and (k // 64) % (sk * 2) == 0 and k // 64 // (sk * 2) >= 8:
        sk *= 2
    return sk

with tempfile.TemporaryDirectory(prefix='width-warp-') as tmp:
    work = Path(tmp)
    cmd = ['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_121', '-Xptxas=-v',
           str(root / 'tools/cuda/width_warp.cu'), '-o', str(work / 'tiles')]
    with (out / 'compile.log').open('w') as log:
        compiled = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if compiled.returncode:
        print((out / 'compile.log').read_text(), flush=True)
        raise SystemExit(compiled.returncode)
    summary = {}
    for family, suffixes in families.items():
        args = []
        for i, suffix in enumerate(suffixes):
            key = sorted(k[:-7] for k in index if k.endswith(suffix + '.weight'))[0]
            prefix = work / (family + str(i))
            fixture = json.loads(subprocess.check_output(['python3', str(root / 'tools/cuda/width_fixture.py'),
                                str(snapshot), key, str(prefix)], text=True))
            args += [str(prefix), str(fixture['n']), str(fixture['k']), str(split(fixture['n'], fixture['k']))]
        result = subprocess.run([str(work / 'tiles'), *args], text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=180)
        (out / (family + '.jsonl')).write_text(result.stdout)
        print(family, result.stdout, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
        samples = [json.loads(s) for s in result.stdout.splitlines() if s.startswith('{')]
        if len(samples) != 351 or not all(s['exact'] for s in samples):
            raise ValueError('Incomplete exact screen')
        configs = []
        for mode in range(27):
            times = [s['ms'] for s in samples if s['mode'] == mode]
            ratios = [next(s['ms'] for s in samples if s['rep'] == r and s['mode'] == 2) /
                      next(s['ms'] for s in samples if s['rep'] == r and s['mode'] == mode) for r in range(13)]
            configs.append(dict(mode=mode, median_ms=statistics.median(times),
                                ratio=statistics.median(ratios), wins=sum(r > 1 for r in ratios)))
        summary[family] = configs
        (out / 'warp-summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps({family: configs}), flush=True)

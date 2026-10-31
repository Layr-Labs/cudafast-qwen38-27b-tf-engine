#!/usr/bin/env python3
"""Check fused wide projection bytes, tails, and cold-weight cost."""
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
with tempfile.TemporaryDirectory(prefix='width-single-') as tmp:
    work = Path(tmp)
    command = ['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_121', '-Xptxas=-v',
               str(root / 'tools/cuda/width_single.cu'), '-o', str(work / 'pair')]
    with (out / 'single-compile.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((out / 'single-compile.log').read_text(), flush=True)
        raise SystemExit(result.returncode)
    prefixes = []
    for role in ['gate', 'up']:
        key = sorted(k[:-7] for k in index if k.endswith('mlp.' + role + '_proj.weight'))[0]
        prefix = work / role
        fixture = json.loads(subprocess.check_output(['python3', str(root / 'tools/cuda/width_fixture.py'),
                            str(snapshot), key, str(prefix)], text=True))
        assert (fixture['n'], fixture['k']) == (17408, 5120)
        prefixes.append(str(prefix))
    result = subprocess.run([str(work / 'pair'), *prefixes], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=180)
    (out / 'single-samples.jsonl').write_text(result.stdout)
    print(result.stdout, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    samples = [json.loads(s) for s in result.stdout.splitlines() if s.startswith('{')]
    assert len(samples) == 182 and all(s['exact'] for s in samples)
    summary = []
    for mode in range(14):
        times = [s['ms'] for s in samples if s['mode'] == mode]
        ratios = [next(s['ms'] for s in samples if s['rep'] == r and s['mode'] == 0) /
                  next(s['ms'] for s in samples if s['rep'] == r and s['mode'] == mode) for r in range(13)]
        summary.append(dict(mode=mode, median_ms=statistics.median(times),
                            median_ratio=statistics.median(ratios), wins=sum(r > 1 for r in ratios)))
    (out / 'single-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)

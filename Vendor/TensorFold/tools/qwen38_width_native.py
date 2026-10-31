#!/usr/bin/env python3
"""Compare fixed native widths and isolated exact wide-kernel features."""
import json
import os
from pathlib import Path
import statistics
import re
from collections import Counter
import runpy
import sys
import qwen38_acceptance_probe as probe

run_case = probe.run_case

def configured(args, *rest):
    enabled = args.wide_backend != 0
    os.environ['TF_WIDE_GROUP'] = str(int(enabled))
    os.environ['TF_WIDE_SLOTS'] = str(int(enabled))
    os.environ['TF_WIDE_KMAJOR'] = str(int(args.wide_backend >= 2))
    os.environ['TF_WIDE_LEAVES'] = str(int(args.wide_backend == 3))
    os.environ['TF_WIDE_ATTN'] = str(int(enabled))
    os.environ['TF_GDN_SLOTS_RECEIPTS'] = '1'
    return run_case(args, *rest)

probe.run_case = configured
runpy.run_path(str(Path(__file__).parent / "cuda/run_width_slots.py"), run_name="__main__")
sys.argv = [sys.argv[0], '--variants',
            'a4:15:16:32:12:0.6:1.5:0:0:2:0:1:0',
            'a4:31:16:32:12:0.6:1.5:0:0:2:0:1:0',
            'a4:31:16:32:12:0.6:1.5:0:0:2:0:1:1',
            'a4:31:16:32:12:0.6:1.5:0:0:2:0:1:2',
            'a4:31:16:32:12:0.6:1.5:0:0:2:0:1:3',
            '--cases', 'gut-5', 'gut-9', '--warmup-case', 'gut-10', '--tokens', '128']
probe.main()
path = Path(os.environ['Q38_OUT']) / 'acceptance/summary.json'
results = json.loads(path.read_text())
slot_plans = {}
for receipt in path.parent.glob('*/result.json'):
    result = json.loads(receipt.read_text())
    lines = receipt.with_name('rounds.log').read_text().splitlines()
    plans = [tuple(map(int, match.groups())) for line in lines
             if (match := re.search(r'\[gdn-slots\] rows=(\d+) peak=(\d+) variant=(\d+)', line))]
    if len(plans) != len(result['rounds']):
        raise ValueError('Native GDN schedule receipts do not match the measured round count')
    slot_plans[(result['name'], result['nodes'], result['wide_backend'], result['rep'])] = dict(
        Counter(f'{rows}:{peak}:{variant}' for rows, peak, variant in plans))
summary = []
for case in ['gut-5', 'gut-9']:
    group = [r for r in results if r['name'] == case]
    baseline = next(r for r in group if r['nodes'] == 15)
    same_paths = [r['decoded']['acceptance_lengths'] for r in group if r['nodes'] == 31]
    if any(p != same_paths[0] for p in same_paths[1:]):
        raise ValueError('Wide backends changed fixed-tree acceptance paths')
    for row in group:
        times = [r['verify_ms'] for r in row['rounds']]
        summary.append(dict(case=case, nodes=row['nodes'], mode=row['wide_backend'],
            mean_verify_ms=statistics.mean(times), median_verify_ms=statistics.median(times),
            round_ms=row['round_ms'], rounds=len(row['rounds']), tokens_per_round=row['tokens_per_round'],
            slot_plans=slot_plans[(row['name'], row['nodes'], row['wide_backend'], row['rep'])],
            seed_decode_ratio=(baseline['seed_ms'] + baseline['decode_ms']) / (row['seed_ms'] + row['decode_ms'])))
path.with_name('width-native-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)

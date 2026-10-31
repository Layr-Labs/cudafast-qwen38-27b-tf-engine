#!/usr/bin/env python3
"""Report fixed-width proxy cost and exactness; these are not ranked scores."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics


def report(directory):
    groups = defaultdict(list)
    for path in sorted(directory.glob('*/result.json')):
        row = json.loads(path.read_text())
        row['_path'] = path
        groups[row['name']].append(row)
    if not groups:
        raise ValueError('No completed native observations')
    results = []
    ratios = defaultdict(list)
    variant_sets = []
    for case, rows in sorted(groups.items()):
        variants = [(r['nodes'], r['wide_backend'], r['rep']) for r in rows]
        if len(variants) != len(set(variants)):
            raise ValueError('Duplicate native observations')
        variant_sets.append(set(variants))
        baselines = [r for r in rows if r['nodes'] == 15 and r['wide_backend'] == 0]
        if len(baselines) != 1:
            raise ValueError('Require one 15-node reference per proxy case')
        baseline = baselines[0]
        digest = baseline['token_sha256']
        base_verify = statistics.mean(r['verify_ms'] for r in baseline['rounds'])
        path31 = None
        for row in sorted(rows, key=lambda r: (r['nodes'], r['wide_backend'], r['rep'])):
            if row['token_sha256'] != digest:
                raise ValueError('Proxy continuations differ')
            if sum(row['decoded']['acceptance_lengths']) != 128:
                raise ValueError('Require all 128 decode tokens')
            if len(row['rounds']) != len(row['decoded']['acceptance_lengths']):
                raise ValueError('Incomplete round receipts')
            if row['fixed_tree'] and any(r['rows'] != row['nodes'] + 1 for r in row['rounds']):
                raise ValueError('A fixed-width observation changed width')
            if row['nodes'] == 31:
                paths = row['decoded']['acceptance_lengths']
                if path31 is not None and path31 != paths:
                    raise ValueError('Kernel choices changed 31-node acceptance paths')
                path31 = paths
            ratio = (baseline['seed_ms'] + baseline['decode_ms']) / (row['seed_ms'] + row['decode_ms'])
            ratios[(row['nodes'], row['wide_backend'])].append(ratio)
            plans = [tuple(map(int, m.groups())) for line in row['_path'].with_name('rounds.log').read_text().splitlines()
                     if (m := re.search(r'\[gdn-slots\] rows=(\d+) peak=(\d+) variant=(\d+)', line))]
            slot_groups = defaultdict(list)
            if plans:
                if len(plans) != len(row['rounds']):
                    raise ValueError('Incomplete slot receipts')
                for (_, peak, variant), timing in zip(plans, row['rounds']):
                    slot_groups[(peak, variant)].append(timing['verify_ms'])
            results.append(dict(case=case, nodes=row['nodes'], backend=row['wide_backend'],
                rounds=len(row['rounds']), tokens_per_round=row['tokens_per_round'],
                round_ms=row['round_ms'], round_tax_ms=row['round_ms'] - baseline['round_ms'],
                verify_ms=statistics.mean(r['verify_ms'] for r in row['rounds']),
                verify_tax_ms=statistics.mean(r['verify_ms'] for r in row['rounds']) - base_verify,
                seeded_decode_ratio=ratio,
                slots=[dict(peak=peak, variant=variant, rounds=len(times), verify_ms=statistics.mean(times))
                       for (peak, variant), times in sorted(slot_groups.items())]))
    if any(v != variant_sets[0] for v in variant_sets):
        raise ValueError('Proxy cases have incomplete variant sets')
    return dict(scope='fixed-width private proxy observations; not an official score', cases=len(groups),
                observations=results, seeded_decode_geomeans=[dict(nodes=n, backend=b, ratio=statistics.geometric_mean(values))
                for (n, b), values in sorted(ratios.items())])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='The fetched acceptance output directory')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(report(args.directory), indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')

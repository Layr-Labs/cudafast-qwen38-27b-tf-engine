#!/usr/bin/env python3
"""Audit diagnostic Landing observations against the next native pre-update state."""
import json
import math
from pathlib import Path
import struct


def parse(text):
    records, current = [], None
    for line in text.splitlines():
        if line.startswith('[landing_begin] '):
            if current is not None:
                raise ValueError('nested Landing receipt')
            current = {k: int(v) for k, v in (x.split('=') for x in line.split()[1:])}
            current.update(bins=[], follow=[], observations=[])
        elif line.startswith('[landing_bin] '):
            index, offered, landed = map(float, line.split()[1].split(','))
            if current is None or index != len(current['bins']) or not 0 <= landed <= offered:
                raise ValueError('invalid Landing bin')
            current['bins'].append([offered, landed])
        elif line.startswith('[landing_follow] '):
            current['follow'].append(list(map(int, line.split()[1].split(','))))
        elif line.startswith('[landing_node] '):
            index, token, parent, bits, eligible, landed = map(int, line.split()[1].split(','))
            own = struct.unpack('<d', struct.pack('<Q', bits))[0]
            if current is None or index != len(current['observations']) or parent >= index or not math.isfinite(own) or own > 1e-10:
                raise ValueError('invalid Landing observation')
            current['observations'].append(dict(index=index, token=token, parent=parent, own=own, eligible=bool(eligible), landed=bool(landed)))
        elif line == '[landing_end]':
            if current is None or len(current['bins']) != 16 or len(current['observations']) != current['nodes']:
                raise ValueError('incomplete Landing receipt')
            records.append(current)
            current = None
    if current is not None:
        raise ValueError('unterminated Landing receipt')
    for previous, following in zip(records, records[1:]):
        if previous['sampled'] != following['sampled']:
            raise ValueError('mixed sampling tables require a separate audit')
        expected = [x.copy() for x in previous['bins']]
        for node in previous['observations']:
            if node['eligible']:
                index = min(15, max(0, math.floor(-node['own'] / .5)))
                expected[index][0] += 1
                expected[index][1] += node['landed']
        if expected != following['bins']:
            raise ValueError(f"Landing update differs after position {previous['position']}")
    return records


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('log',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();records=parse(a.log.read_text())
    if a.out:a.out.write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({'round_receipts':len(records),'eligible':sum(n['eligible'] for r in records for n in r['observations'])}))

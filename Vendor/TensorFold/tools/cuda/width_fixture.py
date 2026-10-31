#!/usr/bin/env python3
"""Export one affine-4 projection for decode_tiles.cu, without loading the model."""
import argparse
import json
import struct
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('snapshot', type=Path)
p.add_argument('key', help='Projection prefix, excluding .weight/.scales/.biases')
p.add_argument('output', type=Path)
a = p.parse_args()
index = json.loads((a.snapshot / 'model.safetensors.index.json').read_text())['weight_map']
shapes = {}
for suffix, dtype in [('weight', 'U32'), ('scales', 'BF16'), ('biases', 'BF16')]:
    key = a.key + '.' + suffix
    with (a.snapshot / index[key]).open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(size))
        entry = header[key]
        if entry['dtype'] != dtype:
            raise ValueError(f'{key}: expected {dtype}, got {entry["dtype"]}')
        start, end = entry['data_offsets']
        f.seek(8 + size + start)
        data = f.read(end - start)
        if len(data) != end - start:
            raise ValueError('Truncated tensor')
        Path(str(a.output) + '.' + suffix).write_bytes(data)
        shapes[suffix] = entry['shape']
n, k8 = shapes['weight']
assert shapes['scales'] == shapes['biases'] == [n, k8 // 8]
print(json.dumps({'prefix': str(a.output), 'n': n, 'k': k8 * 8, 'key': a.key}))

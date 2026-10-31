#!/usr/bin/env python3
"""Run independent recurrence and dependent-projection byte checks sequentially."""
import json
import os
from pathlib import Path
import subprocess
import sys

out = Path(os.environ['Q38_OUT'])
results = []
for source in ['run_width_siblings.py', 'run_width_down_chain.py']:
    path = Path(__file__).parent / source
    with (out / (source + '.log')).open('w') as log:
        result = subprocess.run([sys.executable, str(path)], stdout=log, stderr=subprocess.STDOUT)
    row = dict(source=source, exit_code=result.returncode)
    results.append(row)
    print(json.dumps(row), flush=True)
    if result.returncode:
        print((out / (source + '.log')).read_text()[-8000:], flush=True)
(out / 'followup-status.json').write_text(json.dumps(results, indent=2))
raise SystemExit(int(any(r['exit_code'] for r in results)))

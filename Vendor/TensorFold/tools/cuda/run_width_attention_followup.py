#!/usr/bin/env python3
"""Check the served attention implementation, then screen exact warp ownership."""
from pathlib import Path
import runpy
for name in ['run_width_attention.py', 'run_width_warp.py']:
    runpy.run_path(str(Path(__file__).parent / name), run_name='__main__')

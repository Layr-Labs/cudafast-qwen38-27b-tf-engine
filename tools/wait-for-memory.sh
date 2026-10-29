#!/usr/bin/env bash
# Wait under the GPU lock before benchd starts either workspace's serve script.
#
# Usage: tools/wait-for-memory.sh   (no arguments; configured by the env below)
#
# This only observes memory; it never drops caches or changes the serve plan.
# The weights variables, SERVE_UP_MIN_CACHE_GIB, and SERVE_UP_MEMINFO match
# serve-up.sh. The defaults are 8 GiB for cache and /proc/meminfo.
# WAIT_FOR_MEMORY_TIMEOUT_S (300) and WAIT_FOR_MEMORY_POLL_S (5) accept fractional
# seconds for tests. The extra 2 GiB absorbs small changes before resident boot.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

exec python3 - "${ROOT_DIR}" <<'PY'
import math
import os
from pathlib import Path
import re
import sys
import time


def log(message):
    print(f"wait-for-memory: {message}", file=sys.stderr, flush=True)


def seconds(name, default, allow_zero=False):
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return value


def read_memory(path):
    values = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in ("MemTotal:", "MemAvailable:"):
            if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdigit():
                raise ValueError(f"invalid {fields[0]} in {path}")
            values[fields[0]] = int(fields[1]) * 1024
    if "MemTotal:" not in values or "MemAvailable:" not in values:
        raise ValueError(f"{path} lacks MemTotal/MemAvailable; refusing to wait blind")
    if values["MemTotal:"] <= 0:
        raise ValueError(f"invalid MemTotal in {path}")
    return values["MemTotal:"], values["MemAvailable:"]


try:
    root = Path(sys.argv[1])
    serve_text = (root / "tools/serve-up.sh").read_text()
    # Read the trusted default without sourcing serve-up.sh, which boots or stops
    # residents. Fail closed if the literal changes form.
    overhead = re.findall(r"^HOST_OVERHEAD_GIB=([0-9]+)$",
                          serve_text, re.MULTILINE)
    if len(overhead) != 1:
        raise ValueError("cannot read HOST_OVERHEAD_GIB from tools/serve-up.sh")
    gib = 1024**3
    cache_text = os.environ.get("SERVE_UP_MIN_CACHE_GIB") or overhead[0]
    if not re.fullmatch(r"0|[1-9][0-9]*", cache_text):
        raise ValueError("SERVE_UP_MIN_CACHE_GIB must be a non-negative base-10 integer")
    cache_gib = int(cache_text)
    cache_bytes = cache_gib * gib
    margin_bytes = 2 * gib
    weights_text = (os.environ.get("SERVE_UP_WEIGHTS_DIR") or
                    os.environ.get("MLXFAST_TARGET_SNAPSHOT_DIR"))
    if not weights_text:
        raise ValueError("SERVE_UP_WEIGHTS_DIR/MLXFAST_TARGET_SNAPSHOT_DIR is unset")
    weights = Path(weights_text)
    if not weights.is_dir():
        raise ValueError(f"target weights directory is missing: {weights}")
    body_bytes = sum(path.stat().st_size for path in weights.glob("*.safetensors"))
    if body_bytes <= 0:
        raise ValueError(f"target weights total is 0 bytes in {weights}")
    # TensorFold keeps an OS reserve out of its admission room on unified memory:
    # max(4 GiB, MemTotal / 10) (Vendor/TensorFold/zig/src/native/cuda_memory.zig reserveBytes; serve-up.sh never
    # passes TENSORFOLD_MEMORY_RESERVE_GIB, so the default applies). The gate waits for that too.
    total_now, _ = read_memory(Path(os.environ.get("SERVE_UP_MEMINFO") or "/proc/meminfo"))
    reserve_bytes = max(4 * gib, total_now // 10)
    required = body_bytes + cache_bytes + reserve_bytes + margin_bytes
    meminfo = Path(os.environ.get("SERVE_UP_MEMINFO") or "/proc/meminfo")
    timeout = seconds("WAIT_FOR_MEMORY_TIMEOUT_S", "300", allow_zero=True)
    poll = seconds("WAIT_FOR_MEMORY_POLL_S", "5")
    started = time.monotonic()
    deadline = started + timeout
    log(f"need weights={body_bytes} bytes + cache={cache_bytes} bytes "
        f"({cache_gib} GiB) + engine reserve={reserve_bytes} bytes + margin={margin_bytes} bytes (2 GiB) "
        f"= required={required} bytes; weights_dir={weights}; meminfo={meminfo}; "
        f"timeout={timeout:g}s poll={poll:g}s")
    while True:
        _total, available = read_memory(meminfo)
        now = time.monotonic()
        status = (f"MemAvailable={available / gib:.2f} GiB, "
                  f"required={required / gib:.2f} GiB ({required} bytes)")
        if available >= required:
            log(f"ready after {now - started:.3f}s: {status}")
            break
        if now >= deadline:
            raise ValueError(f"timed out after {now - started:.3f}s: {status}; "
                             "memory did not free; refusing to start measurement")
        log(f"waiting: {status}; elapsed={now - started:.3f}s")
        time.sleep(min(poll, deadline - now))
except (OSError, ValueError, KeyError, TypeError) as exc:
    log(f"ERROR: {exc}")
    sys.exit(1)
PY

#!/usr/bin/env python3
"""Explicit diagnostic: ordinary depth-1 DFlash decodes of caller-supplied public texts under several node budgets.

Each case (PROBE_CASES, default n15 / n31 without the width selector / sel = 31 with it) runs one resident; every
text is decoded `samples` times (the first a warmup) for `count` tokens. Per text and case: complete window ms per
token, rounds, accepted drafts, token-tape digest; tapes must agree across cases (the drafts change, the target's
tokens do not). The resident's stderr (TF_TIMING) is kept per case, including the selector's decisions.
usage: probe_width_select.py <blob> [samples] [count] [cases: name:budget:K=V,K=V;...]
"""
import base64
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import zlib

texts = json.loads(zlib.decompress(base64.b64decode(sys.argv[1])))
samples = int(sys.argv[2]) if len(sys.argv) > 2 else 3
count = int(sys.argv[3]) if len(sys.argv) > 3 else 128
root = Path(os.environ['Q38_TREE'])
spec = importlib.util.spec_from_file_location('author_prompts', root / 'tools/author-prompts.py')
author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)
tokenizer, seed_count, vocab = author.load_pinned_tokenizer(os.environ['Q38_TARGET_DIR'], str(root / 'fixtures/qwen3_8_27b_tf_cuda_v1_track.json'))
def seed_of(text):
    # an authored case [prefix, body, suffix]: its body repeated, cut so the closing instruction ends the seed
    if isinstance(text, list):
        prefix, body, suffix = text
        tail = tokenizer.encode(suffix, add_special_tokens=False).ids
        ids = tokenizer.encode(prefix + body * 12, add_special_tokens=False).ids[:seed_count - len(tail)] + tail
        assert len(ids) == seed_count and all(0 <= t < vocab for t in ids)
        return ids
    return author.seed_ids(tokenizer, text, seed_count, vocab)[0]


seeds = [(name, seed_of(text)) for name, text in texts]
del texts, tokenizer
worker = os.environ['Q38_WORKER']
out = Path(os.environ['Q38_OUT'])
default_cases = 'n15:15:TF_WIDTH_SELECT=0;n31:31:TF_WIDTH_SELECT=0;sel:31:'
cases = []
for item in (sys.argv[4] if len(sys.argv) > 4 else os.environ.get('PROBE_CASES', default_cases)).split(';'):
    name, budget, settings = item.split(':', 2)
    cases.append((name, budget, dict(kv.split('=', 1) for kv in settings.split(',') if kv)))
rows = []
reference = {}
all_exact = True
for case, budget, overrides in cases:
    with tempfile.TemporaryDirectory(prefix='width-select-') as folder:
        folder = Path(folder)
        socket = folder / 'resident.sock'
        home = folder / 'home'
        home.mkdir()
        trace_path = out / f'{case}.resident.log'
        env = dict(os.environ, HOME=str(home), TF_TIMING='1', TF_DRAFT_NODES=budget)
        env.update(overrides)
        env['BENCH_WORKER_RESIDENT_SOCKET'] = str(socket)
        with trace_path.open('wb') as trace:
            resident = subprocess.Popen([worker, 'resident', '--weights', env['Q38_TARGET_DIR'], '--drafter', env['Q38_DRAFTER_DIR'], '--socket', str(socket)], stdout=subprocess.DEVNULL, stderr=trace, env=env)
            client = None
            try:
                deadline = time.monotonic() + float(os.environ.get('PROBE_START_S', '300'))
                while not socket.exists():
                    if resident.poll() is not None:
                        raise RuntimeError(f'resident exited {resident.returncode}')
                    if time.monotonic() > deadline:
                        raise RuntimeError('resident socket timeout')
                    time.sleep(.1)
                client = subprocess.Popen([worker, 'runtime-worker', '--weights', env['Q38_TARGET_DIR']], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=trace, text=True, env=env)
                hello = json.loads(client.stdout.readline())
                if not hello.get('ok') or hello.get('backend') != 'cuda':
                    raise RuntimeError('invalid CUDA worker handshake')
                request_id = 0

                def request(kind, **values):
                    global request_id
                    request_id += 1
                    client.stdin.write(json.dumps({'id': request_id, 'kind': kind, **values}) + '\n')
                    client.stdin.flush()
                    response = json.loads(client.stdout.readline())
                    if not response.get('ok') or response.get('id') != request_id:
                        raise RuntimeError('protocol request failed: ' + str(response.get('error')))
                    return response

                for name, seed in seeds:
                    for sample in range(samples):
                        request('free_decode_begin', seed_tokens=seed, spec={'mode': 'dflash', 'dflash': {'depth': 1}})
                        started = time.monotonic_ns()
                        response = request('free_decode_run', count=count)
                        elapsed_ms = (time.monotonic_ns() - started) / 1e6
                        tokens = response.pop('tokens')
                        if name not in reference:
                            reference[name] = tokens
                        exact = tokens == reference[name]
                        all_exact = all_exact and exact
                        lengths = response.get('acceptance_lengths', [])
                        row = dict(prompt=name, case=case, budget=int(budget), sample=sample, warmup=sample == 0, exact=exact,
                                   window_ms=elapsed_ms, ms_per_token=elapsed_ms / max(1, len(tokens)), rounds=len(lengths),
                                   committed_total=response.get('committed_total'), accepted_total=response.get('accepted_total'),
                                   drafted_total=response.get('drafted_total'),
                                   tokens_sha=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16])
                        rows.append(row)
                        print(json.dumps(row), flush=True)
            finally:
                if client is not None:
                    client.terminate()
                    try:
                        client.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        client.kill()
                        client.wait()
                resident.terminate()
                try:
                    resident.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    resident.kill()
                    resident.wait()

(out / 'width_select_receipt.json').write_text(json.dumps(rows, indent=2) + '\n')
names = [n for n, _ in seeds]
case_names = [c for c, _, _ in cases]
table = collections.defaultdict(dict)
for n in names:
    for c in case_names:
        chosen = [r['ms_per_token'] for r in rows if r['prompt'] == n and r['case'] == c and not r['warmup']]
        if chosen:
            table[n][c] = statistics.mean(chosen)
print('\nms per token (mean of timed samples):')
print(f"{'prompt':10s}" + ''.join(f'{c:>10s}' for c in case_names) + '   best   sel/best')
for n in names:
    vals = table[n]
    fixed = {c: v for c, v in vals.items() if c != 'sel'}
    best = min(fixed, key=fixed.get) if fixed else None
    line = f'{n:10s}' + ''.join(f'{vals.get(c, float("nan")):10.3f}' for c in case_names)
    if best and 'sel' in vals:
        line += f'   {best:5s} {vals["sel"] / fixed[best]:8.4f}'
    print(line)
print(json.dumps({'width_select_probe': True, 'all_tokens_equal': all_exact, 'table': table}), flush=True)
if not all_exact:
    raise SystemExit(1)

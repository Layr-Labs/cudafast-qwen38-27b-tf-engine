#!/usr/bin/env python3
# Explicit diagnostic: per-prompt width evidence from probe_width_select.py traces (TF_TIMING_ROUNDS=1, n15 and n31).
import json, re, sys, statistics, collections, math
# usage: wsel_offline.py <files dir>  (n15.resident.log, n31.resident.log, width_select_receipt.json)
D = sys.argv[1]
rows = json.load(open(f'{D}/width_select_receipt.json'))
pat = re.compile(r'\[timing\] wsel_round limit=(\d+) offered=(\d+) kept=(\d+) kept15=(\d+) tree=(\d) grown=(\d+) ms=(-?[0-9.]+) e15=([0-9.]+) egain=([0-9.]+) c15=([0-9.]+) cgain=([0-9.]+)')
def rounds_of(case):
    out = []
    for line in open(f'{D}/{case}.resident.log', errors='replace'):
        m = pat.search(line)
        if m:
            v = m.groups()
            out.append(dict(limit=int(v[0]), offered=int(v[1]), kept=int(v[2]), kept15=int(v[3]), tree=int(v[4]), grown=int(v[5]), ms=float(v[6]), e15=float(v[7]), egain=float(v[8]), c15=float(v[9]), cgain=float(v[10])))
    return out
streams = {}
for case in ('n15', 'n31'):
    rr = rounds_of(case)
    sel = [r for r in rows if r['case'] == case]
    total = sum(r['rounds'] for r in sel)
    print(case, 'trace rounds', len(rr), 'receipt rounds', total)
    # segment: cumulative committed tokens per stream reach 128 (kept rows a round = committed tokens)
    i = 0
    for r in sel:
        seg = []; got = 0
        while i < len(rr) and got < 128:
            seg.append(rr[i]); got += rr[i]['kept']; i += 1
        streams[(r['prompt'], case, r['sample'])] = (r, seg)
prompts = list(dict.fromkeys(r['prompt'] for r in rows))
# per-prompt summary (timed samples)
print(f"\n{'prompt':15s} {'n15':>7s} {'n31':>7s} {'best':>5s} | n31 rounds: extra/kept15  mass(e) mass(c) | ms31 ms15")
cal_x = []; cal_y = []
per = {}
for p in prompts:
    v15 = [streams[(p, 'n15', s)][0]['ms_per_token'] for s in (1, 2)]
    v31 = [streams[(p, 'n31', s)][0]['ms_per_token'] for s in (1, 2)]
    segs = [streams[(p, 'n31', s)][1] for s in (1, 2)]
    allr = [x for sg in segs for x in sg]
    ext = sum(x['kept'] - x['kept15'] for x in allr); k15 = sum(x['kept15'] for x in allr)
    me = sum(x['egain'] for x in allr); mc = sum(x['cgain'] for x in allr)
    ms31 = statistics.median([x['ms'] for x in allr if x['ms'] > 0])
    r15 = [x for s in (1, 2) for x in streams[(p, 'n15', s)][1]]
    ms15 = statistics.median([x['ms'] for x in r15 if x['ms'] > 0])
    a, b = statistics.mean(v15), statistics.mean(v31)
    per[p] = dict(n15=a, n31=b, extra=ext, k15=k15, me=me, mc=mc)
    print(f'{p:15s} {a:7.2f} {b:7.2f} {"n31" if b < a else "n15":>5s} | {ext:4d}/{k15:4d}={ext/max(1,k15):.3f}  {me/max(1,k15):.3f} {mc/max(1,k15):.3f} | {ms31:.1f} {ms15:.1f}')
    for x in allr:
        if x['tree'] and x['offered'] > 15:
            cal_x.append(x['cgain']); cal_y.append(x['kept'] - x['kept15'])
if cal_x:
    a = sum(cal_y) / max(1e-9, sum(cal_x))
    mx = statistics.mean(cal_x); my = statistics.mean(cal_y)
    cov = sum((x - mx) * (y - my) for x, y in zip(cal_x, cal_y)); vx = sum((x - mx) ** 2 for x in cal_x); vy = sum((y - my) ** 2 for y in cal_y)
    print(f'\nrounds offering >15: {len(cal_x)}  realized extra / calibrated mass = {a:.3f}  per-round corr {cov / math.sqrt(vx * vy + 1e-12):.3f}')
json.dump({'per': per}, open(f'{D}/wsel_offline.json', 'w'), indent=1)

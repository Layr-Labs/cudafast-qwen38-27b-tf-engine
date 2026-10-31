#!/usr/bin/env python3
# Explicit diagnostic: replays per-stream width rules (rows-share threshold after an evidence window) on n15/n31 traces.
import json, re, sys, statistics, math, collections
D = sys.argv[1]
rows = json.load(open(f'{D}/width_select_receipt.json'))
pat = re.compile(r'\[timing\] wsel_round limit=(\d+) offered=(\d+) kept=(\d+) kept15=(\d+) tree=(\d) grown=(\d+) ms=(-?[0-9.]+)')
def rounds_of(case):
    return [dict(kept=int(m[3]), k15=int(m[4]), ms=float(m[7])) for m in (pat.search(l) for l in open(f'{D}/{case}.resident.log', errors='replace')) if m]
streams = {}
for case in ('n15', 'n31'):
    rr = rounds_of(case); i = 0
    for r in [x for x in rows if x['case'] == case]:
        seg = []; got = 0
        while i < len(rr) and got < 128:
            seg.append(rr[i]); got += rr[i]['kept']; i += 1
        streams[(r['prompt'], case, r['sample'])] = (r, seg)
prompts = list(dict.fromkeys(r['prompt'] for r in rows))
def sim(rule_w, thr, recheck=24, burst=3):
    res = {}
    for p in prompts:
        vals = []
        for s in (1, 2):
            r31, seg = streams[(p, 'n31', s)]
            r15, _ = streams[(p, 'n15', s)]
            med = statistics.median([x['ms'] for x in seg if x['ms'] > 0])
            ms = [x['ms'] if x['ms'] > 0 else med for x in seg]
            scale = r31['window_ms'] / sum(ms)
            _, seg15 = streams[(p, 'n15', s)]
            med15 = statistics.median([x['ms'] for x in seg15 if x['ms'] > 0])
            ms15 = [x['ms'] if x['ms'] > 0 else med15 for x in seg15]
            sc15 = r15['window_ms'] / sum(ms15)
            curve = []; ct = 0.0; ck = 0
            for x, m in zip(seg15, ms15):
                curve.append((ck, ct)); ct += m * sc15; ck += x['kept']
            curve.append((ck, ct))
            def t15_at(tok):
                for (k0, t0), (k1, t1) in zip(curve, curve[1:]):
                    if k1 >= tok: return t0 + (t1 - t0) * (tok - k0) / max(1, k1 - k0)
                return curve[-1][1]
            kept = k15 = 0; t = 0.0; tok = 0; narrow_at = None
            for j, x in enumerate(seg):
                t += ms[j] * scale; tok += x['kept']; kept += x['kept']; k15 += x['k15']
                if tok >= 128: break
                if k15 >= rule_w and (kept - k15) / k15 < thr:
                    narrow_at = j; break
            if narrow_at is None or tok >= 128:
                total = r31['window_ms'] if narrow_at is None else t
            else:
                total = t + (r15['window_ms'] - t15_at(tok))
            vals.append(total / 128)
        res[p] = statistics.mean(vals)
    return res
base = {p: (statistics.mean(streams[(p,'n15',s)][0]['ms_per_token'] for s in (1,2)), statistics.mean(streams[(p,'n31',s)][0]['ms_per_token'] for s in (1,2))) for p in prompts}
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
orc = gm([min(a, b) / b for a, b in base.values()])
print(f'oracle/n31 {orc:.4f}  n15/n31 {gm([a / b for a, b in base.values()]):.4f}')
for w in (16, 24, 32, 48, 64, 96):
    for thr in (0.03, 0.04, 0.05, 0.06, 0.08):
        r = sim(w, thr)
        g = gm([r[p] / base[p][1] for p in prompts])
        worst = max(prompts, key=lambda p: r[p] / min(base[p]))
        print(f'W={w:3d} thr={thr:.2f}: rule/n31 {g:.4f} (oracle {orc:.4f}); worst vs oracle {worst} {r[worst]/min(base[worst]):.4f}')
r = sim(32, 0.05)
for p in prompts:
    a, b = base[p]
    print(f'  {p:15s} n15 {a:6.2f} n31 {b:6.2f} rule {r[p]:6.2f}  rule/best {r[p]/min(a,b):.4f}')

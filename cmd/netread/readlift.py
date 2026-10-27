"""Compare netread's reads with the mini-sim reads logged in the turn log.

    python3 readlift.py base.txt reads.jsonl.gz [more reads files...]

For each position, the net read's posterior is prior weight x likelihood,
where the likelihood of a leave is the softmax share of the actual play
among its alternatives, scored by the value head (in atanh space, which
undoes its saturation in decided games) or by the spread head, at a few
temperatures. Lift is log2(posterior / prior) of the true leave; the turn
log's liftBits is the mini-sim read's lift at the same turn. Both are
averaged over the positions where both exist, by bag.
"""
import csv, gzip, json, math, sys
from collections import defaultdict

def halves(rows):
    out, start = [], 0
    for i in range(1, len(rows)):
        try:
            prev, cur = int(rows[i-1][2]), int(rows[i][2])
        except ValueError:
            continue
        if cur <= prev:
            out.append(rows[start:i]); start = i
    out.append(rows[start:])
    return out

def logged_lifts(path):
    by_game = defaultdict(list)
    with open(path) as f:
        for r in csv.reader(f):
            if r[0] == 'playerID' or len(r) < 17:
                continue
            by_game[r[1]].append(r)
    lifts = {}
    for gid, rows in by_game.items():
        for h in halves(rows):
            for i, r in enumerate(h):
                if r[16].strip():
                    lifts[f"{gid}|{h[0][0]}|{i}"] = float(r[16])
    return lifts

def opener(p):
    return gzip.open(p, 'rt') if p.endswith('.gz') else open(p)

def lift(rec, heads, T):
    V = rec[heads]
    ti = rec['trueIdx']
    W = rec['w']
    wsum = sum(W)
    post = []
    for v, w in zip(V, W):
        if heads == 'v':
            v = [math.atanh(max(-0.9999, min(0.9999, x))) for x in v]
        m = max(v)
        z = sum(math.exp((x - m) / T) for x in v)
        post.append(w * math.exp((v[0] - m) / T) / z)
    Z = sum(post)
    return math.log2((post[ti] / Z) / (W[ti] / wsum))

def bucket(bag):
    return '1-7' if bag <= 7 else '8-20' if bag <= 20 else '21-50' if bag <= 50 else '51+'

def main():
    lifts = logged_lifts(sys.argv[1])
    recs = []
    for p in sys.argv[2:]:
        with opener(p) as f:
            recs += [r for r in map(json.loads, f) if r.get('leaves') and r['trueIdx'] >= 0 and r['key'] in lifts]
    print(f"{len(recs)} positions with both reads (true leave among the net's candidates)")
    configs = [('v', T) for T in (0.05, 0.1, 0.2, 0.5)] + [('s', T) for T in (0.01, 0.03, 0.1)]
    groups = defaultdict(list)
    for r in recs:
        groups[bucket(r['bag'])].append(r)
        groups['all'].append(r)
    for g in ('1-7', '8-20', '21-50', '51+', 'all'):
        rs = groups.get(g)
        if not rs:
            continue
        base = sum(lifts[r['key']] for r in rs) / len(rs)
        cells = []
        for heads, T in configs:
            if not all(r.get(heads) for r in rs):
                continue
            cells.append(f"{heads}{T}: {sum(lift(r, heads, T) for r in rs) / len(rs):+.2f}")
        print(f"bag {g:5s} n={len(rs):5d}  mini-sim {base:+.2f}  net " + '  '.join(cells))

main()

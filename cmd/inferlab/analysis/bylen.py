#!/usr/bin/env python3
"""Paired comparison of two inferlab runs, broken down by leave length.

A change to a global constant has to be judged everywhere it applies. The
six-tile bucket is where the read breaks, but the one-to-five-tile buckets are
where most of the run's information comes from, and a fix that trades the one
for the other is not a fix.
"""
import json, math, sys, random, statistics as st
from collections import defaultdict

def load(f):
    out={}
    for line in open(f):
        line=line.strip()
        if not line: continue
        r=json.loads(line)
        if r.get('err'): continue
        if r.get('liftBits') is None: r['liftBits']=float('-inf')
        out[(r['gameID'],r['half'],r['turn'])]=r
    return out

def boot_ci(d, n=20000, seed=7):
    random.seed(seed)
    b=sorted(st.mean(random.choices(d,k=len(d))) for _ in range(n))
    return b[int(.025*n)], b[int(.975*n)]

def main(base_f, var_f):
    B, V = load(base_f), load(var_f)
    keys=sorted(set(B)&set(V))
    print(f"{len(keys)} positions in common: {base_f} vs {var_f}\n")
    bylen=defaultdict(list)
    for k in keys:
        bylen[B[k]['leaveLen']].append(k)
    print(f"  {'tiles':<7}{'n':>5}{'base mean':>11}{'variant':>10}{'paired gain':>13}{'95% CI':>18}{'better':>8}{'ruled out b/v':>15}")
    allD=[]
    for L in sorted(bylen):
        ks=bylen[L]
        d=[V[k]['liftBits']-B[k]['liftBits'] for k in ks
           if math.isfinite(V[k]['liftBits']) and math.isfinite(B[k]['liftBits'])]
        bm=st.mean([B[k]['liftBits'] for k in ks if math.isfinite(B[k]['liftBits'])] or [float('nan')])
        vm=st.mean([V[k]['liftBits'] for k in ks if math.isfinite(V[k]['liftBits'])] or [float('nan')])
        rb=sum(1 for k in ks if B[k]['ruledOut']); rv=sum(1 for k in ks if V[k]['ruledOut'])
        if len(d)>=3:
            lo,hi=boot_ci(d); allD+=d
            print(f"  {L:<7}{len(ks):>5}{bm:>+11.3f}{vm:>+10.3f}{st.mean(d):>+13.3f}   [{lo:+.3f}, {hi:+.3f}]"
                  f"{100*sum(1 for x in d if x>0)/len(d):>7.0f}%{rb:>8}/{rv:<6}")
        else:
            print(f"  {L:<7}{len(ks):>5}{bm:>+11.3f}{vm:>+10.3f}{'(too few)':>13}")
    if allD:
        lo,hi=boot_ci(allD)
        print(f"\n  all sizes together: paired gain {st.mean(allD):+.3f}  [{lo:+.3f}, {hi:+.3f}]  n={len(allD)}")

if __name__=='__main__':
    main(sys.argv[1], sys.argv[2])

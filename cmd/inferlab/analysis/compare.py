import json, math, statistics as st, sys
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

names=sys.argv[1:]
import os
label={n:os.path.basename(n).replace('x-','').replace('.jsonl','') for n in names}
runs={n:load(n) for n in names}
keys=set.intersection(*[set(v) for v in runs.values()])
print(f"{len(keys)} positions common to all {len(names)} variants\n")

def fin(xs): return [x for x in xs if math.isfinite(x)]
def ci(xs):
    xs=fin(xs)
    if len(xs)<2: return float('nan')
    return 1.96*st.stdev(xs)/math.sqrt(len(xs))

print("OUTCOME: how well each proposal read the true leave")
print(f"  {'variant':<11}{'mean lift':>11}{'±95%':>8}{'median':>9}{'ruled out':>11}{'measured':>10}{'rank %ile':>11}")
for n in names:
    R=[runs[n][k] for k in keys]
    l=[r['liftBits'] for r in R]
    pct=[100*r['rank']/r['leaves'] for r in R if r['rank']>0 and r['leaves']>0]
    print(f"  {label[n]:<11}{st.mean(fin(l)):>+11.3f}{ci(l):>8.3f}{st.median(fin(l)):>+9.3f}"
          f"{sum(1 for r in R if r['ruledOut']):>11}{sum(1 for r in R if r['measured']):>10}"
          f"{st.median(pct):>10.2f}%")

print("\nPAIRED against the engine's current proposal (same position, same seed)")
base=names[0]
for n in names[1:]:
    d=[runs[n][k]['liftBits']-runs[base][k]['liftBits'] for k in keys
       if math.isfinite(runs[n][k]['liftBits']) and math.isfinite(runs[base][k]['liftBits'])]
    wins=sum(1 for x in d if x>0)
    print(f"  {label[n]:<11} {st.mean(d):+7.3f} ± {1.96*st.stdev(d)/math.sqrt(len(d)):.3f} bits   "
          f"better on {wins}/{len(d)} ({100*wins/len(d):.0f}%)")
    # did it rescue cases the baseline ruled out?
    resc=sum(1 for k in keys if runs[base][k]['ruledOut'] and not runs[n][k]['ruledOut'])
    lost=sum(1 for k in keys if not runs[base][k]['ruledOut'] and runs[n][k]['ruledOut'])
    mplus=sum(1 for k in keys if not runs[base][k]['measured'] and runs[n][k]['measured'])
    mminus=sum(1 for k in keys if runs[base][k]['measured'] and not runs[n][k]['measured'])
    print(f"              rescued {resc} ruled-out, newly ruled out {lost}; "
          f"measured the truth {mplus} more times, {mminus} fewer")

print("\nPROPOSAL QUALITY per refine round")
for n in names:
    byr=defaultdict(list)
    for k in keys:
        for d in runs[n][k].get('draws',[]): byr[d['Round']].append(d)
    if not byr: continue
    base0=st.mean(d['Measured'] for d in byr[0]) if byr.get(0) else float('nan')
    ref=[d for rd,ds in byr.items() if rd>=1 for d in ds]
    def lg(v): return math.log10(max(v,1e-30))
    def ranks(xs):
        o=sorted(range(len(xs)),key=lambda i:xs[i]); r=[0.0]*len(xs); i=0
        while i<len(o):
            j=i
            while j+1<len(o) and xs[o[j+1]]==xs[o[i]]: j+=1
            a=(i+j)/2+1
            for t in range(i,j+1): r[o[t]]=a
            i=j+1
        return r
    def pear(a,b):
        ma,mb=st.mean(a),st.mean(b)
        num=sum((x-ma)*(y-mb) for x,y in zip(a,b))
        da=math.sqrt(sum((x-ma)**2 for x in a)); db=math.sqrt(sum((y-mb)**2 for y in b))
        return num/(da*db) if da>0 and db>0 else float('nan')
    def sp(a,b): return pear(ranks(a),ranks(b))
    enr=st.mean(d['Measured'] for d in ref)/base0 if base0>0 else float('nan')
    rl=sp([lg(d['Predicted']) for d in ref],[lg(d['Measured']) for d in ref])
    rq=sp([lg(d['Q']) for d in ref],[lg(d['Measured']) for d in ref])
    print(f"  {label[n]:<11} refine draws {len(ref):>6}   enrichment vs blind {enr:>5.2f}x   "
          f"rho(lhat,w) {rl:+.3f}   rho(q,w) {rq:+.3f}")

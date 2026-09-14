#!/usr/bin/env python3
"""Is the imputed posterior in the right order around the true leave?

For every probed position: the true leave sits at some rank in the imputed
posterior. Leaves the model ranked ABOVE it should, when actually measured,
mostly turn out more likely than the truth; leaves ranked BELOW should mostly
turn out less likely. Where that fails is where the imputation is misordering
-- and the direction of the failure says whether it promotes junk over the
answer or buries the answer under junk.
"""
import json, math, sys, statistics as st
from collections import defaultdict

def load(path):
    out=[]
    for line in open(path):
        line=line.strip()
        if not line: continue
        r=json.loads(line)
        if r.get('err') or not r.get('probe'): continue
        r['probe']['Probes'] = r['probe'].get('Probes') or []
        if not r['probe']['Probes']: continue
        out.append(r)
    return out

def mean(xs): xs=list(xs); return st.mean(xs) if xs else float('nan')
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
def pearson(a,b):
    if len(a)<3: return float('nan')
    ma,mb=mean(a),mean(b)
    num=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    da=math.sqrt(sum((x-ma)**2 for x in a)); db=math.sqrt(sum((y-mb)**2 for y in b))
    return num/(da*db) if da>0 and db>0 else float('nan')
def spearman(a,b): return pearson(ranks(a),ranks(b))

def main(path):
    recs=load(path)
    print(f"{len(recs)} probed positions")
    if not recs: return

    # --- 1. the headline: what is actually above and below the truth --------
    print("\n1. Leaves the model ranked ABOVE the truth: are they really more likely?")
    print("   (measured likelihood of the probe vs measured likelihood of the truth)\n")
    rows=defaultdict(lambda: {'n':0,'really_better':0,'ratio':[]})
    for r in recs:
        p=r['probe']; ref=p['Reference']
        if ref['Measured']<=0: continue
        for q in p['Probes']:
            k=q['Stratum']
            rows[k]['n']+=1
            if q['Measured']>ref['Measured']: rows[k]['really_better']+=1
            rows[k]['ratio'].append(lg(q['Measured'])-lg(ref['Measured']))
    print(f"   {'stratum':<9}{'probes':>8}{'really better than truth':>26}{'median log10(w/w_truth)':>26}")
    for k,lab in (('top','top'),('above','above'),('below','below')):
        d=rows[k]
        if not d['n']: continue
        print(f"   {lab:<9}{d['n']:>8}{100*d['really_better']/d['n']:>25.1f}%{st.median(d['ratio']):>+26.2f}")
    print("\n   A correct ordering wants 'top' and 'above' near 100% and 'below' near 0%.")
    print("   'top' is the head of the imputed posterior -- what the simmer actually samples.")

    # --- 2. per-position verdict --------------------------------------------
    print("\n2. Per position: which way did the ordering go wrong?")
    junk_over_truth=0; truth_over_junk=0; fine=0; skipped=0
    for r in recs:
        p=r['probe']; ref=p['Reference']
        if ref['Measured']<=0: skipped+=1; continue
        top=[q for q in p['Probes'] if q['Stratum']=='top']
        if not top: skipped+=1; continue
        worse=sum(1 for q in top if q['Measured']<ref['Measured'])
        if worse>=len(top)*0.7: junk_over_truth+=1
        elif worse<=len(top)*0.3: fine+=1
        else: truth_over_junk+=1
    n=len(recs)-skipped
    print(f"   head of the posterior is mostly junk (>=70% of top probes measure below the truth): "
          f"{junk_over_truth}/{n} ({100*junk_over_truth/n:.0f}%)")
    print(f"   head of the posterior is mostly genuine (<=30% below the truth):                  "
          f"{fine}/{n} ({100*fine/n:.0f}%)")
    print(f"   mixed:                                                                          "
          f"{truth_over_junk}/{n}")

    # --- 3. ordering quality among the probes themselves ---------------------
    print("\n3. Rank agreement between imputed and measured likelihood, within a position")
    rhos=[]; 
    for r in recs:
        qs=r['probe']['Probes']
        if len(qs)<8: continue
        rhos.append(spearman([lg(q['Imputed']) for q in qs],[lg(q['Measured']) for q in qs]))
    rhos=[x for x in rhos if not math.isnan(x)]
    print(f"   Spearman rho(imputed, measured) over ~40 probes per position:")
    print(f"     median {st.median(rhos):+.3f}   mean {mean(rhos):+.3f}   "
          f"positions with rho<0: {sum(1 for x in rhos if x<0)}/{len(rhos)}")
    allq=[q for r in recs for q in r['probe']['Probes']]
    print(f"   pooled over all {len(allq)} probes: rho = "
          f"{spearman([lg(q['Imputed']) for q in allq],[lg(q['Measured']) for q in allq]):+.3f}")

    # --- 4. scale: imputed likelihoods above 1 -------------------------------
    over=[q for q in allq if q['Imputed']>1]
    print(f"\n4. Scale sanity: {len(over)}/{len(allq)} probed leaves carry an imputed likelihood > 1,")
    print(f"   which no probability can be. Median measured for those: {st.median(q['Measured'] for q in over) if over else float('nan'):.2e}")

    # --- 5. the truth itself ---------------------------------------------------
    print("\n5. The truth: imputed vs measured")
    tr=[r['probe']['Reference'] for r in recs if r['probe']['Reference']['Imputed']>0 and r['probe']['Reference']['Measured']>0]
    under=[lg(t['Measured'])-lg(t['Imputed']) for t in tr]
    print(f"   n={len(tr)} with both; median log10(measured/imputed) = {st.median(under):+.2f}")
    print(f"   -> the model under-rates the true leave by a median factor of {10**st.median(under):.0f}x")
    print(f"   under-rated in {sum(1 for u in under if u>0)}/{len(under)}, over-rated in {sum(1 for u in under if u<0)}")

if __name__=='__main__':
    for p in sys.argv[1:]:
        print('='*72); print(p); print('='*72); main(p)

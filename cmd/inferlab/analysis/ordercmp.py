#!/usr/bin/env python3
"""Side-by-side ordering diagnostics for several inferlab probe runs."""
import json, math, sys, statistics as st
from collections import defaultdict
def lg(v): return math.log10(max(v,1e-30))
def load(f):
    out=[]
    for line in open(f):
        line=line.strip()
        if not line: continue
        r=json.loads(line)
        if r.get('err') or not r.get('probe'): continue
        r['probe']['Probes']=r['probe'].get('Probes') or []
        if r['probe']['Probes']: out.append(r)
    return out
def ranks(xs):
    o=sorted(range(len(xs)),key=lambda i:xs[i]); r=[0.0]*len(xs); i=0
    while i<len(o):
        j=i
        while j+1<len(o) and xs[o[j+1]]==xs[o[i]]: j+=1
        a=(i+j)/2+1
        for t in range(i,j+1): r[o[t]]=a
        i=j+1
    return r
def spear(a,b):
    if len(a)<3: return float('nan')
    ra,rb=ranks(a),ranks(b); ma,mb=st.mean(ra),st.mean(rb)
    num=sum((x-ma)*(y-mb) for x,y in zip(ra,rb))
    da=math.sqrt(sum((x-ma)**2 for x in ra)); db=math.sqrt(sum((y-mb)**2 for y in rb))
    return num/(da*db) if da>0 and db>0 else float('nan')

def stats(recs):
    s={}
    st_better=defaultdict(lambda:[0,0]); cal=defaultdict(list)
    junk=fine=n=0; rhos=[]; over1=tot=0; tru=[]; gap=[]; spread=[]; mgap=[]
    for r in recs:
        p=r['probe']; ref=p['Reference']
        if ref['Measured']>0:
            for q in p['Probes']:
                st_better[q['Stratum']][0]+=1
                if q['Measured']>ref['Measured']: st_better[q['Stratum']][1]+=1
            top=[q for q in p['Probes'] if q['Stratum']=='top']
            if top:
                n+=1; w=sum(1 for q in top if q['Measured']<ref['Measured'])
                if w>=0.7*len(top): junk+=1
                elif w<=0.3*len(top): fine+=1
        if ref['Imputed']>0 and ref['Measured']>0: tru.append(lg(ref['Measured'])-lg(ref['Imputed']))
        for q in p['Probes']:
            tot+=1
            if q['Imputed']>1: over1+=1
            if q['Imputed']>0 and q['Measured']>0: cal[q['Stratum']].append(lg(q['Measured'])-lg(q['Imputed']))
        qs=p['Probes']
        if len(qs)>=8: rhos.append(spear([lg(q['Imputed']) for q in qs],[lg(q['Measured']) for q in qs]))
        # the number that decides it: where the model puts the head relative to
        # the truth, against where the mini-sims put it
        topi=[lg(q['Imputed']) for q in qs if q['Stratum']=='top' and q['Imputed']>0]
        topm=[lg(q['Measured']) for q in qs if q['Stratum']=='top' and q['Measured']>0]
        imp=[lg(q['Imputed']) for q in qs if q['Imputed']>0]
        if topi and ref['Imputed']>0: gap.append(st.mean(topi)-lg(ref['Imputed']))
        if topm and ref['Measured']>0: mgap.append(st.mean(topm)-lg(ref['Measured']))
        if len(imp)>=8: spread.append(st.pstdev(imp))
    rhos=[x for x in rhos if not math.isnan(x)]
    s['positions']=len(recs)
    for k in ('top','above','below'):
        a,b=st_better[k]; s['better_'+k]=100*b/a if a else float('nan')
        s['cal_'+k]=st.median(cal[k]) if cal[k] else float('nan')
    s['head_junk']=100*junk/n if n else float('nan'); s['head_fine']=100*fine/n if n else float('nan')
    s['rho']=st.median(rhos) if rhos else float('nan')
    s['over1']=100*over1/tot if tot else float('nan')
    s['truth_under']=st.median(tru) if tru else float('nan')
    s['gap']=st.median(gap) if gap else float('nan')
    s['mgap']=st.median(mgap) if mgap else float('nan')
    s['spread']=st.median(spread) if spread else float('nan')
    return s

names=sys.argv[1:]
import os
label={n:os.path.basename(n).replace('x-','').replace('.jsonl','') for n in names}
S={n:stats(load(n)) for n in names}
rows=[('positions probed','positions','{:.0f}'),
      ('head: % really better than truth','better_top','{:.1f}%'),
      ('rest above: % really better','better_above','{:.1f}%'),
      ('below: % really better','better_below','{:.1f}%'),
      ('positions where head is mostly junk','head_junk','{:.0f}%'),
      ('positions where head is mostly genuine','head_fine','{:.0f}%'),
      ('within-position rho(imputed,measured)','rho','{:+.3f}'),
      ('head over-rated by (log10)','cal_top','{:+.2f}'),
      ('below calibration (log10)','cal_below','{:+.2f}'),
      ('truth under-rated by (log10)','truth_under','{:+.2f}'),
      ('% probes with imputed > 1','over1','{:.1f}%'),
      ('imputed spread within position (sd log10)','spread','{:.2f}'),
      ('head above truth, imputed (log10)','gap','{:+.2f}'),
      ('head above truth, MEASURED (log10)','mgap','{:+.2f}')]
print(f"  {'':<40}"+''.join(f"{label[n]:>13}" for n in names))
for lab,key,fmt in rows:
    print(f"  {lab:<40}"+''.join(f"{fmt.format(S[n][key]):>13}" for n in names))

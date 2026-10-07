#!/usr/bin/env python3
"""
Follow-up checks on the gate analysis (paper Section 5.6).

Run scripts/06_gate_analysis.py first; this reads its gate_analysis.json and
gate_per_pair.json from TAFNET_RESULTS/gate_analysis/. Reports the stability of
the gate across fold models, the relation of each coefficient to the predicted
probability, and whether the relation to the rate of MMSE change survives
adjustment for that probability. Prints to stdout.
"""
from common import PAIRS_CSV, SIDECAR_CSV, RESULTS, fold_json, out_path, load_oof_and_heldout
import json,os,numpy as np
from scipy.stats import spearmanr,rankdata
a=json.load(open(os.path.join(RESULTS,'gate_analysis','gate_analysis.json'))); p=json.load(open(os.path.join(RESULTS,'gate_analysis','gate_per_pair.json')))
C=['alpha','beta','gamma']
print("per-fold mean gate (alpha, beta, gamma): validation | held-out")
for f in a['per_fold']: print(" fold",f['fold'],[round(x,3) for x in f['val_mean_gate']],"|",[round(x,3) for x in f['heldout_mean_gate']])
print("SD across folds of held-out gate, mean over pairs:",a['heldout_fold_agreement'])
def arr(R,k): return np.array([r[k] for r in R],float)
def boot(R,fn,B=2000,seed=20260907):
    s=np.array([r['subject'] for r in R]); u=np.array(sorted(set(s))); rows={x:np.where(s==x)[0] for x in u}
    rng=np.random.default_rng(seed); out=[]
    for _ in range(B):
        idx=np.concatenate([rows[x] for x in rng.choice(u,len(u))]); v=fn(idx)
        if np.isfinite(v): out.append(v)
    return np.percentile(out,[2.5,97.5])
def pcorr(x,y,z):
    rx,ry,rz=rankdata(x),rankdata(y),rankdata(z)
    ex=rx-np.polyval(np.polyfit(rz,rx,1),rz); ey=ry-np.polyval(np.polyfit(rz,ry,1),rz)
    return np.corrcoef(ex,ey)[0,1]
cv=p['cv']; ho=p['heldout']
print("\nCV: Spearman gate vs predicted probability, within each fold")
for c in C:
    print(" ",c,[round(spearmanr(arr([r for r in cv if r['fold']==k],c),arr([r for r in cv if r['fold']==k],'prob'))[0],3) for k in sorted({r['fold'] for r in cv})])
print("\nHeld-out (gate and probability averaged over five folds)")
pr=arr(ho,'prob'); ms=arr(ho,'mmse_slope_per_year'); ok=np.isfinite(ms)
r0=spearmanr(pr[ok],ms[ok])[0]; print(f"  prob vs MMSE slope: rho={r0:.3f} n={ok.sum()}")
for c in C:
    g=arr(ho,c)
    r=spearmanr(g,pr)[0]; lo,hi=boot(ho,lambda i:spearmanr(g[i],pr[i])[0])
    i_ok=np.where(ok)[0]; Rk=[ho[i] for i in i_ok]; gg=g[ok]; mm=ms[ok]; pp=pr[ok]
    pc=pcorr(gg,mm,pp); plo,phi=boot(Rk,lambda i:pcorr(gg[i],mm[i],pp[i]))
    print(f"  {c}: gate vs prob rho={r:+.3f} [{lo:+.3f},{hi:+.3f}] | gate vs MMSE slope rho={spearmanr(gg,mm)[0]:+.3f} | partial given prob={pc:+.3f} [{plo:+.3f},{phi:+.3f}]")

print("\nExtra: spread across patients (fold-averaged, held-out), alpha-beta correlation, contrast log(gamma/(alpha+beta)) vs prob")
for c in C: print("  SD across patients held-out",c,round(arr(ho,c).std(ddof=1),4))
for nm,R in (("cv",cv),("heldout",ho)):
    al,be,ga,q=arr(R,'alpha'),arr(R,'beta'),arr(R,'gamma'),arr(R,'prob')
    con=np.log(np.clip(ga,1e-6,None)/np.clip(al+be,1e-6,None))
    r1=spearmanr(al,be)[0]; r2=spearmanr(con,q)[0]; lo,hi=boot(R,lambda i:spearmanr(con[i],q[i])[0])
    print(f"  {nm}: rho(alpha,beta)={r1:+.3f}  rho(contrast,prob)={r2:+.3f} [{lo:+.3f},{hi:+.3f}]")
print("  CV within-fold rho(contrast,prob):",[round(float(spearmanr(np.log(np.clip(arr(F,'gamma'),1e-6,None)/np.clip(arr(F,'alpha')+arr(F,'beta'),1e-6,None)),arr(F,'prob'))[0]),3) for F in [[r for r in cv if r['fold']==k] for k in sorted({r['fold'] for r in cv})]])
ms=arr(cv,'mmse_slope_per_year'); ok=np.isfinite(ms); Rk=[cv[i] for i in np.where(ok)[0]]
for c in C:
    g=arr(Rk,c); m=arr(Rk,'mmse_slope_per_year'); q=arr(Rk,'prob')
    pc=pcorr(g,m,q); lo,hi=boot(Rk,lambda i:pcorr(g[i],m[i],q[i])); print(f"  CV {c}: partial rho with MMSE slope given prob={pc:+.3f} [{lo:+.3f},{hi:+.3f}]")

#!/usr/bin/env python3
"""
DeLong tests, Brier-score differences, score distributions and the convergent analyses (paper Sections 5.2, 5.5, 5.7).

Risk against the rate of MMSE change and against time to conversion, baseline
MMSE alone, and the Alzheimer's-only sensitivity analysis. Prints to stdout.
"""
from common import PAIRS_CSV, SIDECAR_CSV, RESULTS, fold_json, out_path, load_oof_and_heldout
import json, numpy as np, pandas as pd, sys
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.stats import rankdata, spearmanr, norm
M=["ResNet3D-18","DenseNet3D-121","Siamese-Subtract","CNN-LSTM","TAFNet-InitialOnly","TAFNet-Full"]
df=pd.read_csv(PAIRS_CSV)
subj=df.subject.astype(str).to_numpy(); lab=df.label.astype(int).to_numpy()
pool=sorted(set(subj)); st=np.array([lab[subj==s].max() for s in pool])
keep,test=train_test_split(pool,test_size=0.15,random_state=42,stratify=st); keep=sorted(keep); ts=set(test)
ti=np.array([i for i,s in enumerate(subj) if s in ts])
kst=np.array([lab[subj==s].max() for s in keep]); ci=[]
for tr,va in StratifiedKFold(5,shuffle=True,random_state=42).split(keep,kst):
    vs={keep[i] for i in va}; ci+=[i for i,s in enumerate(subj) if s in vs]
ci=np.array(ci)
cv={m:np.concatenate([json.load(open(fold_json(m,k)))["predictions"]["y_pred"] for k in range(1,6)]) for m in M}
ho={m:np.mean([json.load(open(fold_json(m,k)))["test_y_pred"] for k in range(1,6)],0) for m in M}
def auc(y,p):
    r=rankdata(p); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def mid(x): return rankdata(x)
def delong(y,p1,p2):
    o=np.argsort(-y); y=y[o]; P=np.vstack([p1,p2])[:,o]; m=int(y.sum()); n=len(y)-m
    tx=np.array([mid(P[r,:m]) for r in range(2)]); ty=np.array([mid(P[r,m:]) for r in range(2)]); tz=np.array([mid(P[r]) for r in range(2)])
    a=tz[:,:m].sum(1)/(m*n)-(m+1)/(2*n); v01=(tz[:,:m]-tx)/n; v10=1-(tz[:,m:]-ty)/m
    c=np.cov(v01)/m+np.cov(v10)/n; v=c[0,0]+c[1,1]-2*c[0,1]; d=a[0]-a[1]; return d,2*norm.sf(abs(d/np.sqrt(v)))
def bs(s,B=10000,seed=20260907):
    u=np.array(sorted(set(s))); rows={x:np.where(s==x)[0] for x in u}; rng=np.random.default_rng(seed)
    return [np.concatenate([rows[x] for x in rng.choice(u,len(u))]) for _ in range(B)]
out={}
for name,idx,P in [("CV",ci,cv),("HO",ti,ho)]:
    y=lab[idx].astype(float); s=subj[idx]; B=bs(s); d=df.iloc[idx].reset_index(drop=True)
    print("\n==",name,len(y))
    for a,b in [("TAFNet-Full","CNN-LSTM"),("TAFNet-Full","TAFNet-InitialOnly"),("TAFNet-Full","Siamese-Subtract"),("TAFNet-Full","DenseNet3D-121"),("TAFNet-Full","ResNet3D-18"),("CNN-LSTM","DenseNet3D-121")]:
        dd,p=delong(y,P[a],P[b]); print(f" DeLong {a} vs {b}: d={dd:+.3f} p={p:.4f}  r={np.corrcoef(P[a],P[b])[0,1]:.3f}")
    for a,b in [("TAFNet-Full","CNN-LSTM"),("TAFNet-Full","Siamese-Subtract"),("TAFNet-Full","TAFNet-InitialOnly")]:
        br=lambda i,m:np.mean((P[m][i]-y[i])**2); al=np.arange(len(y)); d0=br(al,a)-br(al,b)
        x=np.array([br(i,a)-br(i,b) for i in B]); lo,hi=np.percentile(x,[2.5,97.5]); p=min(1,2*min((x<=0).mean(),(x>=0).mean()))
        print(f" dBrier {a} vs {b}: {d0:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}")
    for m in ["TAFNet-Full","CNN-LSTM"]:
        print(f" scores {m}: median {np.median(P[m]):.3f} pct<0.01 {100*(P[m]<0.01).mean():.1f} mean {P[m].mean():.3f}")
    p=P["TAFNet-Full"]
    def spear(mask,xv):
        mask=np.asarray(mask); r0=spearmanr(p[mask],xv[mask])[0]
        x=[]
        for i in B[:5000]:
            mm=mask[i]
            if mm.sum()>5: x.append(spearmanr(p[i][mm],xv[i][mm])[0])
        lo,hi=np.nanpercentile(x,[2.5,97.5]); return r0,lo,hi,int(mask.sum())
    sl=d.mmse_slope_per_year.to_numpy(float); print(" rho risk vs MMSE slope",spear(~np.isnan(sl),sl))
    dt=d.days_to_conversion.to_numpy(float); print(" rho risk vs time to conv (converters)",spear((y==1)&~np.isnan(dt),dt))
    mb=d.mmse_baseline.to_numpy(float); ok=~np.isnan(mb); a0=auc(y[ok],-mb[ok]); x=[auc(y[i][ok[i]],-mb[i][ok[i]]) for i in B[:5000]]
    print(f" MMSE alone AUC {a0:.3f} [{np.percentile(x,2.5):.3f},{np.percentile(x,97.5):.3f}] n={ok.sum()}; rho risk vs MMSE base {spearmanr(p[ok],mb[ok])[0]:.2f}")
    et=d.etiology.fillna('').astype(str).to_numpy(); keepm=(y==0)|(et=='AD'); a1=auc(y[keepm],p[keepm]); x=[auc(y[i][keepm[i]],p[i][keepm[i]]) for i in B[:5000]]
    print(f" AD-only positives AUC {a1:.3f} [{np.percentile(x,2.5):.3f},{np.percentile(x,97.5):.3f}] npos={int(y[keepm].sum())}; all AUC {auc(y,p):.3f}; etiologies {pd.Series(et[y==1]).value_counts().to_dict()}")

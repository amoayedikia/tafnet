#!/usr/bin/env python3
"""
TAFNet against CNN-LSTM within acquisition subgroups of the cross-validation pairs (paper Section 5.7).

Mixed against same acceleration, interval, and scanner vendor, with interaction
tests. Needs TAFNET_SIDECAR_CSV (image_id, vendor, series). Prints to stdout.
"""
from common import PAIRS_CSV, SIDECAR_CSV, RESULTS, fold_json, out_path, load_oof_and_heldout
import json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.stats import rankdata
df=pd.read_csv(PAIRS_CSV)
sc=pd.read_csv(SIDECAR_CSV).set_index("image_id")
acc=lambda s: any(k in str(s).upper() for k in ["SENSE","GRAPPA","ACCEL","ASSET"])
a1=df.scan1.map(lambda i:acc(sc.series[i])); a2=df.scan2.map(lambda i:acc(sc.series[i]))
v1=df.scan1.map(lambda i:sc.vendor[i]); v2=df.scan2.map(lambda i:sc.vendor[i])
df["mixacc"]=(a1!=a2); df["vendor"]=np.where(v1==v2,v1,"mixed"); df["short"]=df.interval_days<=396
subj=df.subject.astype(str).to_numpy(); lab=df.label.astype(int).to_numpy()
pool=sorted(set(subj)); st=np.array([lab[subj==s].max() for s in pool])
keep,test=train_test_split(pool,test_size=0.15,random_state=42,stratify=st); keep=sorted(keep)
kst=np.array([lab[subj==s].max() for s in keep]); idx=[]
for tr,va in StratifiedKFold(5,shuffle=True,random_state=42).split(keep,kst):
    vs={keep[i] for i in va}; idx+= [i for i,s in enumerate(subj) if s in vs]
idx=np.array(idx); y=lab[idx].astype(float); s=subj[idx]
P={m:np.concatenate([json.load(open(fold_json(m,k)))["predictions"]["y_pred"] for k in range(1,6)]) for m in ["TAFNet-Full","CNN-LSTM"]}
d=df.iloc[idx].reset_index(drop=True)
print("all pairs: mixacc",int(df.mixacc.sum()),"short(<=396d)",int(df.short.sum()),"vendor",df.vendor.value_counts().to_dict())
def auc(y,p):
    if y.sum()==0 or y.sum()==len(y): return np.nan
    r=rankdata(p); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
u=np.array(sorted(set(s))); rows={x:np.where(s==x)[0] for x in u}; rng=np.random.default_rng(20260907)
B=[np.concatenate([rows[x] for x in rng.choice(u,len(u))]) for _ in range(5000)]
def grp(mask,name):
    mask=np.asarray(mask); n=int(mask.sum()); npos=int(y[mask].sum())
    at=auc(y[mask],P["TAFNet-Full"][mask]); ac=auc(y[mask],P["CNN-LSTM"][mask])
    dd=np.array([auc(y[b][mask[b]],P["TAFNet-Full"][b][mask[b]])-auc(y[b][mask[b]],P["CNN-LSTM"][b][mask[b]]) for b in B]); dd=dd[~np.isnan(dd)]
    lo,hi=np.percentile(dd,[2.5,97.5]); print(f"{name:28s} n={n:3d} pos={npos:3d} TAFNet {at:.3f} CNN-LSTM {ac:.3f} d={at-ac:+.3f} [{lo:+.3f},{hi:+.3f}]"); return dd
def inter(m1,m2,name):
    m1=np.asarray(m1);m2=np.asarray(m2)
    f=lambda b,m: auc(y[b][m[b]],P["TAFNet-Full"][b][m[b]])-auc(y[b][m[b]],P["CNN-LSTM"][b][m[b]])
    x=np.array([f(b,m1)-f(b,m2) for b in B]); x=x[~np.isnan(x)]; lo,hi=np.percentile(x,[2.5,97.5])
    print(f"   interaction {name}: [{lo:+.3f},{hi:+.3f}] p={min(1,2*min((x<=0).mean(),(x>=0).mean())):.3f}")
grp(d.mixacc.values,"mixed acceleration"); grp(~d.mixacc.values,"same acceleration"); inter(d.mixacc.values,~d.mixacc.values,"mixed vs same")
grp(d.short.values,"interval <=13 months"); grp(~d.short.values,"interval >13 months"); inter(d.short.values,~d.short.values,"short vs long")
for v in ["Siemens","GE","Philips","mixed"]: grp((d.vendor==v).values,"vendor "+v)

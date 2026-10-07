#!/usr/bin/env python3
"""
AUC of every method, paired differences and the equivalence interval (paper Sections 5.2, 5.3).

Pooled out-of-fold cross-validation and the held-out five-fold ensemble, with
10,000 participant-clustered bootstrap resamples (seed 20260907). The 90%
interval of TAFNet-Full minus CNN-LSTM is the one used for the equivalence test
(margin 0.03 AUC). Also prints the mean and SD of the per-fold AUCs.
Writes auc_comparisons.json (read by figures.py).
"""
from common import PAIRS_CSV, SIDECAR_CSV, RESULTS, fold_json, out_path, load_oof_and_heldout
import json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.stats import rankdata
M=["ResNet3D-18","DenseNet3D-121","Siamese-Subtract","CNN-LSTM","TAFNet-InitialOnly","TAFNet-Full"]
df=pd.read_csv(PAIRS_CSV)
subj=df.subject.astype(str).to_numpy(); lab=df.label.astype(int).to_numpy()
pool=sorted(set(subj)); strata=np.array([lab[subj==s].max() for s in pool])
keep,test=train_test_split(pool,test_size=0.15,random_state=42,stratify=strata)
ts=set(test); test_idx=[i for i,s in enumerate(subj) if s in ts]; keep=sorted(keep)
kst=np.array([lab[subj==s].max() for s in keep])
folds=[]
for tr,va in StratifiedKFold(5,shuffle=True,random_state=42).split(keep,kst):
    vs={keep[i] for i in va}; folds.append([i for i,s in enumerate(subj) if s in vs])
def auc(y,p):
    r=rankdata(p); n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
cv={};ho={};pf={};pfh={}
for m in M:
    ps=[];hs=[];a=[];ah=[]
    for k in range(5):
        j=json.load(open(fold_json(m,k+1)))
        yt=np.array(j["predictions"]["y_true"],float)
        assert np.array_equal(yt,lab[folds[k]]),(m,k,"CV split mismatch")
        assert np.array_equal(np.array(j["test_y_true"],float),lab[test_idx]),(m,k,"test mismatch")
        ps.append(np.array(j["predictions"]["y_pred"],float)); hs.append(np.array(j["test_y_pred"],float))
        a.append(j["metrics"]["AUC"]); ah.append(j["test_metrics"]["AUC"])
    cv[m]=np.concatenate(ps); ho[m]=np.mean(hs,0); pf[m]=a; pfh[m]=ah
cvi=np.concatenate(folds); ycv=lab[cvi].astype(float); scv=subj[cvi]
yho=lab[test_idx].astype(float); sho=subj[test_idx]
print("split verified: CV",len(ycv),"pairs",len(set(scv)),"subj",int(ycv.sum()),"pos | held-out",len(yho),len(set(sho)),int(yho.sum()))
def boot(y,s,P,B=10000,seed=20260907):
    u=np.array(sorted(set(s))); rows={x:np.where(s==x)[0] for x in u}; rng=np.random.default_rng(seed); out=[]
    while len(out)<B:
        idx=np.concatenate([rows[x] for x in rng.choice(u,len(u))]); yy=y[idx]
        if yy.sum()==0 or yy.sum()==len(yy): continue
        out.append([auc(yy,P[m][idx]) for m in M])
    return np.array(out)
res={}
for name,y,s,P in [("CV pooled OOF",ycv,scv,cv),("Held-out ensemble",yho,sho,ho)]:
    b=boot(y,s,P); print("\n==",name)
    res[name]={}
    for i,m in enumerate(M):
        a=auc(y,P[m]); lo,hi=np.percentile(b[:,i],[2.5,97.5]); print(f"{m:20s} AUC {a:.3f} [{lo:.3f},{hi:.3f}]")
        res[name][m]=dict(auc=a,lo=lo,hi=hi)
    def cmp(a_,b_):
        i,j=M.index(a_),M.index(b_); d=auc(y,P[a_])-auc(y,P[b_]); dd=b[:,i]-b[:,j]
        lo,hi=np.percentile(dd,[2.5,97.5]); l90,h90=np.percentile(dd,[5,95]); p=min(1,2*min((dd<=0).mean(),(dd>=0).mean()))
        print(f"  {a_} - {b_}: d={d:+.3f} 95%[{lo:+.3f},{hi:+.3f}] 90%[{l90:+.3f},{h90:+.3f}] p={p:.4f} SE={dd.std():.4f}")
        res[name][f"{a_} vs {b_}"]=dict(d=d,lo=lo,hi=hi,l90=l90,h90=h90,p=p)
    for a_,b_ in [("TAFNet-Full","CNN-LSTM"),("TAFNet-Full","DenseNet3D-121"),("TAFNet-Full","ResNet3D-18"),("TAFNet-Full","Siamese-Subtract"),("TAFNet-Full","TAFNet-InitialOnly"),("CNN-LSTM","DenseNet3D-121"),("CNN-LSTM","ResNet3D-18"),("CNN-LSTM","Siamese-Subtract"),("CNN-LSTM","TAFNet-InitialOnly"),("Siamese-Subtract","DenseNet3D-121")]: cmp(a_,b_)
print("\nper-fold CV AUC mean/SD(ddof0) | per-fold held-out mean/SD")
for m in M: print(f"{m:20s} {np.mean(pf[m]):.3f} {np.std(pf[m]):.3f} {[round(x,3) for x in pf[m]]} | {np.mean(pfh[m]):.3f} {np.std(pfh[m]):.3f}")
print("corr TAFNet/CNN-LSTM preds: CV",np.corrcoef(cv['TAFNet-Full'],cv['CNN-LSTM'])[0,1],"HO",np.corrcoef(ho['TAFNet-Full'],ho['CNN-LSTM'])[0,1])
json.dump(res,open(out_path("auc_comparisons.json"),"w"),indent=1,default=float)

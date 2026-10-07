#!/usr/bin/env python3
"""
The eight Temporal Fusion Module ablations against the full model (paper Section 5.4, Table 4).

Each variant is trained with scripts/run_ablations.sh into
TAFNET_RESULTS/ablation/<variant>/. Same split, bootstrap and seed as
auc_comparisons.py. Differences are variant minus full model. Writes ablations.json.
"""
from common import PAIRS_CSV, SIDECAR_CSV, RESULTS, fold_json, out_path, load_oof_and_heldout
import json, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.stats import rankdata
V=["gate_position","no_residual","branch_diff","branch_attn","branch_concat","branch_no_diff","branch_no_attn","branch_no_concat"]
M=["full"]+V
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
cv={};ho={};pf={};pfh={};sig={}
for m in M:
    ps=[];hs=[];a=[];ah=[];sg=set()
    for k in range(5):
        j=json.load(open(fold_json("TAFNet-Full",k+1,None if m=="full" else m)))
        assert np.array_equal(np.array(j["predictions"]["y_true"],float),lab[folds[k]]),(m,k,"CV split mismatch")
        assert np.array_equal(np.array(j["test_y_true"],float),lab[test_idx]),(m,k,"test mismatch")
        ps.append(np.array(j["predictions"]["y_pred"],float)); hs.append(np.array(j["test_y_pred"],float))
        a.append(j["metrics"]["AUC"]); ah.append(j["test_metrics"]["AUC"]); sg.add(j.get("signature"))
    cv[m]=np.concatenate(ps); ho[m]=np.mean(hs,0); pf[m]=a; pfh[m]=ah; sig[m]=sorted(sg)
cvi=np.concatenate(folds); ycv=lab[cvi].astype(float); scv=subj[cvi]
yho=lab[test_idx].astype(float); sho=subj[test_idx]
print("split verified: CV",len(ycv),"pairs",len(set(scv)),"subj",int(ycv.sum()),"pos | held-out",len(yho),len(set(sho)),int(yho.sum()))
print("signatures:",sig)
def boot(y,s,P,B=10000,seed=20260907):
    u=np.array(sorted(set(s))); rows={x:np.where(s==x)[0] for x in u}; rng=np.random.default_rng(seed); out=[]
    while len(out)<B:
        idx=np.concatenate([rows[x] for x in rng.choice(u,len(u))]); yy=y[idx]
        if yy.sum()==0 or yy.sum()==len(yy): continue
        out.append([auc(yy,P[m][idx]) for m in M])
    return np.array(out)
res={"fold_auc":pf,"fold_auc_heldout":pfh}
for name,y,s,P in [("CV pooled OOF",ycv,scv,cv),("Held-out ensemble",yho,sho,ho)]:
    b=boot(y,s,P); print("\n==",name); res[name]={}
    for i,m in enumerate(M):
        a=auc(y,P[m]); lo,hi=np.percentile(b[:,i],[2.5,97.5])
        if m=="full":
            print(f"{m:18s} AUC {a:.3f} [{lo:.3f},{hi:.3f}]"); res[name][m]=dict(auc=a,lo=lo,hi=hi); continue
        d=a-auc(y,P["full"]); dd=b[:,i]-b[:,0]; dl,dh=np.percentile(dd,[2.5,97.5])
        p=min(1,2*min((dd<=0).mean(),(dd>=0).mean()))
        r=np.corrcoef(P[m],P["full"])[0,1]
        print(f"{m:18s} AUC {a:.3f} [{lo:.3f},{hi:.3f}]  variant-full d={d:+.3f} [{dl:+.3f},{dh:+.3f}] p={p:.3f}  r(pred)={r:.3f}")
        res[name][m]=dict(auc=a,lo=lo,hi=hi,d=d,dlo=dl,dhi=dh,p=p,r=r)
print("\nper-fold CV AUC mean (SD ddof0) folds | mean of per-fold held-out")
for m in M: print(f"{m:18s} {np.mean(pf[m]):.3f} ({np.std(pf[m]):.3f}) {[round(x,3) for x in pf[m]]} | {np.mean(pfh[m]):.3f}")
json.dump(res,open(out_path("ablations.json"),"w"),indent=1,default=float)

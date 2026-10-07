#!/usr/bin/env python3
"""
Draws the six result figures of the paper (Figures 5 to 10).

Run auc_comparisons.py and scripts/06_gate_analysis.py first. Writes fig_roc,
fig_forest, fig_calibration, fig_gate, fig_scores and fig_validity as PDF into
TAFNET_ANALYSIS_OUT/figures/, with PNG previews.
"""
from common import PAIRS_CSV, SIDECAR_CSV, RESULTS, fold_json, out_path, load_oof_and_heldout
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import spearmanr
M=["ResNet3D-18","DenseNet3D-121","Siamese-Subtract","CNN-LSTM","TAFNet-InitialOnly","TAFNet-Full"]
NAME={"TAFNet-Full":"TAFNet"}
INK="#0b0b0b"; INK2="#52514e"; MUTED="#8a8983"; GRID="#e6e5e1"
COL={"TAFNet-Full":"#2a78d6","CNN-LSTM":"#eb6834","Siamese-Subtract":"#1baf7a",
     "TAFNet-InitialOnly":"#4a3aa7","ResNet3D-18":"#008300","DenseNet3D-121":"#eda100"}
LS={"TAFNet-Full":"-","CNN-LSTM":"--","Siamese-Subtract":":","TAFNet-InitialOnly":"-","ResNet3D-18":"--","DenseNet3D-121":":"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.edgecolor":GRID,"axes.linewidth":0.6,
 "axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,"xtick.labelsize":7.5,"ytick.labelsize":7.5,
 "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.5,"axes.axisbelow":True,"axes.spines.top":False,"axes.spines.right":False,
 "axes.titlesize":9,"axes.titlelocation":"left","axes.titlecolor":INK,"legend.frameon":False,"legend.fontsize":7.5,
 "xtick.major.size":0,"ytick.major.size":0,"pdf.fonttype":42,"savefig.bbox":"tight","savefig.pad_inches":0.04})

# ---- data: same split reconstruction as stats_v2.py ----
df=pd.read_csv(PAIRS_CSV)
subj=df.subject.astype(str).to_numpy(); lab=df.label.astype(int).to_numpy()
pool=sorted(set(subj)); strata=np.array([lab[subj==s].max() for s in pool])
keep,test=train_test_split(pool,test_size=0.15,random_state=42,stratify=strata)
ts=set(test); test_idx=[i for i,s in enumerate(subj) if s in ts]; keep=sorted(keep)
kst=np.array([lab[subj==s].max() for s in keep]); folds=[]
for tr,va in StratifiedKFold(5,shuffle=True,random_state=42).split(keep,kst):
    vs={keep[i] for i in va}; folds.append([i for i,s in enumerate(subj) if s in vs])
cv={};ho={}
for m in M:
    ps=[];hs=[]
    for k in range(5):
        j=json.load(open(fold_json(m,k+1)))
        assert np.array_equal(np.array(j["predictions"]["y_true"],float),lab[folds[k]]),(m,k)
        assert np.array_equal(np.array(j["test_y_true"],float),lab[test_idx]),(m,k)
        ps.append(np.array(j["predictions"]["y_pred"],float)); hs.append(np.array(j["test_y_pred"],float))
    cv[m]=np.concatenate(ps); ho[m]=np.mean(hs,0)
cvi=np.concatenate(folds); ycv=lab[cvi]; yho=lab[test_idx]
S=json.load(open(out_path("auc_comparisons.json")))
def save(fig,name): fig.savefig(out_path("figures",name+".pdf")); fig.savefig(out_path("figures","preview",name+".png"),dpi=110); plt.close(fig)

# ---- 1. ROC, held-out ensemble ----
fig,ax=plt.subplots(1,2,figsize=(6.9,3.3),sharey=True)
for a,title,ms in [(ax[0],"Single-timepoint",["ResNet3D-18","DenseNet3D-121","TAFNet-InitialOnly"]),(ax[1],"Two-timepoint",["Siamese-Subtract","TAFNet-Full","CNN-LSTM"])]:
    a.plot([0,1],[0,1],color=MUTED,lw=0.6)
    for m in ms:
        f,t,_=roc_curve(yho,ho[m]); au=roc_auc_score(yho,ho[m])
        assert abs(au-S["Held-out ensemble"][m]["auc"])<1e-9
        a.plot(f,t,color=COL[m],ls=LS[m],lw=1.8,solid_capstyle="round",label=f"{NAME.get(m,m)}  {au:.3f}")
    a.set_title(title); a.set_xlabel("False positive rate"); a.set_xlim(-0.01,1.01); a.set_ylim(-0.01,1.01); a.set_aspect("equal")
    a.legend(loc="lower right",title="Held-out AUC",title_fontsize=7.5,handlelength=2.6)
ax[0].set_ylabel("True positive rate")
fig.tight_layout(); save(fig,"fig_roc")

# ---- 2. Forest of paired AUC differences ----
rows=[("TAFNet-Full","ResNet3D-18"),("TAFNet-Full","DenseNet3D-121"),("TAFNet-Full","Siamese-Subtract"),("TAFNet-Full","TAFNet-InitialOnly"),("TAFNet-Full","CNN-LSTM"),("CNN-LSTM","DenseNet3D-121")]
fig,ax=plt.subplots(1,2,figsize=(6.9,2.9),sharex=True,sharey=True)
for a,title,key in [(ax[0],"Held-out (85 pairs)","Held-out ensemble"),(ax[1],"Cross-validation, pooled (519 pairs)","CV pooled OOF")]:
    a.axvline(0,color=INK2,lw=0.7)
    for i,(x,y) in enumerate(rows):
        r=S[key][f"{x} vs {y}"]; sig=r["lo"]>0 or r["hi"]<0; c="#2a78d6" if sig else MUTED
        a.plot([r["lo"],r["hi"]],[i,i],color=c,lw=2,solid_capstyle="round")
        a.plot(r["d"],i,"o",ms=6,color=c,mec="white",mew=1.2)
    a.set_title(title); a.set_xlabel("Difference in AUC"); a.grid(axis="y",visible=False)
ax[0].set_yticks(range(len(rows))); ax[0].set_yticklabels([f"{NAME.get(x,x)} vs. {NAME.get(y,y)}" for x,y in rows]); ax[0].invert_yaxis()
ax[0].set_ylim(len(rows)-0.5,-0.5)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([],[],color="#2a78d6",lw=2,marker="o",mec="white",label="95% interval excludes zero"),Line2D([],[],color=MUTED,lw=2,marker="o",mec="white",label="95% interval includes zero")],loc="lower center",ncol=2,bbox_to_anchor=(0.58,-0.06))
fig.tight_layout(); save(fig,"fig_forest")

# ---- 3. Calibration, pooled out-of-fold, ten equal-count bins ----
fig,a=plt.subplots(figsize=(4.4,3.5))
a.plot([0,1],[0,1],color=MUTED,lw=0.6)
for m,mk in [("TAFNet-Full","o"),("CNN-LSTM","s")]:
    p=cv[m]; o=np.argsort(p,kind="stable"); b=np.array_split(o,10)
    a.plot([p[i].mean() for i in b],[ycv[i].mean() for i in b],color=COL[m],ls=LS[m],lw=1.8,marker=mk,ms=5.5,mec="white",mew=1.1,label=NAME.get(m,m))
a.set_xlabel("Mean predicted probability"); a.set_ylabel("Observed conversion rate"); a.set_xlim(-0.02,1.02); a.set_ylim(-0.02,1.02); a.set_aspect("equal")
a.text(0.80,0.86,"perfect\ncalibration",color=INK2,fontsize=7,ha="left",va="top"); a.legend(loc="upper left",handlelength=2.6)
fig.tight_layout(); save(fig,"fig_calibration")

# ---- 4. Gate coefficients by fold model ----
G=json.load(open(os.path.join(RESULTS,"gate_analysis","gate_analysis.json")))
GC=["#2a78d6","#eb6834","#1baf7a"]; GL=[r"$\alpha$  difference",r"$\beta$  attention",r"$\gamma$  concatenation"]
fig,ax=plt.subplots(1,2,figsize=(6.9,2.7),sharey=True)
for a,title,key in [(ax[0],"Validation pairs","val_mean_gate"),(ax[1],"Held-out pairs","heldout_mean_gate")]:
    for f in G["per_fold"]:
        y=f["fold"]; left=0
        for j,v in enumerate(f[key]):
            a.barh(y,v,left=left,height=0.62,color=GC[j],edgecolor="white",linewidth=1.5,label=GL[j] if y==1 else None)
            if v>=0.12: a.text(left+v/2,y,f"{v:.2f}",ha="center",va="center",color="white" if j!=2 else INK,fontsize=7)
            left+=v
    a.set_title(title); a.set_xlabel("Mean gate weight"); a.set_xlim(0,1); a.grid(axis="y",visible=False)
ax[0].set_yticks([1,2,3,4,5]); ax[0].set_yticklabels([f"Fold {k}" for k in range(1,6)]); ax[0].invert_yaxis()
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=3,bbox_to_anchor=(0.54,-0.07))
fig.tight_layout(); save(fig,"fig_gate")

# ---- 5. Score distributions, pooled out-of-fold ----
fig,a=plt.subplots(figsize=(4.9,3.0)); bins=np.linspace(0,1,41)
for m in ["CNN-LSTM","TAFNet-Full"]:
    a.hist(cv[m],bins=bins,histtype="stepfilled",facecolor=COL[m]+"1a",edgecolor=COL[m],lw=1.6,ls=LS[m],label=NAME.get(m,m))
a.axvline(0.5,color=INK2,lw=0.7); a.text(0.51,a.get_ylim()[1]*0.62,"0.5 threshold",color=INK2,fontsize=7)
a.set_xlabel("Predicted probability of conversion"); a.set_ylabel("Pairs"); a.set_xlim(0,1); a.legend(loc="upper right",handlelength=2.6)
fig.tight_layout(); save(fig,"fig_scores")
print("scores: pct<0.01", {m:round(100*np.mean(cv[m]<0.01),1) for m in ["TAFNet-Full","CNN-LSTM"]})

# ---- 6. Convergent validity, pooled out-of-fold TAFNet risk ----
P=json.load(open(os.path.join(RESULTS,"gate_analysis","gate_per_pair.json")))["cv"]
idx=np.array([r["index"] for r in P]); pr=np.array([r["prob"] for r in P]); ms=np.array([r["mmse_slope_per_year"] for r in P],float); dc=np.array([r["days_to_conversion"] for r in P],float)
pos={i:n for n,i in enumerate(cvi)}; ref=np.array([cv["TAFNet-Full"][pos[i]] for i in idx])
assert np.allclose(pr,ref,atol=0.01),"gate-analysis probabilities differ from the fold predictions"
pr=ref  # plot the stored fold predictions (the gate rerun reproduces them to within 0.007)
fig,ax=plt.subplots(1,2,figsize=(6.9,3.1))
for a,y,title,yl,c in [(ax[0],ms,"Rate of cognitive change","MMSE change (points per year)","#2a78d6"),(ax[1],dc/30.4375,"Time to conversion, converters only","Months from baseline to conversion","#eb6834")]:
    ok=np.isfinite(y); x=pr[ok]; yy=y[ok]; rho=spearmanr(x,yy)[0]
    a.scatter(x,yy,s=16,color=c,edgecolor="white",linewidth=0.6,alpha=0.85)
    k=np.polyfit(x,yy,1); xs=np.array([0,1]); a.plot(xs,np.polyval(k,xs),color=INK,lw=1.0)
    a.set_title(title); a.set_xlabel("Predicted probability of conversion"); a.set_ylabel(yl); a.set_xlim(-0.03,1.03)
    a.text(0.97,0.95,rf"Spearman $\rho$ = {rho:+.3f}   n = {ok.sum()}",transform=a.transAxes,ha="right",va="top",color=INK,fontsize=7.5,bbox=dict(fc="white",ec="none",pad=1.5))
    print(title,"rho",round(rho,3),"n",ok.sum())
fig.tight_layout(); save(fig,"fig_validity")
print("done")

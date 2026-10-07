#!/usr/bin/env python3
"""
Calibration (paper Section 5.5, Table 6).

Uncalibrated: Brier score, calibration slope/intercept, ECE, on the held-out
five-fold ensemble and on the pooled out-of-fold CV predictions.

Recalibration is fitted ON VALIDATION ONLY: for each fold, Platt (logistic) and
isotonic maps are fitted on that fold's validation predictions and applied to
that fold's held-out predictions; the five recalibrated probability vectors are
then averaged. The held-out partition is never used to fit a calibration map.
"""
import json
import numpy as np
import pandas as pd
from common import PAIRS_CSV, SIDECAR_CSV, RESULTS, fold_json, out_path, load_oof_and_heldout
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

B = 10000
SEED = 20260907
EPS = 1e-6
METHODS = ["ResNet3D-18", "DenseNet3D-121", "Siamese-Subtract",
           "CNN-LSTM", "TAFNet-InitialOnly", "TAFNet-Full"]

df = pd.read_csv(PAIRS_CSV)
subj = df["subject"].astype(str).to_numpy(); lab = df["label"].astype(int).to_numpy()
pool = sorted(set(subj)); strata = np.array([lab[subj == s].max() for s in pool])
keep, test = train_test_split(pool, test_size=0.15, random_state=42, stratify=strata)
keep = sorted(keep); tset = set(test)
test_idx = [i for i, s in enumerate(subj) if s in tset]
test_subj = subj[test_idx]

kstrata = np.array([lab[subj == s].max() for s in keep])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sizes, cv_subj_order = [], []
for tr, va in skf.split(keep, kstrata):
    vaset = {keep[i] for i in va}
    rows = [i for i, s in enumerate(subj) if s in vaset]
    sizes.append(len(rows)); cv_subj_order += [subj[i] for i in rows]
bounds = np.cumsum([0] + sizes)
cv_subj = np.array(cv_subj_order)

oof, bundle = load_oof_and_heldout(METHODS)
y_test = np.asarray(bundle["test_y_true"], float)
y_oof = np.asarray(oof["TAFNet-Full"]["y_true"], float)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def slope_intercept(y, p):
    """Cox calibration: logistic regression of y on logit(p). Ideal (1, 0)."""
    x = logit(p).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(x, y)
    slope = float(lr.coef_[0][0])
    # intercept at fixed slope 1 = calibration-in-the-large
    lr0 = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000,
                             fit_intercept=True)
    lr0.fit(np.zeros((len(y), 1)), y)   # placeholder, replaced below
    off = logit(p)
    # fit intercept only, with logit(p) as offset -> simple 1-D search
    from scipy.optimize import minimize_scalar
    def nll(a):
        z = a + off
        return float(np.mean(np.logaddexp(0, z) - y * z))
    a = minimize_scalar(nll, bounds=(-10, 10), method="bounded").x
    return slope, float(a)


def ece(y, p, nbins=5):
    edges = np.quantile(p, np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    tot = 0.0
    for i in range(nbins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        tot += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(tot)


def summarise(y, p):
    s, a = slope_intercept(y, p)
    return dict(Brier=brier(y, p), slope=s, intercept=a, ECE=ece(y, p),
                mean_p=float(p.mean()), obs=float(y.mean()))


# ---------------------------------------------------------- recalibration
def recalibrated(m, kind):
    """Fit on each fold's validation, apply to that fold's held-out preds, average."""
    p_oof = np.asarray(oof[m]["y_pred"], float)
    ft = np.asarray(bundle["test_y_pred_per_fold"][m])
    out = []
    for k in range(5):
        pv = p_oof[bounds[k]:bounds[k+1]]
        yv = y_oof[bounds[k]:bounds[k+1]]
        if kind == "platt":
            f = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
            f.fit(logit(pv).reshape(-1, 1), yv)
            out.append(f.predict_proba(logit(ft[k]).reshape(-1, 1))[:, 1])
        else:
            f = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            f.fit(pv, yv)
            out.append(f.predict(ft[k]))
    return np.mean(out, axis=0)


# bootstrap machinery (subject-clustered, held-out)
uniq = np.array(sorted(set(test_subj.tolist())))
srows = {s: np.where(test_subj == s)[0] for s in uniq}
rng = np.random.default_rng(SEED)
boot = []
while len(boot) < B:
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    r = np.concatenate([srows[s] for s in pick])
    if y_test[r].min() != y_test[r].max():
        boot.append(r)

res = {}
print("Held-out ensemble, uncalibrated (85 pairs, 20 positive; prevalence 0.235)")
print(f"{'Method':22s} {'Brier':>7s} {'(95% CI)':>16s} {'slope':>7s} {'intcpt':>7s} "
      f"{'ECE':>6s} {'mean p':>7s}")
for m in METHODS:
    ens = np.asarray(bundle["test_y_pred_per_fold"][m]).mean(axis=0)
    d = summarise(y_test, ens)
    bs = np.array([brier(y_test[r], ens[r]) for r in boot])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    d["Brier_CI"] = [float(lo), float(hi)]
    res[m] = {"heldout_raw": d}
    print(f"{m:22s} {d['Brier']:7.4f} [{lo:.4f}, {hi:.4f}] {d['slope']:7.3f} "
          f"{d['intercept']:7.3f} {d['ECE']:6.3f} {d['mean_p']:7.3f}")

print("\nPooled out-of-fold CV (519 pairs, 113 positive; prevalence 0.218)")
print(f"{'Method':22s} {'Brier':>7s} {'slope':>7s} {'intcpt':>7s} {'ECE':>6s} {'mean p':>7s}")
for m in METHODS:
    p = np.asarray(oof[m]["y_pred"], float)
    d = summarise(y_oof, p)
    res[m]["cv_raw"] = d
    print(f"{m:22s} {d['Brier']:7.4f} {d['slope']:7.3f} {d['intercept']:7.3f} "
          f"{d['ECE']:6.3f} {d['mean_p']:7.3f}")

print("\nHeld-out after recalibration fitted on validation folds only")
print(f"{'Method':22s} {'raw':>7s} {'Platt':>7s} {'Isotonic':>9s}   (Brier; lower is better)")
for m in METHODS:
    ens = np.asarray(bundle["test_y_pred_per_fold"][m]).mean(axis=0)
    pl = recalibrated(m, "platt"); iso = recalibrated(m, "isotonic")
    res[m]["heldout_platt"] = summarise(y_test, pl)
    res[m]["heldout_isotonic"] = summarise(y_test, iso)
    np.save(out_path(f"cal_{m}.npy"),
            np.vstack([ens, pl, iso]))
    print(f"{m:22s} {brier(y_test, ens):7.4f} {brier(y_test, pl):7.4f} "
          f"{brier(y_test, iso):9.4f}")

print("\nSlope/intercept after Platt (ideal 1.000 / 0.000)")
for m in METHODS:
    d = res[m]["heldout_platt"]
    print(f"{m:22s} slope {d['slope']:6.3f}   intercept {d['intercept']:6.3f}   "
          f"ECE {d['ECE']:.3f}")

json.dump(dict(n_test=85, n_test_pos=20, bootstrap_B=B, seed=SEED, per_method=res),
          open(out_path("calibration.json"), "w"), indent=2)
np.save(out_path("y_test.npy"), y_test)
np.save(out_path("cv_meta.npy"), np.array(bounds))
print("\nwrote", out_path("calibration.json"))

#!/usr/bin/env python3
"""
Validation-tuned operating points (paper Section 5.5, Table 5).

Thresholds are selected ONLY on validation data:
  * per fold: on that fold's validation predictions, applied to that fold's
    held-out predictions;
  * ensemble: the mean of the five validation-derived thresholds, applied to the
    five-fold mean held-out probability.
The held-out partition is never used to choose a threshold.

Two criteria: Youden's J (primary) and specificity fixed at >= 0.80.
"""
import json
import numpy as np
import pandas as pd
from common import PAIRS_CSV, SIDECAR_CSV, RESULTS, fold_json, out_path, load_oof_and_heldout
from sklearn.model_selection import train_test_split, StratifiedKFold

B = 10000
SEED = 20260907
METHODS = ["ResNet3D-18", "DenseNet3D-121", "Siamese-Subtract",
           "CNN-LSTM", "TAFNet-InitialOnly", "TAFNet-Full"]

# ------------------------------------------------------------ splits
df = pd.read_csv(PAIRS_CSV)
subj = df["subject"].astype(str).to_numpy()
lab = df["label"].astype(int).to_numpy()

pool = sorted(set(subj))
strata = np.array([lab[subj == s].max() for s in pool])
keep, test = train_test_split(pool, test_size=0.15, random_state=42,
                              stratify=strata)
keep = sorted(keep)
test_set = set(test)
test_idx = [i for i, s in enumerate(subj) if s in test_set]
test_subj = subj[test_idx]

kstrata = np.array([lab[subj == s].max() for s in keep])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_rows, fold_sizes = [], []
for tr, va in skf.split(keep, kstrata):
    vaset = {keep[i] for i in va}
    rows = [i for i, s in enumerate(subj) if s in vaset]
    fold_rows.append(rows)
    fold_sizes.append(len(rows))
bounds = np.cumsum([0] + fold_sizes)
print(f"validation fold sizes: {fold_sizes}  (total {bounds[-1]})")

# ------------------------------------------------------------ predictions
oof, bundle = load_oof_and_heldout(METHODS)
y_test = np.asarray(bundle["test_y_true"], float)
y_oof = np.asarray(oof["TAFNet-Full"]["y_true"], float)
assert len(y_oof) == bounds[-1] == 519

val_true = [y_oof[bounds[k]:bounds[k+1]] for k in range(5)]


def metrics(y, yhat):
    tp = float(((yhat == 1) & (y == 1)).sum()); tn = float(((yhat == 0) & (y == 0)).sum())
    fp = float(((yhat == 1) & (y == 0)).sum()); fn = float(((yhat == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    bacc = (sens + spec) / 2
    return dict(Sens=sens, Spec=spec, PPV=prec, NPV=npv, F1=f1, BAcc=bacc,
                Acc=(tp + tn) / len(y))


def pick_threshold(y, p, criterion):
    """Choose a cut-point on VALIDATION data only."""
    cand = np.unique(np.concatenate([[0.0], np.sort(p), [1.0]]))
    best, best_t = -np.inf, 0.5
    for t in cand:
        m = metrics(y, (p >= t).astype(int))
        if criterion == "youden":
            score = m["Sens"] + m["Spec"] - 1
        elif criterion == "spec80":
            # highest sensitivity among cut-points with specificity >= 0.80
            score = m["Sens"] if m["Spec"] >= 0.80 else -np.inf
        if score > best:
            best, best_t = score, float(t)
    return best_t


rows, thr_table = [], {}
for crit in ["youden", "spec80"]:
    for m in METHODS:
        p_oof = np.asarray(oof[m]["y_pred"], float)
        folds_test = np.asarray(bundle["test_y_pred_per_fold"][m])   # (5, 85)

        # --- per-fold: threshold from that fold's validation, applied to its own test preds
        ts, permetric = [], []
        for k in range(5):
            pv = p_oof[bounds[k]:bounds[k+1]]
            t = pick_threshold(val_true[k], pv, crit)
            ts.append(t)
            permetric.append(metrics(y_test, (folds_test[k] >= t).astype(int)))
        ts = np.array(ts)

        # --- ensemble: mean validation threshold applied to mean probability
        ens = folds_test.mean(axis=0)
        t_ens = float(ts.mean())
        me = metrics(y_test, (ens >= t_ens).astype(int))

        thr_table[(crit, m)] = dict(thresholds=ts.tolist(), t_ens=t_ens,
                                    ensemble=me,
                                    perfold_mean={k: float(np.mean([d[k] for d in permetric]))
                                                  for k in permetric[0]},
                                    perfold_sd={k: float(np.std([d[k] for d in permetric], ddof=1))
                                                for k in permetric[0]})
        rows.append((crit, m, t_ens, ts.std(ddof=1), me))

# ------------------------------------------------- clustered bootstrap on the operating point
uniq = np.array(sorted(set(test_subj.tolist())))
srows = {s: np.where(test_subj == s)[0] for s in uniq}
rng = np.random.default_rng(SEED)
boot = []
while len(boot) < B:
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    r = np.concatenate([srows[s] for s in pick])
    if y_test[r].min() != y_test[r].max():
        boot.append(r)

for crit in ["youden", "spec80"]:
    print(f"\n{'='*96}\nHeld-out at validation-derived threshold — criterion: "
          f"{'Youden J' if crit=='youden' else 'specificity >= 0.80'}\n{'='*96}")
    print(f"{'Method':22s} {'thr':>6s} {'Sens':>6s} {'(95% CI)':>16s} "
          f"{'Spec':>6s} {'(95% CI)':>16s} {'F1':>6s} {'PPV':>6s} {'NPV':>6s} {'BAcc':>6s}")
    for m in METHODS:
        d = thr_table[(crit, m)]
        t = d["t_ens"]; me = d["ensemble"]
        ens = np.asarray(bundle["test_y_pred_per_fold"][m]).mean(axis=0)
        bs = np.array([[metrics(y_test[r], (ens[r] >= t).astype(int))[k]
                        for k in ("Sens", "Spec")] for r in boot])
        sl, sh = np.nanpercentile(bs[:, 0], [2.5, 97.5])
        pl, ph = np.nanpercentile(bs[:, 1], [2.5, 97.5])
        d["Sens_CI"] = [float(sl), float(sh)]
        d["Spec_CI"] = [float(pl), float(ph)]
        print(f"{m:22s} {t:6.3f} {me['Sens']:6.3f} [{sl:.3f}, {sh:.3f}] "
              f"{me['Spec']:6.3f} [{pl:.3f}, {ph:.3f}] {me['F1']:6.3f} "
              f"{me['PPV']:6.3f} {me['NPV']:6.3f} {me['BAcc']:6.3f}")
    print(f"\n{'Method':22s} per-fold threshold mean±SD    per-fold held-out Sens / Spec (mean±SD)")
    for m in METHODS:
        d = thr_table[(crit, m)]
        ts = np.array(d["thresholds"])
        print(f"{m:22s} {ts.mean():.3f} ± {ts.std(ddof=1):.3f}            "
              f"{d['perfold_mean']['Sens']:.3f} ± {d['perfold_sd']['Sens']:.3f} / "
              f"{d['perfold_mean']['Spec']:.3f} ± {d['perfold_sd']['Spec']:.3f}")

out = {f"{c}|{m}": v for (c, m), v in thr_table.items()}
json.dump(dict(n_test_pairs=85, n_test_subjects=int(len(uniq)),
               bootstrap_B=B, seed=SEED, results=out),
          open(out_path("operating_points.json"), "w"), indent=2)
print("\nwrote", out_path("operating_points.json"))

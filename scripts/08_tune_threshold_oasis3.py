#!/usr/bin/env python3
"""
tune_threshold_oasis3.py
========================
Choose a decision cut-off for the TAFNet v4 ensemble score on OASIS-3, and
estimate how well that cut-off would do on new patients.

Needs only numpy + pandas. Resampling is by SUBJECT (101 subjects, 125 pairs),
so pairs from the same person always move together.

Reports
  1. Youden cut-off chosen on all 125 pairs, with a bootstrap 95% CI for the
     cut-off itself (how stable it is).
  2. Honest sensitivity/specificity: repeated 5-fold subject-grouped CV. The
     cut-off is chosen on 4/5 of subjects and applied to the unseen 1/5.
  3. Sens/spec (bootstrap 95% CI) at fixed cut-offs: 0.5, ADNI 0.191, the
     OASIS-3 Youden cut-off, and the highest cut-off with sensitivity >= 0.80.

Usage:  python3 scripts/08_tune_threshold_oasis3.py --scores results_oasis3_v4/OASIS3_125_tafnet_v4_scores.csv
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd


def auc(y, s):
    """Mann-Whitney AUC with tie handling."""
    y = np.asarray(y); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                       # average ranks for ties
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    npos, nneg = (y == 1).sum(), (y == 0).sum()
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def rates(y, s, t):
    p = s >= t
    tp = np.sum(p & (y == 1)); fn = np.sum(~p & (y == 1))
    tn = np.sum(~p & (y == 0)); fp = np.sum(p & (y == 0))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return dict(sens=sens, spec=spec, ppv=ppv, npv=npv, J=sens + spec - 1,
                tp=int(tp), fn=int(fn), tn=int(tn), fp=int(fp))


def candidates(s):
    u = np.unique(s)
    mids = (u[:-1] + u[1:]) / 2                  # cut between neighbouring scores
    return np.concatenate([[u[0] - 1e-9], mids, [u[-1] + 1e-9]])


def youden(y, s):
    best_t, best_j = None, -np.inf
    for t in candidates(s):
        r = rates(y, s, t)
        if r["J"] > best_j + 1e-12:              # ties -> keep the lower cut-off
            best_t, best_j = t, r["J"]
    return float(best_t)


def cut_for_sens(y, s, target=0.80):
    ok = [t for t in candidates(s) if rates(y, s, t)["sens"] >= target]
    return float(max(ok))


def resample_subjects(df, rng):
    subs = df["OASISID"].unique()
    pick = rng.choice(subs, size=len(subs), replace=True)
    return pd.concat([df[df.OASISID == x] for x in pick], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="output CSV of 07_score_oasis3_v4.py")
    ap.add_argument("--score-col", default="tafnet_v4_ensemble")
    ap.add_argument("--label-col", default="pair_label_converter")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--cv-repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    df = pd.read_csv(a.scores)[["OASISID", a.label_col, a.score_col]].rename(
        columns={a.label_col: "y", a.score_col: "s"})
    y, s = df.y.to_numpy(int), df.s.to_numpy(float)
    groups = {k: g for k, g in df.groupby("OASISID")}
    subs = np.array(list(groups))

    print(f"Pairs {len(df)}  subjects {len(subs)}  converter pairs {y.sum()}  "
          f"AUC {auc(y, s):.3f}")
    print(f"Score median: converters {np.median(s[y==1]):.3f}, stable {np.median(s[y==0]):.3f}\n")

    # ---- 1. Youden cut-off on all data + stability ----------------------------
    t_youden = youden(y, s)
    boot_t = []
    for _ in range(a.n_boot):
        b = pd.concat([groups[x] for x in rng.choice(subs, len(subs))], ignore_index=True)
        if b.y.nunique() == 2:
            boot_t.append(youden(b.y.to_numpy(int), b.s.to_numpy(float)))
    boot_t = np.array(boot_t)
    t_lo, t_hi = np.percentile(boot_t, [2.5, 97.5])
    print(f"1) OASIS-3 Youden cut-off: {t_youden:.3f}   "
          f"bootstrap 95% CI [{t_lo:.3f}, {t_hi:.3f}]")

    # ---- 2. honest performance: repeated subject-grouped 5-fold CV ------------
    has_pos = np.array([groups[x].y.max() for x in subs])
    cv = []
    for _ in range(a.cv_repeats):
        fold_of = {}
        for flag in (0, 1):                       # stratify subjects by converter status
            members = rng.permutation(subs[has_pos == flag])
            for i, x in enumerate(members):
                fold_of[x] = i % 5
        f = df.OASISID.map(fold_of).to_numpy()
        pred = np.zeros(len(df), bool)
        for k in range(5):
            tr, te = f != k, f == k
            t = youden(y[tr], s[tr])
            pred[te] = s[te] >= t
        tp = np.sum(pred & (y == 1)); tn = np.sum(~pred & (y == 0))
        cv.append((tp / y.sum(), tn / (y == 0).sum()))
    cv = np.array(cv)
    cv_sens, cv_spec = cv.mean(0)
    print(f"2) Honest (cut-off chosen on other subjects, 5-fold x {a.cv_repeats}): "
          f"sens {cv_sens:.3f}, spec {cv_spec:.3f}, BAcc {(cv_sens+cv_spec)/2:.3f}")

    # ---- 3. fixed cut-offs with bootstrap CIs ---------------------------------
    t_s80 = cut_for_sens(y, s, 0.80)
    fixed = [("default 0.5", 0.5), ("ADNI val-derived 0.191", 0.191),
             ("OASIS-3 Youden", t_youden), ("OASIS-3 sens>=0.80", t_s80)]
    boots = [pd.concat([groups[x] for x in rng.choice(subs, len(subs))], ignore_index=True)
             for _ in range(a.n_boot)]
    print("\n3) Performance at fixed cut-offs (apparent, subject-bootstrap 95% CI)")
    print(f"   {'cut-off':<24}{'value':>7}{'sens':>20}{'spec':>20}{'PPV':>7}{'NPV':>7}")
    table = []
    for name, t in fixed:
        r = rates(y, s, t)
        bs = np.array([[rates(b.y.to_numpy(int), b.s.to_numpy(float), t)[k] for k in ("sens", "spec")]
                       for b in boots])
        lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
        print(f"   {name:<24}{t:>7.3f}{r['sens']:>7.3f} [{lo[0]:.2f},{hi[0]:.2f}]"
              f"{r['spec']:>7.3f} [{lo[1]:.2f},{hi[1]:.2f}]{r['ppv']:>7.3f}{r['npv']:>7.3f}")
        table.append(dict(name=name, threshold=t, **{k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                                      for k, v in r.items()},
                          sens_ci=[float(lo[0]), float(hi[0])], spec_ci=[float(lo[1]), float(hi[1])]))
    print("\n   Rows 3-4 are chosen and scored on the same 125 pairs, so they are optimistic;\n"
          "   quote the cut-off from (1) and the performance from (2).")

    out = dict(n_pairs=len(df), n_subjects=len(subs), n_pos_pairs=int(y.sum()), auc=float(auc(y, s)),
               youden_threshold=t_youden, youden_threshold_ci=[float(t_lo), float(t_hi)],
               cv_honest=dict(sens=float(cv_sens), spec=float(cv_spec), repeats=a.cv_repeats,
                              sens_range=[float(x) for x in np.percentile(cv[:, 0], [2.5, 97.5])],
                              spec_range=[float(x) for x in np.percentile(cv[:, 1], [2.5, 97.5])]),
               fixed_cutoffs=table, score_col=a.score_col, seed=a.seed, n_boot=a.n_boot)
    path = os.path.join(os.path.dirname(os.path.abspath(a.scores)), "oasis3_threshold_tuning.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\n[OK] Wrote {path}")


if __name__ == "__main__":
    main()

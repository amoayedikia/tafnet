#!/usr/bin/env python3
"""
Step 4 — Adaptive Temporal Gate analysis.

Extracts the per-patient gate coefficients (alpha, beta, gamma) =
(difference, attention, concat) from the cached per-fold TAFNet-Full
checkpoints and relates them to clinical progression:

    days_to_conversion      (converter pairs only)
    mmse_slope_per_year     (all pairs with a slope)

Two partitions, reported separately and never pooled:

    cv       out-of-fold: each CV pair is scored by the one fold whose
             validation split contains it.
    heldout  the 85 held-out pairs, scored by all five folds; the reported
             gate is the across-fold mean, matching how the held-out
             predictions were ensembled for the paper.

All inference is subject-clustered: confidence intervals and p-values come
from a cluster bootstrap that resamples SUBJECTS with replacement and takes
every pair belonging to a drawn subject. Pairs from one subject are not
independent, and 604 pairs come from 325 subjects.

Note on compositionality: alpha + beta + gamma = 1 by construction, so the
three marginal correlations are not independent. The log-ratios are reported
alongside them as the compositional-aware view, computed only where every
coefficient exceeds a floor.

Usage
-----
    cd /data/code/tafnet-1July26 && PYTHONPATH=src python3 scripts/06_gate_analysis.py \
        --config configs/default.yaml \
        --results-dir /data/results \
        --out /data/results/gate_analysis

Reads nothing but the config, the pairs CSV, the preprocessed volumes and
`<results-dir>/folds/<method>_fold{1..5}.pth`. Writes nothing outside `--out`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tafnet.config import load_config              # noqa: E402
from tafnet.data import LabelledPairDataset        # noqa: E402
from tafnet.models import TAFNet                   # noqa: E402
from tafnet.models.tafnet import BRANCH_ORDER      # noqa: E402
from tafnet.utils import get_device, set_seed      # noqa: E402

COEFS = ("alpha", "beta", "gamma")
COEF_BRANCH = dict(zip(COEFS, BRANCH_ORDER))
TARGETS = ("days_to_conversion", "mmse_slope_per_year")
N_BOOT = 2000
LOGRATIO_FLOOR = 1e-4


# --------------------------------------------------------------------------
# gate extraction
# --------------------------------------------------------------------------

def build_tafnet_full(config, device: str) -> TAFNet:
    """Same construction as benchmarks._build_model('TAFNet-Full'), no encoder
    checkpoint: the fold state_dict already carries the encoder weights."""
    arch = config.architecture
    return TAFNet(
        encoder_channels=arch.encoder_channels,
        use_dcca=arch.use_dcca,
        feature_dim=arch.feature_dim,
        num_heads=arch.num_heads,
        dropout=arch.dropout,
        use_longitudinal=True,
        freeze_encoder=True,
        gate_mode=getattr(arch, "gate_mode", "patient"),
        baseline_residual=getattr(arch, "baseline_residual", True),
        branches=getattr(arch, "branches", None),
    ).to(device)


@torch.no_grad()
def gates_for_indices(model, full_ds, indices: Sequence[int], device: str,
                      batch_size: int, num_workers: int) -> Dict[str, np.ndarray]:
    """Return {'gate': (n, 3), 'prob': (n,)} in the order of `indices`."""
    if len(indices) == 0:
        return {"gate": np.zeros((0, 3)), "prob": np.zeros(0)}
    sub = full_ds.subset(list(indices), is_training=False)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    model.eval()
    gates, probs = [], []
    for t1, t2, _labels in loader:
        # float32 throughout: autocast is deliberately off here. The gate is a
        # softmax over three logits and the published concern is a variance of
        # order 1e-2, which fp16 rounding would sit inside.
        logits, aux = model(t1.to(device), t2.to(device), return_aux=True)
        gates.append(aux["gate"].float().cpu().numpy())
        probs.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
    return {"gate": np.concatenate(gates, axis=0),
            "prob": np.concatenate(probs, axis=0)}


# --------------------------------------------------------------------------
# subject-clustered inference
# --------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(stats.pearsonr(x, y).statistic)


def cluster_bootstrap(stat_fn, x: np.ndarray, y: np.ndarray,
                      subjects: np.ndarray, n_boot: int = N_BOOT,
                      seed: int = 42) -> Dict[str, float]:
    """
    Cluster bootstrap over subjects.

    Returns the point estimate, a percentile 95% CI, and a two-sided p-value
    for H0: statistic = 0, taken as twice the smaller bootstrap tail mass on
    either side of zero (floored at 1/n_boot rather than reported as 0).
    """
    point = stat_fn(x, y)
    uniq = np.unique(subjects)
    by_subj = {s: np.flatnonzero(subjects == s) for s in uniq}
    rng = np.random.default_rng(seed)
    boot: List[float] = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_subj[s] for s in drawn])
        v = stat_fn(x[idx], y[idx])
        if np.isfinite(v):
            boot.append(v)
    if len(boot) < 100 or not np.isfinite(point):
        return {"estimate": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "p": float("nan"), "n_boot_ok": len(boot)}
    b = np.asarray(boot)
    frac_le = float(np.mean(b <= 0.0))
    frac_ge = float(np.mean(b >= 0.0))
    p = 2.0 * min(frac_le, frac_ge)
    p = max(min(p, 1.0), 1.0 / len(b))
    return {"estimate": point,
            "ci_low": float(np.percentile(b, 2.5)),
            "ci_high": float(np.percentile(b, 97.5)),
            "p": p, "n_boot_ok": len(b)}


def cluster_bootstrap_meandiff(values: np.ndarray, group: np.ndarray,
                               subjects: np.ndarray, n_boot: int = N_BOOT,
                               seed: int = 42) -> Dict[str, float]:
    """Mean(values | group==1) - mean(values | group==0), subject-clustered."""
    def _diff(v, g):
        if g.sum() == 0 or (1 - g).sum() == 0:
            return float("nan")
        return float(v[g == 1].mean() - v[g == 0].mean())

    point = _diff(values, group)
    uniq = np.unique(subjects)
    by_subj = {s: np.flatnonzero(subjects == s) for s in uniq}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_subj[s] for s in drawn])
        v = _diff(values[idx], group[idx])
        if np.isfinite(v):
            boot.append(v)
    if len(boot) < 100 or not np.isfinite(point):
        return {"estimate": point, "ci_low": float("nan"),
                "ci_high": float("nan"), "p": float("nan"), "n_boot_ok": len(boot)}
    b = np.asarray(boot)
    p = 2.0 * min(float(np.mean(b <= 0)), float(np.mean(b >= 0)))
    return {"estimate": point,
            "ci_low": float(np.percentile(b, 2.5)),
            "ci_high": float(np.percentile(b, 97.5)),
            "p": max(min(p, 1.0), 1.0 / len(b)), "n_boot_ok": len(b)}


# --------------------------------------------------------------------------
# analysis over one partition
# --------------------------------------------------------------------------

def analyse_partition(records: List[dict], name: str,
                      n_boot: int = N_BOOT) -> dict:
    """All correlations and descriptives for one partition."""
    gate = np.array([[r["alpha"], r["beta"], r["gamma"]] for r in records])
    subjects = np.array([r["subject"] for r in records])
    labels = np.array([r["label"] for r in records], dtype=int)
    prob = np.array([r["prob"] for r in records], dtype=float)

    out: dict = {
        "partition": name,
        "n_pairs": len(records),
        "n_subjects": int(len(np.unique(subjects))),
        "n_converter_pairs": int(labels.sum()),
        "descriptives": {},
        "correlations": [],
        "converter_contrast": {},
        "gate_vs_prediction": {},
        "logratio_correlations": [],
    }

    # Descriptives — the degeneracy check. The published analysis reported
    # sigma_alpha = 0.013 with predictions clustered at P ~= 0.52; if the gate
    # is again near-constant, no correlation can be meaningful whatever its
    # p-value, and that is the finding.
    for j, c in enumerate(COEFS):
        v = gate[:, j]
        out["descriptives"][c] = {
            "branch": COEF_BRANCH[c],
            "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max()),
            "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
            "range": float(v.max() - v.min()),
        }
    out["descriptives"]["prediction"] = {
        "mean": float(prob.mean()), "sd": float(prob.std(ddof=1)),
        "min": float(prob.min()), "max": float(prob.max()),
    }

    # Correlations against the two clinical targets
    for target in TARGETS:
        vals = np.array([r.get(target, np.nan) for r in records], dtype=float)
        mask = np.isfinite(vals)
        if target == "days_to_conversion":
            # Only defined for converters.
            mask &= labels == 1
        if mask.sum() < 8:
            out["correlations"].append(
                {"target": target, "n": int(mask.sum()), "skipped": "n < 8"})
            continue
        for j, c in enumerate(COEFS):
            x, y, s = gate[mask, j], vals[mask], subjects[mask]
            row = {"target": target, "coef": c, "branch": COEF_BRANCH[c],
                   "n": int(mask.sum()),
                   "n_subjects": int(len(np.unique(s))),
                   "spearman": cluster_bootstrap(_spearman, x, y, s, n_boot),
                   "pearson": cluster_bootstrap(_pearson, x, y, s, n_boot)}
            out["correlations"].append(row)

        # Compositional view: log-ratios, computed only where the gate is not
        # numerically degenerate.
        g = gate[mask]
        ok = np.all(g > LOGRATIO_FLOOR, axis=1)
        if ok.sum() >= 8:
            lr = {
                "log_alpha_over_gamma": np.log(g[ok, 0] / g[ok, 2]),
                "log_alpha_over_beta": np.log(g[ok, 0] / g[ok, 1]),
                "log_beta_over_gamma": np.log(g[ok, 1] / g[ok, 2]),
            }
            yy, ss = vals[mask][ok], subjects[mask][ok]
            for key, xx in lr.items():
                out["logratio_correlations"].append(
                    {"target": target, "ratio": key, "n": int(ok.sum()),
                     "spearman": cluster_bootstrap(_spearman, xx, yy, ss, n_boot)})

    # Converter vs non-converter contrast in each coefficient
    if 0 < labels.sum() < len(labels):
        for j, c in enumerate(COEFS):
            out["converter_contrast"][c] = cluster_bootstrap_meandiff(
                gate[:, j], labels, subjects, n_boot)

    # Does the gate track the model's own output?
    for j, c in enumerate(COEFS):
        out["gate_vs_prediction"][c] = cluster_bootstrap(
            _spearman, gate[:, j], prob, subjects, n_boot)

    return out


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default=None,
                    help="Where folds/ lives. Defaults to config paths.output_dir.")
    ap.add_argument("--method", default="TAFNet-Full")
    ap.add_argument("--out", default=None,
                    help="Output directory. Defaults to <results-dir>/gate_analysis.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    cfg = load_config(args.config)
    results_dir = args.results_dir or cfg.paths.output_dir
    out_dir = args.out or os.path.join(results_dir, "gate_analysis")
    os.makedirs(out_dir, exist_ok=True)

    set_seed(cfg.training.seed)
    device = get_device()
    print(f"  device={device}  results={results_dir}  out={out_dir}")

    full_ds = LabelledPairDataset(
        pairs_csv=cfg.paths.pairs_csv, data_dir=cfg.paths.data_dir,
        is_training=False, verify_files=True,
    )
    holdout_frac = float(getattr(cfg.paths, "holdout_frac", 0.0) or 0.0)
    test_idx, cv_subjects = full_ds.get_holdout_split(
        test_frac=holdout_frac, random_state=cfg.training.seed)
    folds = full_ds.get_subject_level_split_indices(
        n_splits=cfg.training.num_folds, random_state=cfg.training.seed,
        subjects_subset=cv_subjects)
    full_ds.assert_no_subject_leakage(test_idx, folds[0][0], folds[0][1])
    print(f"  held-out {len(test_idx)} pairs · CV {sum(len(v) for _, v in folds)} "
          f"pairs over {len(folds)} folds")

    def meta(i: int) -> dict:
        s = full_ds.samples[i]
        return {"index": int(i), "subject": str(s["subject"]),
                "label": int(s["label"]),
                "days_to_conversion": float(s.get("days_to_conversion", np.nan)
                                            if s.get("days_to_conversion") is not None
                                            else np.nan),
                "mmse_slope_per_year": float(s.get("mmse_slope_per_year", np.nan)
                                             if s.get("mmse_slope_per_year") is not None
                                             else np.nan)}

    cv_records: List[dict] = []
    heldout_gate_by_fold: List[np.ndarray] = []
    heldout_prob_by_fold: List[np.ndarray] = []
    per_fold_summary = []

    for k, (_train_idx, val_idx) in enumerate(folds):
        ckpt = os.path.join(results_dir, "folds", f"{args.method}_fold{k+1}.pth")
        if not os.path.exists(ckpt):
            print(f"  [X] missing checkpoint {ckpt}")
            return 1
        model = build_tafnet_full(cfg, device)
        state = torch.load(ckpt, map_location=device)
        # strict=True on purpose: a silent partial load is exactly the failure
        # mode that produced the original branch-ordering defect.
        model.load_state_dict(state, strict=True)
        print(f"  fold {k+1}: loaded {os.path.basename(ckpt)}")

        v = gates_for_indices(model, full_ds, val_idx, device,
                              args.batch_size, args.num_workers)
        for pos, i in enumerate(val_idx):
            rec = meta(i)
            rec.update(fold=k + 1, alpha=float(v["gate"][pos, 0]),
                       beta=float(v["gate"][pos, 1]),
                       gamma=float(v["gate"][pos, 2]),
                       prob=float(v["prob"][pos]))
            cv_records.append(rec)

        t = gates_for_indices(model, full_ds, test_idx, device,
                              args.batch_size, args.num_workers)
        heldout_gate_by_fold.append(t["gate"])
        heldout_prob_by_fold.append(t["prob"])
        per_fold_summary.append({
            "fold": k + 1,
            "val_mean_gate": v["gate"].mean(axis=0).round(4).tolist(),
            "val_sd_gate": v["gate"].std(axis=0, ddof=1).round(4).tolist(),
            "heldout_mean_gate": t["gate"].mean(axis=0).round(4).tolist(),
            "heldout_sd_gate": t["gate"].std(axis=0, ddof=1).round(4).tolist(),
        })
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    heldout_gate = np.mean(np.stack(heldout_gate_by_fold), axis=0)
    heldout_prob = np.mean(np.stack(heldout_prob_by_fold), axis=0)
    heldout_records = []
    for pos, i in enumerate(test_idx):
        rec = meta(i)
        rec.update(fold=0, alpha=float(heldout_gate[pos, 0]),
                   beta=float(heldout_gate[pos, 1]),
                   gamma=float(heldout_gate[pos, 2]),
                   prob=float(heldout_prob[pos]))
        heldout_records.append(rec)

    result = {
        "method": args.method,
        "results_dir": results_dir,
        "branch_order": list(BRANCH_ORDER),
        "gate_mode": getattr(cfg.architecture, "gate_mode", "patient"),
        "n_boot": args.n_boot,
        "per_fold": per_fold_summary,
        "cv": analyse_partition(cv_records, "cv", args.n_boot),
        "heldout": analyse_partition(heldout_records, "heldout", args.n_boot),
        "heldout_fold_agreement": {
            "sd_across_folds_mean": np.stack(heldout_gate_by_fold).std(axis=0).mean(axis=0).round(4).tolist(),
        },
    }

    with open(os.path.join(out_dir, "gate_analysis.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    with open(os.path.join(out_dir, "gate_per_pair.json"), "w") as fh:
        json.dump({"cv": cv_records, "heldout": heldout_records}, fh, indent=2)

    # ---- human-readable summary ------------------------------------------
    lines = [f"# Gate analysis — {args.method}", "",
             f"Branch order: {BRANCH_ORDER} → (alpha, beta, gamma)",
             f"Gate mode: {result['gate_mode']}   bootstrap: {args.n_boot} "
             f"subject-clustered resamples", ""]
    for part in ("cv", "heldout"):
        r = result[part]
        lines += [f"## {part}  ({r['n_pairs']} pairs / {r['n_subjects']} subjects "
                  f"/ {r['n_converter_pairs']} converter)", "",
                  "| coef | branch | mean | SD | range |", "|---|---|---|---|---|"]
        for c in COEFS:
            d = r["descriptives"][c]
            lines.append(f"| {c} | {d['branch']} | {d['mean']:.4f} | {d['sd']:.4f} | "
                         f"{d['min']:.4f}–{d['max']:.4f} |")
        lines += ["", "| target | coef | n | Spearman | 95% CI | p |",
                  "|---|---|---|---|---|---|"]
        for row in r["correlations"]:
            if "skipped" in row:
                lines.append(f"| {row['target']} | — | {row['n']} | skipped ({row['skipped']}) | | |")
                continue
            sp = row["spearman"]
            lines.append(
                f"| {row['target']} | {row['coef']} ({row['branch']}) | {row['n']} | "
                f"{sp['estimate']:.3f} | [{sp['ci_low']:.3f}, {sp['ci_high']:.3f}] | "
                f"{sp['p']:.3f} |")
        if r["converter_contrast"]:
            lines += ["", "Converter minus non-converter, mean gate weight:", ""]
            for c, d in r["converter_contrast"].items():
                lines.append(f"- {c} ({COEF_BRANCH[c]}): {d['estimate']:+.4f} "
                             f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}], p = {d['p']:.3f}")
        lines.append("")
    with open(os.path.join(out_dir, "gate_analysis.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n  wrote {out_dir}/gate_analysis.json, gate_per_pair.json, gate_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
04_oasis3_zeroshot.py
=====================
OASIS-3 zero-shot external validation for TAFNet.

Loads the ADNI-trained FULL TAFNet checkpoint (tafnet_v4_fold1_best.pth) and
applies it, with NO further training, to the preprocessed OASIS-3 longitudinal
pairs produced by preprocess_oasis3.py.

Unlike the OASIS-2 path, this script consumes EXPLICIT, pre-labelled pairs
(oasis3_pairs_preprocessed.csv) rather than rebuilding pairs from a CDR-group
trajectory -- this preserves the handover's labelling (CDR-primary with AD-
etiology confirmation, 36-month horizon, censoring). Tensor preparation reuses
the core `_load_nifti_volume`, so the model sees byte-identical inputs to the
ADNI / OASIS-2 runs: a (1,128,128,128) float32 volume in [0,1].

It reports, per the handover's next-steps:
  * pooled all-pairs AUC (the headline external-validation number) with a
    SUBJECT-CLUSTERED bootstrap 95% CI (resampling subjects, not pairs, because
    19 subjects contribute multiple pairs -- treating pairs as independent would
    understate the CI),
  * AUC stratified by scan-interval bin, so the headline is not attributed
    solely to long intervals (the 6-24mo band is the trained-regime confirmatory
    row),
  * a one-pair-per-subject sensitivity result (each subject's earliest eligible
    pair) as a fully independent-sample check.

Because there are only ~25 converter subjects, the CI is expected to be wide;
that is a property of the cohort, not the model, and the pooled AUC is the
summary to lead with.

Run location
------------
125 pairs is a few minutes on CPU, so this can run on the Mac directly after
preprocessing -- no VM round-trip needed. Point --checkpoint at wherever
tafnet_v4_fold1_best.pth lives (the 12 MB file). It also runs unchanged on
medical-vm, where the checkpoint already sits under ~/oasis2/checkpoints/.

Usage
-----
    python scripts/04_oasis3_zeroshot.py \
        --config configs/oasis2.yaml \
        --pairs-csv oasis3_pairs_preprocessed.csv \
        --checkpoint ${TAFNET_DATA}/tafnet_v4_fold1_best.pth \
        --output-dir ${TAFNET_DATA}/oasis3_results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tafnet.config import load_config                      # noqa: E402
from tafnet.data.datasets import _load_nifti_volume        # noqa: E402
from tafnet.evaluation import compute_all_metrics          # noqa: E402
from tafnet.external import (                               # noqa: E402
    build_tafnet_from_cfg,
    load_full_checkpoint,
    run_inference,
)
from tafnet.utils import get_device, set_seed              # noqa: E402

# Scan-interval bins (months). The 6-24 band matches TAFNet's ADNI training
# regime and is the confirmatory row; the others contextualise the long tail.
INTERVAL_BINS: List[Tuple[str, float, float]] = [
    ("<6mo",      0.0,   6.0),
    ("6-24mo*",   6.0,  24.0),   # * trained regime
    ("24-60mo",  24.0,  60.0),
    (">60mo",    60.0, np.inf),
]


# --------------------------------------------------------------------------- #
# Dataset over EXPLICIT pre-labelled pairs
# --------------------------------------------------------------------------- #
class OASIS3PairsDataset(Dataset):
    """Yields (t1, t2, label) for each row of a preprocessed-pairs CSV.

    Requires columns: baseline_path, followup_path, label. Carries OASISID,
    interval_months, day_baseline as metadata (kept index-aligned with the
    inference outputs, since the loader runs with shuffle=False).
    """

    def __init__(self, csv_path: str, verify_files: bool = True):
        df = pd.read_csv(csv_path)
        for col in ("baseline_path", "followup_path", "label"):
            if col not in df.columns:
                raise ValueError(f"pairs CSV missing required column '{col}'")
        df = df[df["label"].isin([0, 1])].copy()
        if verify_files:
            ok = df.apply(
                lambda r: os.path.exists(str(r["baseline_path"]))
                and os.path.exists(str(r["followup_path"])),
                axis=1,
            )
            n_missing = int((~ok).sum())
            if n_missing:
                print(f"  [warn] {n_missing} pair(s) skipped: a preprocessed "
                      f"volume was not found on disk.")
            df = df[ok].copy()
        self.meta = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        t1 = _load_nifti_volume(str(row["baseline_path"]))
        t2 = _load_nifti_volume(str(row["followup_path"]))
        label = torch.tensor([float(row["label"])], dtype=torch.float32)
        return t1, t2, label


# --------------------------------------------------------------------------- #
# Analysis helpers (unit-testable, model-independent)
# --------------------------------------------------------------------------- #
def subject_bootstrap_auc(
    df: pd.DataFrame, n_boot: int = 2000, seed: int = 42,
) -> Tuple[float, float, float, int]:
    """Subject-clustered bootstrap CI for AUC.

    df needs columns OASISID, y_true, y_proba. Resamples whole subjects (all
    their pairs move together), recomputes AUC on each resample, returns
    (median, lo2.5, hi97.5, n_valid_resamples). Resamples missing a class are
    skipped.
    """
    rng = np.random.default_rng(seed)
    subjects = df["OASISID"].to_numpy()
    uniq = np.unique(subjects)
    groups = {s: df[df["OASISID"] == s] for s in uniq}
    aucs: List[float] = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sample = pd.concat([groups[s] for s in pick], ignore_index=True)
        if sample["y_true"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(sample["y_true"], sample["y_proba"]))
    if not aucs:
        return (float("nan"), float("nan"), float("nan"), 0)
    return (float(np.median(aucs)),
            float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)),
            len(aucs))


def stratified_table(meta: pd.DataFrame) -> List[dict]:
    """Per-interval-bin counts and AUC (where both classes are present)."""
    rows = []
    if "interval_months" not in meta.columns:
        return rows
    for name, lo, hi in INTERVAL_BINS:
        sub = meta[(meta["interval_months"] >= lo) & (meta["interval_months"] < hi)]
        n, n_pos = len(sub), int((sub["y_true"] == 1).sum())
        n_neg = n - n_pos
        auc = (float(roc_auc_score(sub["y_true"], sub["y_proba"]))
               if n_pos > 0 and n_neg > 0 else None)
        rows.append({"bin": name, "n": n, "pos": n_pos, "neg": n_neg, "AUC": auc})
    return rows


def earliest_pair_per_subject(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per subject: the earliest eligible pair (min baseline day, then
    min interval as a tie-break)."""
    sort_cols = [c for c in ("day_baseline", "interval_months") if c in meta.columns]
    if not sort_cols:
        return meta.drop_duplicates(subset=["OASISID"], keep="first")
    return (meta.sort_values(sort_cols)
                .drop_duplicates(subset=["OASISID"], keep="first")
                .reset_index(drop=True))


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OASIS-3 zero-shot external validation.")
    p.add_argument("--config", default="configs/oasis2.yaml",
                   help="YAML providing architecture/checkpoint/eval blocks "
                        "(oasis2.yaml is reused; only --pairs-csv differs).")
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--pairs-csv", default="oasis3_pairs_preprocessed.csv")
    p.add_argument("--checkpoint", default=None,
                   help="Override the checkpoint path (default: cfg.paths.oasis2_checkpoint).")
    p.add_argument("--output-dir", default=None,
                   help="Where to write results (default: cfg.paths.oasis2_output_dir).")
    p.add_argument("--threshold", type=float, default=None,
                   help="Decision threshold (default: cfg.evaluation.threshold).")
    p.add_argument("--bootstrap", type=int, default=2000,
                   help="Subject-clustered bootstrap resamples for the AUC CI.")
    return p.parse_args()


def _expand(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)

    pairs_csv  = _expand(args.pairs_csv)
    checkpoint = _expand(args.checkpoint or cfg.paths.oasis2_checkpoint)
    output_dir = _expand(args.output_dir or cfg.paths.oasis2_output_dir)
    threshold  = args.threshold if args.threshold is not None else cfg.evaluation.threshold

    set_seed(cfg.training.seed)
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/3] Building OASIS-3 pairs dataset (explicit labels, no augmentation)...")
    dataset = OASIS3PairsDataset(pairs_csv, verify_files=True)
    if len(dataset) == 0:
        raise SystemExit("[X] No OASIS-3 pairs found. Check --pairs-csv and the volume paths.")
    meta = dataset.meta
    n_pos = int((meta["label"] == 1).sum())
    print(f"  Pairs: {len(dataset)}  ({n_pos} pMCI / {len(dataset) - n_pos} sMCI)  "
          f"from {meta['OASISID'].nunique()} subjects")

    loader = DataLoader(
        dataset, batch_size=cfg.phase56.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    print("\n[2/3] Loading ADNI-trained TAFNet checkpoint...")
    model = build_tafnet_from_cfg(cfg, use_longitudinal=True, freeze_encoder=False)
    load_full_checkpoint(model, checkpoint, device=device)

    print("\n[3/3] Running zero-shot inference...")
    y_true, y_proba = run_inference(model, loader, device=device)

    # index-aligned because the loader did not shuffle
    meta = meta.copy()
    meta["y_true"] = y_true.astype(int)
    meta["y_proba"] = y_proba

    metrics = compute_all_metrics(y_true, y_proba, threshold=threshold)
    med, lo, hi, n_valid = subject_bootstrap_auc(
        meta[["OASISID", "y_true", "y_proba"]], n_boot=args.bootstrap,
        seed=cfg.training.seed,
    )
    strat = stratified_table(meta)

    one = earliest_pair_per_subject(meta)
    one_metrics = compute_all_metrics(one["y_true"], one["y_proba"], threshold=threshold)
    o_med, o_lo, o_hi, _ = subject_bootstrap_auc(
        one[["OASISID", "y_true", "y_proba"]], n_boot=args.bootstrap,
        seed=cfg.training.seed,
    )

    # ---- report ------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  OASIS-3 ZERO-SHOT RESULTS  (all pairs, pooled)")
    print("=" * 64)
    print(f"  Pairs        : {len(y_true)}  (subjects: {meta['OASISID'].nunique()})")
    print(f"  Prevalence   : {float(y_true.mean()):.3f}")
    print(f"  AUC          : {metrics['AUC']:.3f}   "
          f"[95% CI {lo:.3f}-{hi:.3f}, subject-clustered, {n_valid} resamples]")
    print(f"  Sensitivity  : {metrics['Sensitivity']:.3f}   (@ {threshold:g})")
    print(f"  Specificity  : {metrics['Specificity']:.3f}")
    print(f"  F1           : {metrics['F1']:.3f}")
    print(f"  Accuracy     : {metrics['Accuracy']:.3f}")

    print("\n  AUC by scan interval (* = trained 6-24mo regime):")
    print(f"    {'bin':<10}{'n':>5}{'pos':>5}{'neg':>5}{'AUC':>9}")
    for r in strat:
        auc_s = f"{r['AUC']:.3f}" if r["AUC"] is not None else "  n/a"
        print(f"    {r['bin']:<10}{r['n']:>5}{r['pos']:>5}{r['neg']:>5}{auc_s:>9}")

    print("\n  One-pair-per-subject (earliest pair; independent samples):")
    print(f"    Pairs/subjects : {len(one)}  ({int((one['y_true']==1).sum())} pMCI)")
    print(f"    AUC            : {one_metrics['AUC']:.3f}   "
          f"[95% CI {o_lo:.3f}-{o_hi:.3f}]")
    print(f"    Sensitivity    : {one_metrics['Sensitivity']:.3f}   (@ {threshold:g})")
    print(f"    Specificity    : {one_metrics['Specificity']:.3f}")
    print("=" * 64)

    # ---- write -------------------------------------------------------------
    preds_path   = os.path.join(output_dir, "oasis3_zeroshot_predictions.json")
    metrics_path = os.path.join(output_dir, "oasis3_zeroshot_metrics.json")
    pred_cols = [c for c in ("OASISID", "day_baseline", "day_followup",
                             "interval_months", "y_true", "y_proba")
                 if c in meta.columns]
    with open(preds_path, "w") as f:
        json.dump(meta[pred_cols].to_dict(orient="list"), f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump({
            "n_pairs": int(len(y_true)),
            "n_subjects": int(meta["OASISID"].nunique()),
            "prevalence": float(y_true.mean()),
            "threshold": float(threshold),
            "pooled": metrics,
            "auc_ci_95_subject_clustered": {"median": med, "lo": lo, "hi": hi,
                                            "n_resamples": n_valid},
            "by_interval": strat,
            "one_pair_per_subject": {
                "n": int(len(one)),
                "metrics": one_metrics,
                "auc_ci_95": {"median": o_med, "lo": o_lo, "hi": o_hi},
            },
        }, f, indent=2)

    print(f"\n[OK] Wrote {preds_path}")
    print(f"[OK] Wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

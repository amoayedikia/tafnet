"""
Phase 5/6: full benchmark sweep on the multi-timepoint longitudinal cohort.

Methods (any subset, controlled by config.benchmarks.* flags):
    ResNet3D-18        (single timepoint baseline)
    DenseNet3D-121     (single timepoint baseline)
    Siamese-Subtract   (longitudinal — subtraction fusion)
    CNN-LSTM           (longitudinal — LSTM over (T1, T2))
    TAFNet-InitialOnly (proposed-arch ablation, T1 only)
    TAFNet-Full        (proposed, T1 + T2 with three-branch fusion)

5-fold subject-level stratified CV. The TAFNet-Full fold-1 checkpoint is saved.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from ..data import (LabelledPairDataset,
                    MultiTimepointLongitudinalDataset,
                    subset_longitudinal)
from ..evaluation.metrics import aggregate_fold_metrics, compute_all_metrics
from ..models.tafnet import BRANCH_ORDER, normalise_branches
from ..models import (
    CNNLSTM3D,
    DenseNet3D121,
    ResNet3D18,
    SiameseCNNSubtract,
    TAFNet,
)
from .train_fold import train_model_fold


def _build_model(method_name: str, config, encoder_checkpoint: Optional[str],
                 device: str):
    """Instantiate one of the benchmark models by name."""
    arch = config.architecture

    if method_name == "ResNet3D-18":
        return ResNet3D18(dropout=arch.dropout)

    if method_name == "DenseNet3D-121":
        return DenseNet3D121(dropout=arch.dropout)

    if method_name == "Siamese-Subtract":
        m = SiameseCNNSubtract(
            encoder_channels=arch.encoder_channels,
            use_dcca=arch.use_dcca,
            feature_dim=arch.feature_dim,
            dropout=arch.dropout,
            freeze_encoder=True,
        )
        if encoder_checkpoint:
            m.load_pretrained_encoder(encoder_checkpoint, device)
        return m

    if method_name == "CNN-LSTM":
        m = CNNLSTM3D(
            encoder_channels=arch.encoder_channels,
            use_dcca=arch.use_dcca,
            feature_dim=arch.feature_dim,
            dropout=arch.dropout,
            freeze_encoder=True,
        )
        if encoder_checkpoint:
            m.load_pretrained_encoder(encoder_checkpoint, device)
        return m

    if method_name == "TAFNet-InitialOnly":
        m = TAFNet(
            encoder_channels=arch.encoder_channels,
            use_dcca=arch.use_dcca,
            feature_dim=arch.feature_dim,
            num_heads=arch.num_heads,
            dropout=arch.dropout,
            use_longitudinal=False,
            freeze_encoder=True,
        )
        if encoder_checkpoint:
            m.load_pretrained_encoder(encoder_checkpoint, device)
        return m

    if method_name == "TAFNet-Full":
        m = TAFNet(
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
        )
        if encoder_checkpoint:
            m.load_pretrained_encoder(encoder_checkpoint, device)
        return m

    raise ValueError(f"Unknown method: {method_name}")


def _enabled_methods(config) -> List[Tuple[str, str]]:
    """Return the (display_name, kind) of every benchmark enabled in config."""
    flags = config.benchmarks
    out: List[Tuple[str, str]] = []
    if flags.get("resnet18_single"):
        out.append(("ResNet3D-18", "single"))
    if flags.get("densenet121_single"):
        out.append(("DenseNet3D-121", "single"))
    if flags.get("siamese_subtract"):
        out.append(("Siamese-Subtract", "longitudinal"))
    if flags.get("cnn_lstm"):
        out.append(("CNN-LSTM", "longitudinal"))
    if flags.get("tafnet_initial_only"):
        out.append(("TAFNet-InitialOnly", "ablation"))
    if flags.get("tafnet_full"):
        out.append(("TAFNet-Full", "proposed"))
    return out


def _fold_signature(config, full_ds, train_idx, val_idx, test_idx) -> str:
    """
    Identity of a fold. A cached result is only reused when the cohort, the
    split and the architecture all match, so changing the label file, the seed,
    the holdout fraction or the gate mode invalidates the cache rather than
    silently mixing results from two different experiments.
    """
    arch = config.architecture
    payload = {
        "n_pairs": int(len(full_ds)),
        "n_subjects": int(len(set(full_ds.subjects.tolist()))),
        "labels_sum": int(full_ds.labels.sum()),
        "num_folds": int(config.training.num_folds),
        "seed": int(config.training.seed),
        "holdout_frac": float(getattr(config.paths, "holdout_frac", 0.0) or 0.0),
        "train": sorted(map(int, train_idx)),
        "val": sorted(map(int, val_idx)),
        "test": sorted(map(int, test_idx)),
        "gate_mode": getattr(arch, "gate_mode", "patient"),
        "baseline_residual": bool(getattr(arch, "baseline_residual", True)),
        "num_heads": int(arch.num_heads),
        "feature_dim": int(arch.feature_dim),
    }
    # Branch ablation. MUST affect the signature: without it a diff-only run
    # silently reuses the full model's cached folds and reports its numbers.
    # The key is added ONLY when a branch is actually disabled, so a full model
    # hashes to exactly what it hashed to before this field existed and the
    # existing 30-fold v4 cache stays valid.
    branches = normalise_branches(getattr(arch, "branches", None))
    if tuple(branches) != BRANCH_ORDER:
        payload["branches"] = list(branches)

    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _fold_paths(output_dir: str, method_name: str, fold_idx: int):
    d = os.path.join(output_dir, "folds")
    os.makedirs(d, exist_ok=True)
    safe = method_name.replace("/", "_")
    return (os.path.join(d, f"{safe}_fold{fold_idx+1}.json"),
            os.path.join(d, f"{safe}_fold{fold_idx+1}.pth"))


def _load_fold(json_path: str, signature: str) -> Optional[Dict]:
    """Return a cached fold result, or None if absent or stale."""
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path) as fh:
            rec = json.load(fh)
    except Exception as err:  # noqa: BLE001 - a corrupt cache must not be fatal
        print(f"    [cache] unreadable, will recompute: {err}")
        return None
    if rec.get("signature") != signature:
        print("    [cache] signature changed (cohort, split or architecture) "
              "— recomputing")
        return None
    return rec


def _save_fold(json_path: str, ckpt_path: str, signature: str,
               method_name: str, fold_idx: int, metrics: Dict,
               predictions: Dict, t_metrics: Optional[Dict],
               t_true, t_prob, best_state) -> None:
    rec = {
        "signature": signature,
        "method": method_name,
        "fold": fold_idx + 1,
        "metrics": metrics,
        "predictions": {"y_true": [float(v) for v in predictions["y_true"]],
                        "y_pred": [float(v) for v in predictions["y_pred"]]},
    }
    if t_metrics is not None:
        rec["test_metrics"] = t_metrics
        rec["test_y_true"] = [float(v) for v in t_true]
        rec["test_y_pred"] = [float(v) for v in t_prob]
    tmp = json_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, json_path)          # atomic: a killed process never leaves
    if best_state is not None:          # a half-written cache entry
        torch.save(best_state, ckpt_path)


def _predict(model, loader, device: str, use_amp: bool):
    """Return (y_true, y_prob) for every sample in `loader`. No gradients."""
    model.eval()
    use_longitudinal = getattr(model, "use_longitudinal", False)
    y_true, y_prob = [], []
    with torch.no_grad():
        for t1, t2, labels in loader:
            t1 = t1.to(device)
            t2 = t2.to(device) if use_longitudinal else None
            with autocast(device_type="cuda", enabled=use_amp and device == "cuda"):
                logits = model(t1, t2)
            y_prob.extend(torch.sigmoid(logits.float()).cpu().numpy().flatten())
            y_true.extend(labels.numpy().flatten())
    return np.asarray(y_true), np.asarray(y_prob)


def run_all_benchmarks(config, device: str, encoder_checkpoint: Optional[str]):
    """
    Run every enabled benchmark with 5-fold subject-level CV.

    Returns (all_results, all_predictions, full_dataset).
    """
    print("\n" + "=" * 70)
    print("  PHASE 5/6: BENCHMARK COMPARISON (Multi-Timepoint Longitudinal)")
    print("=" * 70)

    csv_path = config.paths.csv_path
    data_dir = config.paths.data_dir
    output_dir = config.paths.output_dir
    trcfg = config.training
    p56 = config.phase56
    visit_pairs = tuple(tuple(p) for p in config.visit_pairs)

    pairs_csv = getattr(config.paths, "pairs_csv", None)
    holdout_frac = float(getattr(config.paths, "holdout_frac", 0.0) or 0.0)
    use_pairs = bool(pairs_csv)

    if use_pairs:
        # Real conversion labels from label_adni_pairs.py. csv_path and
        # visit_pairs are deliberately unused here: that path labels by the
        # enrolment `Group` column, which is constant per subject.
        full_ds = LabelledPairDataset(
            pairs_csv=pairs_csv, data_dir=data_dir,
            is_training=False, verify_files=True,
        )
    else:
        print("\n[!] paths.pairs_csv is not set — falling back to the enrolment-"
              "`Group` label.\n    For ADNI this trains AD-enrolled vs rest, "
              "NOT conversion. See data/pairs.py.")
        full_ds = MultiTimepointLongitudinalDataset(
            csv_path=csv_path, data_dir=data_dir,
            visit_pairs=visit_pairs, is_training=False, verify_files=True,
        )

    if len(full_ds) < trcfg.num_folds * 2:
        print(f"\n[!] Only {len(full_ds)} pairs — not enough for {trcfg.num_folds}-fold CV.")
        return None, None, full_ds

    test_idx: list = []
    if use_pairs and holdout_frac > 0:
        test_idx, cv_subjects = full_ds.get_holdout_split(
            test_frac=holdout_frac, random_state=trcfg.seed,
        )
        folds = full_ds.get_subject_level_split_indices(
            n_splits=trcfg.num_folds, random_state=trcfg.seed,
            subjects_subset=cv_subjects,
        )
        full_ds.assert_no_subject_leakage(test_idx, folds[0][0], folds[0][1])
        n_test_pos = int(full_ds.labels[test_idx].sum())
        print(f"\n  HELD-OUT TEST SET: {len(test_idx)} pairs / "
              f"{len(set(full_ds.subjects[test_idx].tolist()))} subjects "
              f"(converter pairs={n_test_pos}) — excluded from all CV folds.")
    else:
        if use_pairs:
            print("\n[!] holdout_frac = 0 — reported metrics are best-epoch "
                  "validation on the partition used for early stopping (audit B1).")
        folds = full_ds.get_subject_level_split_indices(
            n_splits=trcfg.num_folds, random_state=trcfg.seed,
        )

    test_loader = None
    if test_idx:
        test_ds = full_ds.subset(test_idx, is_training=False)
        test_loader = DataLoader(
            test_ds, batch_size=p56.batch_size, shuffle=False,
            num_workers=trcfg.num_workers, pin_memory=True,
        )

    methods = _enabled_methods(config)
    print(f"\n  Methods to evaluate: {[m[0] for m in methods]}")
    print(f"  Number of folds: {trcfg.num_folds}")
    print(f"  Total pairs: {len(full_ds)}")

    all_results: Dict[str, Dict] = {}
    all_predictions: Dict[str, Dict[str, list]] = {}

    for method_name, method_type in methods:
        print("\n" + "=" * 70)
        print(f"  METHOD: {method_name} ({method_type})")
        print("=" * 70)

        fold_metrics = []
        fold_predictions: Dict[str, list] = {"y_true": [], "y_pred": []}
        test_fold_metrics: List[Dict] = []
        test_fold_probs: List[np.ndarray] = []
        test_y_true: Optional[np.ndarray] = None

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\n  --- Fold {fold_idx+1}/{trcfg.num_folds} ---")
            train_labels = full_ds.labels[train_idx]
            print(f"  Train: {len(train_idx)} (converter={int(train_labels.sum())})")
            print(f"  Val:   {len(val_idx)} "
                  f"(converter={int(full_ds.labels[val_idx].sum())})")

            # ---- resume ------------------------------------------------
            # A completed fold is written to disk before the next one starts,
            # so a Spot reclamation (or any crash) costs the fold in progress,
            # not the run. Re-running the same command picks up where it left
            # off; see _fold_signature for what invalidates a cached fold.
            sig = _fold_signature(config, full_ds, train_idx, val_idx, test_idx)
            fold_json, fold_ckpt = _fold_paths(output_dir, method_name, fold_idx)
            cached = _load_fold(fold_json, sig)

            if cached is not None:
                metrics = cached["metrics"]
                predictions = cached["predictions"]
                t_metrics = cached.get("test_metrics")
                print(f"  [resumed from {os.path.basename(fold_json)}]  "
                      f"AUC={metrics['AUC']:.4f}"
                      + (f"  test AUC={t_metrics['AUC']:.4f}" if t_metrics else ""))
                if t_metrics is not None:
                    test_fold_metrics.append(t_metrics)
                    test_fold_probs.append(np.asarray(cached["test_y_pred"]))
                    test_y_true = np.asarray(cached["test_y_true"])
            else:
                if use_pairs:
                    train_ds = full_ds.subset(train_idx, is_training=True)
                    val_ds = full_ds.subset(val_idx, is_training=False)
                else:
                    train_ds = subset_longitudinal(
                        full_ds, train_idx, csv_path=csv_path, data_dir=data_dir,
                        visit_pairs=visit_pairs, is_training=True,
                    )
                    val_ds = subset_longitudinal(
                        full_ds, val_idx, csv_path=csv_path, data_dir=data_dir,
                        visit_pairs=visit_pairs, is_training=False,
                    )

                train_loader = DataLoader(
                    train_ds, batch_size=p56.batch_size, shuffle=True,
                    num_workers=trcfg.num_workers, pin_memory=True,
                )
                val_loader = DataLoader(
                    val_ds, batch_size=p56.batch_size, shuffle=False,
                    num_workers=trcfg.num_workers, pin_memory=True,
                )

                model = _build_model(method_name, config, encoder_checkpoint, device)

                metrics, predictions, best_state = train_model_fold(
                    model, train_loader, val_loader, train_labels,
                    fold_idx, config, device, model_name=method_name,
                )

                print(f"  Fold {fold_idx+1} Results: "
                      f"AUC={metrics['AUC']:.4f}, "
                      f"Sens={metrics['Sensitivity']:.3f}, "
                      f"Spec={metrics['Specificity']:.3f}, "
                      f"F1={metrics['F1']:.3f}")

                # train_model_fold reloads best_state into `model` before
                # returning, so this is the early-stopped model, scored on data
                # no fold saw.
                t_metrics = t_true = t_prob = None
                if test_loader is not None:
                    t_true, t_prob = _predict(model, test_loader, device,
                                              trcfg.use_amp)
                    t_metrics = compute_all_metrics(t_true, t_prob)
                    test_fold_metrics.append(t_metrics)
                    test_fold_probs.append(t_prob)
                    test_y_true = t_true
                    print(f"    held-out test: AUC={t_metrics['AUC']:.4f}, "
                          f"Sens={t_metrics['Sensitivity']:.3f}, "
                          f"Spec={t_metrics['Specificity']:.3f}")

                _save_fold(fold_json, fold_ckpt, sig, method_name, fold_idx,
                           metrics, predictions, t_metrics, t_true, t_prob,
                           best_state)

                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

            fold_metrics.append(metrics)
            fold_predictions["y_true"].extend(predictions["y_true"])
            fold_predictions["y_pred"].extend(predictions["y_pred"])

            # Legacy filename kept for downstream analysis notebooks
            if method_name == "TAFNet-Full" and fold_idx == 0 \
                    and os.path.exists(fold_ckpt):
                legacy = os.path.join(output_dir, "tafnet_v4_fold1_best.pth")
                if not os.path.exists(legacy):
                    import shutil
                    shutil.copyfile(fold_ckpt, legacy)
                    print(f"  Saved: {legacy}")

        aggregated = aggregate_fold_metrics(fold_metrics)

        if test_fold_metrics:
            # (a) per-fold test scores, and (b) the fold ensemble: mean
            # probability across folds, which is the single number to headline.
            test_agg = aggregate_fold_metrics(test_fold_metrics)
            ens_prob = np.mean(np.vstack(test_fold_probs), axis=0)
            ens_metrics = compute_all_metrics(test_y_true, ens_prob)
            for k, v in test_agg.items():
                aggregated[f"Test_{k}"] = v
            for k, v in ens_metrics.items():
                aggregated[f"TestEnsemble_{k}"] = v
            aggregated["Test_n_pairs"] = int(len(test_y_true))
            aggregated["Test_n_positive"] = int(test_y_true.sum())
            fold_predictions["test_y_true"] = test_y_true.tolist()
            fold_predictions["test_y_pred_ensemble"] = ens_prob.tolist()
            fold_predictions["test_y_pred_folds"] = [
                p.tolist() for p in test_fold_probs
            ]

        all_results[method_name] = aggregated
        all_predictions[method_name] = fold_predictions

        print(f"\n  {method_name} SUMMARY:")
        print(f"    AUC:         {aggregated['AUC_mean']:.4f} +/- {aggregated['AUC_std']:.4f}")
        print(f"    Sensitivity: {aggregated['Sensitivity_mean']:.4f} +/- {aggregated['Sensitivity_std']:.4f}")
        print(f"    Specificity: {aggregated['Specificity_mean']:.4f} +/- {aggregated['Specificity_std']:.4f}")
        print(f"    F1:          {aggregated['F1_mean']:.4f} +/- {aggregated['F1_std']:.4f}")
        print(f"    Accuracy:    {aggregated['Accuracy_mean']:.4f} +/- {aggregated['Accuracy_std']:.4f}")

        if "Test_AUC_mean" in aggregated:
            print(f"\n  {method_name} HELD-OUT TEST "
                  f"({aggregated['Test_n_pairs']} pairs, "
                  f"{aggregated['Test_n_positive']} positive):")
            print(f"    AUC (per-fold):  {aggregated['Test_AUC_mean']:.4f} "
                  f"+/- {aggregated['Test_AUC_std']:.4f}")
            print(f"    AUC (ensemble):  {aggregated['TestEnsemble_AUC']:.4f}"
                  f"   <-- report this one")
            print(f"    Sens/Spec (ens): {aggregated['TestEnsemble_Sensitivity']:.3f}"
                  f" / {aggregated['TestEnsemble_Specificity']:.3f}")
            gap = aggregated['AUC_mean'] - aggregated['TestEnsemble_AUC']
            print(f"    CV - test gap:   {gap:+.4f}"
                  f"   (CV is best-epoch validation, so a positive gap is the "
                  f"optimism audit B1 describes)")

    return all_results, all_predictions, full_ds

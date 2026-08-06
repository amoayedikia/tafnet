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

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..data import MultiTimepointLongitudinalDataset, subset_longitudinal
from ..evaluation.metrics import aggregate_fold_metrics
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

    full_ds = MultiTimepointLongitudinalDataset(
        csv_path=csv_path, data_dir=data_dir,
        visit_pairs=visit_pairs, is_training=False, verify_files=True,
    )
    if len(full_ds) < trcfg.num_folds * 2:
        print(f"\n[!] Only {len(full_ds)} pairs — not enough for {trcfg.num_folds}-fold CV.")
        return None, None, full_ds

    folds = full_ds.get_subject_level_split_indices(
        n_splits=trcfg.num_folds, random_state=trcfg.seed,
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

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\n  --- Fold {fold_idx+1}/{trcfg.num_folds} ---")
            train_labels = full_ds.labels[train_idx]
            print(f"  Train: {len(train_idx)} (converter={int(train_labels.sum())})")
            print(f"  Val:   {len(val_idx)} "
                  f"(converter={int(full_ds.labels[val_idx].sum())})")

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

            fold_metrics.append(metrics)
            fold_predictions["y_true"].extend(predictions["y_true"])
            fold_predictions["y_pred"].extend(predictions["y_pred"])

            print(f"  Fold {fold_idx+1} Results: "
                  f"AUC={metrics['AUC']:.4f}, "
                  f"Sens={metrics['Sensitivity']:.3f}, "
                  f"Spec={metrics['Specificity']:.3f}, "
                  f"F1={metrics['F1']:.3f}")

            # Save TAFNet-Full fold 1 checkpoint (useful for downstream analysis)
            if method_name == "TAFNet-Full" and fold_idx == 0 and best_state is not None:
                ckpt = os.path.join(output_dir, "tafnet_v4_fold1_best.pth")
                torch.save(best_state, ckpt)
                print(f"  Saved: {ckpt}")

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        aggregated = aggregate_fold_metrics(fold_metrics)
        all_results[method_name] = aggregated
        all_predictions[method_name] = fold_predictions

        print(f"\n  {method_name} SUMMARY:")
        print(f"    AUC:         {aggregated['AUC_mean']:.4f} +/- {aggregated['AUC_std']:.4f}")
        print(f"    Sensitivity: {aggregated['Sensitivity_mean']:.4f} +/- {aggregated['Sensitivity_std']:.4f}")
        print(f"    Specificity: {aggregated['Specificity_mean']:.4f} +/- {aggregated['Specificity_std']:.4f}")
        print(f"    F1:          {aggregated['F1_mean']:.4f} +/- {aggregated['F1_std']:.4f}")
        print(f"    Accuracy:    {aggregated['Accuracy_mean']:.4f} +/- {aggregated['Accuracy_std']:.4f}")

    return all_results, all_predictions, full_ds

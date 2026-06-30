#!/usr/bin/env python3
"""
Stage 5: OASIS-2 warm-start fine-tuning A/B comparison (subject-level 5-fold CV).

For each fold, the SAME ADNI-trained checkpoint is:
    (a) evaluated zero-shot on the fold's test partition, then
    (b) warm-start fine-tuned on the fold's train partition (encoder frozen,
        only the Temporal Fusion Module + classifier trainable, LR 1e-5,
        class-weighted BCE, fixed number of epochs, no early stopping), then
    (c) re-evaluated on the SAME test partition.

Because both arms are scored on identical test pairs within each fold, the
comparison isolates the contribution of cohort-specific adaptation from
test-set composition. This reproduces the manuscript's null fine-tuning result
(mean dAUC ~ +0.008, Wilcoxon p ~ 0.31) and the operating-point shift
(sensitivity up, specificity down) driven by the heavy positive-class weight.

Subject-level folds come from the core dataset's
``get_subject_level_split_indices`` — the same splitter the main training code
uses — so there is no separate split-CSV step.

Example
-------
    python scripts/05_oasis2_finetune_cv.py --config configs/oasis2.yaml

    # quick smoke test (2 folds, 2 epochs)
    python scripts/05_oasis2_finetune_cv.py --config configs/oasis2.yaml \\
        --override oasis2_finetune.epochs=2 --override training.num_folds=2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tafnet.config import load_config                       # noqa: E402
from tafnet.data import subset_longitudinal                 # noqa: E402
from tafnet.evaluation import compute_all_metrics           # noqa: E402
from tafnet.external import (                                # noqa: E402
    DEFAULT_OASIS2_VISIT_PAIRS,
    build_oasis2_dataset,
    build_tafnet_from_cfg,
    load_full_checkpoint,
    pos_weight_from_labels,
    run_inference,
)
from tafnet.utils import get_device, set_seed, verify_drive_mount  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OASIS-2 fine-tuning A/B (5-fold CV).")
    p.add_argument("--config", required=True, help="Path to OASIS-2 YAML config.")
    p.add_argument("--override", action="append", default=[],
                   help="Override key.path=value (repeatable).")
    p.add_argument("--no-drive-check", action="store_true")
    return p.parse_args()


def _expand(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def _make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


def finetune_one_fold(cfg, device, checkpoint, train_ds, pos_weight):
    """Warm-start fine-tune a fresh checkpoint on one fold's train split."""
    model = build_tafnet_from_cfg(cfg, use_longitudinal=True, freeze_encoder=True)
    load_full_checkpoint(model, checkpoint, device=device)
    model.to(device)

    # Encoder frozen: no grad + eval mode so BatchNorm stats don't drift on the
    # small OASIS-2 cohort. Only fusion + classifier adapt.
    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.oasis2_finetune.lr,
        weight_decay=cfg.training.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    loader = _make_loader(train_ds, cfg.oasis2_finetune.batch_size, True,
                          cfg.training.num_workers)

    n_epochs = int(cfg.oasis2_finetune.epochs)
    for epoch in range(1, n_epochs + 1):
        model.train()
        model.encoder.eval()  # keep frozen encoder's BN in eval mode
        running = 0.0
        n_batches = 0
        for t1, t2, label in loader:
            t1, t2, label = t1.to(device), t2.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(t1, t2)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
        avg = running / max(n_batches, 1)
        print(f"      epoch {epoch:02d}/{n_epochs}  train_loss={avg:.4f}")

    return model  # final-epoch weights (no early stopping)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)

    csv_path   = _expand(cfg.paths.oasis2_csv)
    data_dir   = _expand(cfg.paths.oasis2_data_dir)
    checkpoint = _expand(cfg.paths.oasis2_checkpoint)
    output_dir = _expand(cfg.paths.oasis2_output_dir)

    if not args.no_drive_check:
        verify_drive_mount(cfg.paths.drive_root)

    set_seed(cfg.training.seed)
    device = get_device()
    results_dir = os.path.join(output_dir, "results_finetune")
    os.makedirs(results_dir, exist_ok=True)

    visit_pairs = DEFAULT_OASIS2_VISIT_PAIRS

    print("\n[1/2] Building full OASIS-2 dataset for subject-level splitting...")
    full_ds = build_oasis2_dataset(
        csv_path=csv_path, data_dir=data_dir,
        visit_pairs=visit_pairs, is_training=False,
    )
    if len(full_ds) == 0:
        raise SystemExit("[X] No OASIS-2 pairs found. Check oasis2_csv / oasis2_data_dir.")

    folds = full_ds.get_subject_level_split_indices(
        n_splits=cfg.training.num_folds, random_state=cfg.training.seed,
    )

    print("\n[2/2] Running per-fold zero-shot vs fine-tuned A/B comparison...")
    per_fold = []
    pooled = {"zs_true": [], "zs_pred": [], "ft_true": [], "ft_pred": []}

    for fi, (train_idx, test_idx) in enumerate(folds, 1):
        print(f"\n  ===== Fold {fi}/{len(folds)} "
              f"(train={len(train_idx)} pairs, test={len(test_idx)} pairs) =====")

        train_ds = subset_longitudinal(
            full_ds, train_idx, csv_path, data_dir, visit_pairs, is_training=True,
        )
        test_ds = subset_longitudinal(
            full_ds, test_idx, csv_path, data_dir, visit_pairs, is_training=False,
        )
        test_loader = _make_loader(test_ds, cfg.phase56.batch_size, False,
                                   cfg.training.num_workers)

        # (a) zero-shot on the test partition (fresh, untouched checkpoint)
        zs_model = build_tafnet_from_cfg(cfg, use_longitudinal=True)
        load_full_checkpoint(zs_model, checkpoint, device=device)
        zs_true, zs_pred = run_inference(zs_model, test_loader, device=device)
        zs_metrics = compute_all_metrics(zs_true, zs_pred, threshold=cfg.evaluation.threshold)

        # (b) warm-start fine-tune on the train partition
        pos_weight = pos_weight_from_labels(train_ds.labels)
        print(f"    pos_weight (neg/pos on train) = {pos_weight:.2f}")
        ft_model = finetune_one_fold(cfg, device, checkpoint, train_ds, pos_weight)

        # (c) fine-tuned on the SAME test partition
        ft_true, ft_pred = run_inference(ft_model, test_loader, device=device)
        ft_metrics = compute_all_metrics(ft_true, ft_pred, threshold=cfg.evaluation.threshold)

        d_auc = ft_metrics["AUC"] - zs_metrics["AUC"]
        print(f"    ZS  AUC={zs_metrics['AUC']:.3f}  Sens={zs_metrics['Sensitivity']:.3f}  "
              f"Spec={zs_metrics['Specificity']:.3f}")
        print(f"    FT  AUC={ft_metrics['AUC']:.3f}  Sens={ft_metrics['Sensitivity']:.3f}  "
              f"Spec={ft_metrics['Specificity']:.3f}   dAUC={d_auc:+.3f}")

        fold_dir = os.path.join(results_dir, f"fold{fi}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(ft_model.state_dict(),
                   os.path.join(fold_dir, f"tafnet_oasis2_fold{fi}_best.pth"))
        with open(os.path.join(fold_dir, "zeroshot_test_predictions.json"), "w") as f:
            json.dump({"y_true": zs_true.tolist(), "y_pred": zs_pred.tolist()}, f, indent=2)
        with open(os.path.join(fold_dir, "finetuned_test_predictions.json"), "w") as f:
            json.dump({"y_true": ft_true.tolist(), "y_pred": ft_pred.tolist()}, f, indent=2)

        per_fold.append({
            "fold": fi,
            "n_train": len(train_idx), "n_test": len(test_idx),
            "n_test_pos": int(zs_true.sum()),
            "zeroshot": zs_metrics, "finetuned": ft_metrics, "dAUC": d_auc,
        })
        pooled["zs_true"].extend(zs_true.tolist())
        pooled["zs_pred"].extend(zs_pred.tolist())
        pooled["ft_true"].extend(ft_true.tolist())
        pooled["ft_pred"].extend(ft_pred.tolist())

    # -------- aggregate --------
    zs_aucs = np.array([f["zeroshot"]["AUC"] for f in per_fold])
    ft_aucs = np.array([f["finetuned"]["AUC"] for f in per_fold])
    pooled_zs = compute_all_metrics(pooled["zs_true"], pooled["zs_pred"],
                                    threshold=cfg.evaluation.threshold)
    pooled_ft = compute_all_metrics(pooled["ft_true"], pooled["ft_pred"],
                                    threshold=cfg.evaluation.threshold)
    try:
        _, p_wilcox = wilcoxon(ft_aucs, zs_aucs)
    except Exception:    # noqa: BLE001 — e.g. all-zero differences
        p_wilcox = 1.0

    comparison = {
        "n_folds": len(per_fold),
        "zeroshot_AUC_mean": float(zs_aucs.mean()),
        "zeroshot_AUC_std": float(zs_aucs.std()),
        "finetuned_AUC_mean": float(ft_aucs.mean()),
        "finetuned_AUC_std": float(ft_aucs.std()),
        "dAUC_mean": float((ft_aucs - zs_aucs).mean()),
        "wilcoxon_p": float(p_wilcox),
        "pooled_zeroshot_AUC": pooled_zs["AUC"],
        "pooled_finetuned_AUC": pooled_ft["AUC"],
        "pooled_zeroshot_Sensitivity": pooled_zs["Sensitivity"],
        "pooled_finetuned_Sensitivity": pooled_ft["Sensitivity"],
        "pooled_zeroshot_Specificity": pooled_zs["Specificity"],
        "pooled_finetuned_Specificity": pooled_ft["Specificity"],
        "per_fold": per_fold,
    }
    out_path = os.path.join(results_dir, "comparison.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 64)
    print("  OASIS-2 FINE-TUNING A/B SUMMARY")
    print("=" * 64)
    print(f"  Zero-shot  AUC : {zs_aucs.mean():.3f} ± {zs_aucs.std():.3f}  "
          f"(pooled {pooled_zs['AUC']:.3f})")
    print(f"  Fine-tuned AUC : {ft_aucs.mean():.3f} ± {ft_aucs.std():.3f}  "
          f"(pooled {pooled_ft['AUC']:.3f})")
    print(f"  dAUC (mean)    : {(ft_aucs - zs_aucs).mean():+.3f}   "
          f"Wilcoxon p = {p_wilcox:.3f}")
    print(f"  Operating point: Sens {pooled_zs['Sensitivity']:.3f}->{pooled_ft['Sensitivity']:.3f}, "
          f"Spec {pooled_zs['Specificity']:.3f}->{pooled_ft['Specificity']:.3f}")
    print(f"\n[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Phase 4: pretrain JDACEncoder3D on cross-sectional CN vs AD scans.

Split is by SUBJECT (not by scan) to prevent leakage between train and val.
Outputs the best checkpoint to {output_dir}/phase4_encoder_best.pth.
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..data import CrossSectionalDataset, subset_cross_sectional
from ..evaluation.metrics import compute_all_metrics
from ..models import Phase4Model


def train_phase4(config, device: str) -> Tuple[Optional[str], float]:
    """
    Pretrain encoder on cross-sectional CN vs AD.

    Returns
    -------
    (checkpoint_path, best_auc)
        checkpoint_path is None if training failed (e.g. too few samples).
    """
    print("\n" + "=" * 70)
    print("  PHASE 4: ENCODER PRETRAINING (Cross-Sectional CN vs AD)")
    print("=" * 70)

    csv_path = config.paths.csv_path
    data_dir = config.paths.data_dir
    output_dir = config.paths.output_dir
    os.makedirs(output_dir, exist_ok=True)

    arch = config.architecture
    p4 = config.phase4
    trcfg = config.training

    # Load full cohort once with verification (slow but worth it)
    full_ds = CrossSectionalDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        is_training=False,
        verify_files=True,
    )

    if len(full_ds) < 50:
        print(f"\n[!] Only {len(full_ds)} samples — too few for pretraining.")
        return None, 0.0

    # Subject-level split (avoids leakage)
    subjects = list({s["subject"] for s in full_ds.samples})
    subject_labels = []
    for subj in subjects:
        for s in full_ds.samples:
            if s["subject"] == subj:
                subject_labels.append(s["label"])
                break
    subject_labels = np.array(subject_labels)

    train_subjs, val_subjs = train_test_split(
        subjects, test_size=0.2, stratify=subject_labels,
        random_state=trcfg.seed,
    )
    train_set = set(train_subjs)
    val_set = set(val_subjs)
    train_indices = [i for i, s in enumerate(full_ds.samples)
                     if s["subject"] in train_set]
    val_indices = [i for i, s in enumerate(full_ds.samples)
                   if s["subject"] in val_set]

    train_ds = subset_cross_sectional(
        full_ds, train_indices,
        csv_path=csv_path, data_dir=data_dir, is_training=True,
    )
    val_ds = subset_cross_sectional(
        full_ds, val_indices,
        csv_path=csv_path, data_dir=data_dir, is_training=False,
    )

    n_tr_pos = int(train_ds.labels.sum())
    n_val_pos = int(val_ds.labels.sum())
    print(f"\n  Train: {len(train_ds)} scans ({len(train_subjs)} subjects) - AD={n_tr_pos}")
    print(f"  Val:   {len(val_ds)} scans ({len(val_subjs)} subjects) - AD={n_val_pos}")

    train_loader = DataLoader(
        train_ds, batch_size=p4.batch_size, shuffle=True,
        num_workers=trcfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=p4.batch_size, shuffle=False,
        num_workers=trcfg.num_workers, pin_memory=True,
    )

    model = Phase4Model(
        encoder_channels=arch.encoder_channels,
        use_dcca=arch.use_dcca,
        feature_dim=arch.feature_dim,
        dropout=arch.dropout,
    ).to(device)
    print(f"\n  Model parameters: {model.count_parameters():,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=p4.lr, weight_decay=trcfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    n_neg = len(train_ds.labels) - n_tr_pos
    if n_tr_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / n_tr_pos], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"  pos_weight={pos_weight.item():.2f}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    scaler = GradScaler(enabled=trcfg.use_amp)

    best_auc = 0.0
    patience_counter = 0
    best_state = None

    print(f"\n  Training for up to {p4.epochs} epochs...\n")

    for epoch in range(p4.epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            with autocast(device_type="cuda", enabled=trcfg.use_amp):
                logits = model(imgs)
                loss = criterion(logits, labels) / trcfg.accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % trcfg.accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * trcfg.accumulation_steps

        scheduler.step(epoch)
        avg_loss = epoch_loss / max(len(train_loader), 1)

        # ---- Validate
        model.eval()
        all_targets, all_probs = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                with autocast(device_type="cuda", enabled=trcfg.use_amp):
                    logits = model(imgs)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_targets.extend(labels.numpy().flatten())
                all_probs.extend(probs)

        metrics = compute_all_metrics(all_targets, all_probs)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"  Ep {epoch+1:3d}/{p4.epochs} | "
              f"Loss: {avg_loss:.4f} | AUC: {metrics['AUC']:.4f} | "
              f"Sens: {metrics['Sensitivity']:.3f} | "
              f"Spec: {metrics['Specificity']:.3f} | "
              f"LR: {lr:.2e} | {elapsed:.0f}s")

        if metrics["AUC"] > best_auc:
            best_auc = metrics["AUC"]
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            print(f"    >>> Best AUC={best_auc:.4f} - saved.")
        else:
            patience_counter += 1
            if patience_counter >= p4.patience:
                print(f"  Early stopping at epoch {epoch+1}.")
                break

    checkpoint_path = os.path.join(output_dir, "phase4_encoder_best.pth")
    if best_state is not None:
        torch.save(best_state, checkpoint_path)
        print(f"\n  [OK] Phase 4 complete. Best AUC: {best_auc:.4f}")
        print(f"  Saved: {checkpoint_path}")
    else:
        print("\n  [!] No improvement during training.")
        checkpoint_path = None

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return checkpoint_path, best_auc

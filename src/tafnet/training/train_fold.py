"""
Single-fold training function.

Trains any model that takes (x_t1, x_t2_or_None) as input. Supports:
    * mixed precision via torch.amp
    * gradient accumulation
    * cosine annealing with warm restarts
    * pos_weight-balanced BCE for class imbalance
    * grad clipping (max_norm=1.0)
    * early stopping on validation AUC

Returns the best fold metrics, predictions for ROC curves, and best state dict.
"""
from __future__ import annotations

import copy
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..evaluation.metrics import compute_all_metrics


def train_model_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_labels: np.ndarray,
    fold_idx: int,
    config,
    device: str,
    model_name: str = "Model",
) -> Tuple[Dict, Dict, Dict]:
    """Train one fold of one method. Returns (metrics, predictions, best_state)."""
    p56 = config.phase56
    trcfg = config.training

    model = model.to(device)
    use_longitudinal = getattr(model, "use_longitudinal", False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=p56.lr,
                            weight_decay=trcfg.weight_decay)
    print(f"  [{model_name}] Optimizing "
          f"{sum(p.numel() for p in trainable_params):,} parameters")
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    n_pos = int(train_labels.sum())
    n_neg = len(train_labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    scaler = GradScaler(enabled=trcfg.use_amp)

    best_auc = 0.0
    patience_counter = 0
    best_state = None
    best_predictions: Dict | None = None

    for epoch in range(p56.epochs):
        t0 = time.time()
        model.train()
        # Keep encoder in eval mode if frozen — BatchNorm stats should stay fixed
        if getattr(model, "freeze_encoder", False) and hasattr(model, "encoder"):
            model.encoder.eval()

        epoch_loss = 0.0
        optimizer.zero_grad()

        for i, (t1, t2, labels) in enumerate(train_loader):
            t1 = t1.to(device)
            t2 = t2.to(device) if use_longitudinal else None
            labels = labels.to(device)

            with autocast(device_type="cuda", enabled=trcfg.use_amp):
                logits = model(t1, t2)
                loss = criterion(logits, labels) / trcfg.accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % trcfg.accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
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
            for t1, t2, labels in val_loader:
                t1 = t1.to(device)
                t2 = t2.to(device) if use_longitudinal else None
                with autocast(device_type="cuda", enabled=trcfg.use_amp):
                    logits = model(t1, t2)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_targets.extend(labels.numpy().flatten())
                all_probs.extend(probs)

        metrics = compute_all_metrics(all_targets, all_probs)
        elapsed = time.time() - t0

        if (epoch + 1) % 5 == 0 or epoch == 0 or metrics["AUC"] > best_auc:
            print(f"    Ep {epoch+1:2d} | Loss: {avg_loss:.4f} | "
                  f"AUC: {metrics['AUC']:.4f} | "
                  f"Sens: {metrics['Sensitivity']:.3f} | "
                  f"Spec: {metrics['Specificity']:.3f} | {elapsed:.0f}s")

        if metrics["AUC"] > best_auc:
            best_auc = metrics["AUC"]
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            best_predictions = {
                "y_true": list(all_targets),
                "y_pred": list(all_probs),
            }
        else:
            patience_counter += 1
            if patience_counter >= p56.patience:
                print(f"    Early stopping at epoch {epoch+1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = compute_all_metrics(
        best_predictions["y_true"], best_predictions["y_pred"],
    )
    return final_metrics, best_predictions, best_state

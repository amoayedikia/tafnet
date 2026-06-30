#!/usr/bin/env python3
"""
Stage 4: OASIS-2 zero-shot external validation.

Loads the ADNI-trained FULL TAFNet checkpoint and applies it, with NO further
training, to every eligible OASIS-2 longitudinal pair. Reports AUC, sensitivity,
specificity, F1, and accuracy, and writes raw predictions for downstream ROC /
analysis.

This reproduces the headline external-validation number (full-cohort
AUC 0.788) reported in the manuscript.

Example
-------
    python scripts/04_oasis2_zeroshot.py --config configs/oasis2.yaml

    python scripts/04_oasis2_zeroshot.py \\
        --config configs/oasis2.yaml \\
        --override paths.oasis2_csv=~/oasis2/oasis2_inventory_mci_only.csv \\
        --override paths.oasis2_output_dir=~/oasis2/results_mci_only \\
        --no-drive-check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tafnet.config import load_config                       # noqa: E402
from tafnet.evaluation import compute_all_metrics           # noqa: E402
from tafnet.external import (                                # noqa: E402
    build_oasis2_dataset,
    build_tafnet_from_cfg,
    load_full_checkpoint,
    run_inference,
)
from tafnet.utils import get_device, set_seed, verify_drive_mount  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OASIS-2 zero-shot external validation.")
    p.add_argument("--config", required=True, help="Path to OASIS-2 YAML config.")
    p.add_argument("--override", action="append", default=[],
                   help="Override key.path=value (repeatable).")
    p.add_argument("--no-drive-check", action="store_true",
                   help="Skip the rclone Drive-mount sanity check.")
    return p.parse_args()


def _expand(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


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
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/3] Building OASIS-2 dataset (zero-shot, no augmentation)...")
    dataset = build_oasis2_dataset(
        csv_path=csv_path, data_dir=data_dir, is_training=False,
    )
    if len(dataset) == 0:
        raise SystemExit("[X] No OASIS-2 pairs found. Check oasis2_csv / oasis2_data_dir.")

    loader = DataLoader(
        dataset, batch_size=cfg.phase56.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    print("\n[2/3] Loading ADNI-trained TAFNet checkpoint...")
    model = build_tafnet_from_cfg(cfg, use_longitudinal=True, freeze_encoder=False)
    load_full_checkpoint(model, checkpoint, device=device)

    print("\n[3/3] Running zero-shot inference...")
    y_true, y_proba = run_inference(model, loader, device=device)
    metrics = compute_all_metrics(y_true, y_proba, threshold=cfg.evaluation.threshold)

    print("\n" + "=" * 60)
    print("  OASIS-2 ZERO-SHOT RESULTS")
    print("=" * 60)
    print(f"  Pairs       : {len(y_true)}")
    print(f"  Prevalence  : {float(y_true.mean()):.3f}")
    print(f"  AUC         : {metrics['AUC']:.3f}")
    print(f"  Sensitivity : {metrics['Sensitivity']:.3f}")
    print(f"  Specificity : {metrics['Specificity']:.3f}")
    print(f"  F1          : {metrics['F1']:.3f}")
    print(f"  Accuracy    : {metrics['Accuracy']:.3f}")

    preds_path   = os.path.join(output_dir, "oasis2_zeroshot_predictions.json")
    metrics_path = os.path.join(output_dir, "oasis2_zeroshot_metrics.json")
    with open(preds_path, "w") as f:
        json.dump({"y_true": y_true.tolist(), "y_pred": y_proba.tolist()}, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[OK] Wrote {preds_path}")
    print(f"[OK] Wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

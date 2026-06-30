#!/usr/bin/env python3
"""
Stage 2: Train TAFNet and all benchmark methods.

This is the canonical entry point — the Python equivalent of running
TAFNet_v4_Comprehensive.ipynb end-to-end:

    PHASE 4  : Pretrain the JDAC encoder on cross-sectional CN/AD scans.
    PHASE 5/6: Train each benchmark (incl. TAFNet) with 5-fold subject-level
               CV on the multi-timepoint longitudinal cohort.
    REPORT   : Print summary table, save JSON, ROC curves, AUC box plot.

All outputs land under config.paths.output_dir.

Example
-------
    python scripts/02_train.py --config configs/default.yaml

Run a faster sanity check by overriding epochs and disabling the heaviest
benchmarks:

    python scripts/02_train.py --config configs/default.yaml \\
        --override phase4.epochs=2 \\
        --override phase56.epochs=2 \\
        --override benchmarks.densenet121_single=false \\
        --override benchmarks.cnn_lstm=false
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tafnet.config import dump_config, load_config
from tafnet.evaluation import (
    generate_results_summary,
    plot_boxplots,
    plot_roc_curves,
)
from tafnet.training import run_all_benchmarks, train_phase4
from tafnet.utils import get_device, set_seed, verify_drive_mount


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True,
                        help="Path to training YAML config.")
    parser.add_argument("--override", action="append", default=[],
                        help="Override key.path=value (repeatable).")
    parser.add_argument("--skip-phase4", action="store_true",
                        help="Re-use existing phase4_encoder_best.pth in output_dir.")
    parser.add_argument("--no-drive-check", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    if not args.no_drive_check:
        verify_drive_mount(cfg.paths.drive_root)

    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    # Archive the exact config used for this run
    dump_config(cfg, os.path.join(cfg.paths.output_dir, "run_config.yaml"))

    set_seed(cfg.training.seed)
    device = get_device()

    # -------- PHASE 4 --------
    ckpt_default = os.path.join(cfg.paths.output_dir, "phase4_encoder_best.pth")
    if args.skip_phase4 and os.path.exists(ckpt_default):
        encoder_checkpoint = ckpt_default
        print(f"\n[skip-phase4] Reusing {encoder_checkpoint}")
        phase4_auc = float("nan")
    else:
        encoder_checkpoint, phase4_auc = train_phase4(cfg, device)
        if encoder_checkpoint is None:
            print("\n[X] Phase 4 failed. Cannot continue.")
            return 1
        print(f"\n  Phase 4 encoder saved: {encoder_checkpoint}")
        print(f"  Phase 4 validation AUC: {phase4_auc:.4f}")

    # -------- PHASE 5/6 --------
    all_results, all_predictions, _full_ds = run_all_benchmarks(
        cfg, device, encoder_checkpoint,
    )
    if all_results is None:
        print("\n[X] No results — too few pairs for CV.")
        return 1
    print("\n[OK] All benchmarks complete.")

    # -------- Report --------
    generate_results_summary(all_results, all_predictions, cfg.paths.output_dir)
    plot_roc_curves(all_predictions, cfg.paths.output_dir)
    plot_boxplots(all_results, cfg.paths.output_dir)
    print("\n[OK] TAFNet v4 Comprehensive Evaluation Complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Stage 1: Run the 5-step ADNI preprocessing pipeline on raw NIfTI scans.

Pipeline:
    Brain extraction (ANTsPyNet) -> SyNRA registration to MNI152 ->
    intensity normalisation -> Gaussian denoising -> 128^3 centre-crop

Input  : config.paths.raw_dir         (.nii / .nii.gz files, any nested layout)
Output : config.paths.preprocessed_dir ({subject}_{image_id}.nii.gz, flat)

Example
-------
    python scripts/01_preprocess.py --config configs/preprocessing.yaml
    python scripts/01_preprocess.py --config configs/preprocessing.yaml --dry-run

If your DICOM->NIfTI step landed files at paths.nifti_dir rather than
paths.raw_dir, override:

    python scripts/01_preprocess.py \\
        --config configs/preprocessing.yaml \\
        --override paths.raw_dir=/mnt/drive/MyDrive/ADNI_nifti
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tafnet.config import load_config
from tafnet.preprocessing import run_pipeline
from tafnet.utils.drive import verify_drive_mount


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true",
                        help="List files only without processing.")
    parser.add_argument("--no-drive-check", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    if args.dry_run:
        cfg.execution.dry_run = True

    if not args.no_drive_check:
        verify_drive_mount(Path(cfg.paths.raw_dir).parents[0])

    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

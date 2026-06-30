#!/usr/bin/env python3
"""
Stage 0: Convert raw ADNI DICOM directories to NIfTI (.nii.gz).

Skip this script entirely if your Drive already has the data as .nii or
.nii.gz — go straight to scripts/01_preprocess.py.

Example
-------
    python scripts/00_dicom_to_nifti.py \\
        --config configs/preprocessing.yaml

To override paths on the CLI:

    python scripts/00_dicom_to_nifti.py \\
        --config configs/preprocessing.yaml \\
        --override paths.raw_dir=/mnt/drive/MyDrive/ADNI_DICOM \\
        --override paths.nifti_dir=/mnt/drive/MyDrive/ADNI_nifti
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tafnet.config import load_config
from tafnet.preprocessing import convert_all
from tafnet.utils.drive import verify_drive_mount


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True,
                        help="Path to preprocessing YAML config.")
    parser.add_argument("--override", action="append", default=[],
                        help="Override key.path=value (repeatable).")
    parser.add_argument("--no-drive-check", action="store_true",
                        help="Skip the drive mount sanity check.")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    if not args.no_drive_check:
        # raw_dir might be on Drive; verify ancestor mount exists
        verify_drive_mount(Path(cfg.paths.raw_dir).parents[0])

    summary = convert_all(
        raw_root=cfg.paths.raw_dir,
        out_root=cfg.paths.nifti_dir,
        preserve_subject_structure=True,
    )
    print("\nSummary:", summary["converted"], "converted,",
          summary["skipped"], "skipped,",
          summary["errors"], "errors")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

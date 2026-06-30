"""
DICOM -> NIfTI conversion for ADNI exports.

ADNI typically delivers MRI scans as DICOM series in a nested folder layout:

    <raw_root>/<subject>/<modality_folder>/<date>/<series_uid>/<*.dcm>

This module walks the tree, groups files by series directory, and runs
`dcm2niix` (a binary, faster and more accurate than pure-Python options) on
each series to emit a single .nii.gz alongside its sidecar JSON.

The output directory mirrors the structure expected by the preprocessing
pipeline driver (raw_dir layout with .nii.gz instead of .dcm).

Requires dcm2niix on PATH:
    sudo apt-get install -y dcm2niix    # Debian/Ubuntu (GCP default)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from glob import glob
from pathlib import Path
from typing import List


def _have_dcm2niix() -> str:
    """Return the dcm2niix binary path or raise with install instructions."""
    bin_path = shutil.which("dcm2niix")
    if bin_path is None:
        sys.stderr.write(
            "\n[ERROR] dcm2niix not found on PATH.\n"
            "Install on Ubuntu/Debian (default for GCP Compute Engine):\n"
            "    sudo apt-get update && sudo apt-get install -y dcm2niix\n"
        )
        raise FileNotFoundError("dcm2niix not installed")
    return bin_path


def find_dicom_series_dirs(raw_root: str) -> List[str]:
    """
    Return every directory that contains DICOM (.dcm) files anywhere under
    raw_root. Each such directory is treated as one series.
    """
    series_dirs = set()
    for root, _dirs, files in os.walk(raw_root):
        for f in files:
            lower = f.lower()
            if lower.endswith(".dcm") or lower.endswith(".ima"):
                series_dirs.add(root)
                break
    return sorted(series_dirs)


def convert_series(series_dir: str, out_dir: str, dcm2niix: str | None = None,
                   filename_template: str = "%i_%s") -> List[str]:
    """
    Convert one DICOM series directory to NIfTI in out_dir using dcm2niix.

    `filename_template` follows dcm2niix conventions:
        %i = patient ID, %s = series number, %p = protocol, %t = time

    Returns the list of .nii.gz files produced.
    """
    dcm2niix = dcm2niix or _have_dcm2niix()
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        dcm2niix,
        "-z", "y",                       # gzip output (.nii.gz)
        "-b", "y",                       # write JSON sidecar
        "-f", filename_template,
        "-o", out_dir,
        series_dir,
    ]
    completed = subprocess.run(
        cmd, capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        sys.stderr.write(
            f"[dcm2niix] non-zero exit ({completed.returncode}) for {series_dir}:\n"
            f"  stderr: {completed.stderr.strip()}\n"
        )
        return []
    return sorted(glob(os.path.join(out_dir, "*.nii.gz")))


def convert_all(raw_root: str, out_root: str,
                preserve_subject_structure: bool = True) -> dict:
    """
    Convert every DICOM series under raw_root.

    If preserve_subject_structure is True (recommended for ADNI), each subject
    gets its own output sub-directory under out_root so that downstream code
    can still extract the subject ID from the file path.

    Returns a summary dict with counts and a per-series list.
    """
    dcm2niix = _have_dcm2niix()
    os.makedirs(out_root, exist_ok=True)

    raw_root = os.path.normpath(raw_root)
    raw_root_basename = os.path.basename(raw_root)

    print(f"Scanning {raw_root} for DICOM series...")
    series_dirs = find_dicom_series_dirs(raw_root)
    print(f"Found {len(series_dirs)} series directories\n")

    results = []
    t_start = time.time()

    for i, sd in enumerate(series_dirs, 1):
        # Pick an output sub-folder per series. Default: mirror first 2 path
        # levels under raw_root (typically <subject>/<modality_folder>).
        rel = os.path.relpath(sd, raw_root)
        parts = rel.split(os.sep)
        if preserve_subject_structure and len(parts) >= 2:
            sub_out = os.path.join(out_root, parts[0], parts[1])
        else:
            sub_out = os.path.join(out_root, rel)
        os.makedirs(sub_out, exist_ok=True)

        # Skip if any nii.gz already exists in target (cheap auto-resume)
        if glob(os.path.join(sub_out, "*.nii.gz")):
            print(f"[{i}/{len(series_dirs)}] SKIP (exists): {rel}")
            results.append({"series": sd, "status": "skipped",
                            "files": glob(os.path.join(sub_out, "*.nii.gz"))})
            continue

        print(f"[{i}/{len(series_dirs)}] CONVERT: {rel}")
        files = convert_series(sd, sub_out, dcm2niix=dcm2niix)
        results.append({"series": sd, "status": "ok" if files else "error",
                        "files": files})

    elapsed = time.time() - t_start
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_err = sum(1 for r in results if r["status"] == "error")
    print(f"\nDone in {elapsed/60:.1f} min  "
          f"(converted={n_ok}, skipped={n_skip}, errors={n_err})")
    return {"converted": n_ok, "skipped": n_skip, "errors": n_err,
            "results": results}

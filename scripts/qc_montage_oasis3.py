#!/usr/bin/env python3
"""
qc_montage_oasis3.py
====================
Visual + numeric quality control for preprocessed OASIS-3 volumes before
zero-shot inference. The one failure mode the numeric pipeline can't catch is a
silent orientation flip; this renders mid-slices so you can confirm by eye that
the OASIS-3 volumes are the same brain-extracted, MNI-aligned 128^3 object as
your ADNI/OASIS-2 data.

For each volume it shows axial / coronal / sagittal central slices and prints
shape, intensity range, and the non-zero (brain) voxel fraction. Pass a
--reference (an ADNI or OASIS-2 preprocessed volume) to put a known-good row at
the top for direct comparison.

Usage
-----
    python qc_montage_oasis3.py \
        --dir ${TAFNET_DATA}/oasis3_preprocessed \
        --n 4 \
        --reference ${TAFNET_DATA}/oasis2_preprocessed/<some_pp>.nii.gz \
        --out oasis3_qc_montage.png

If --reference is omitted, only OASIS-3 volumes are shown.
Dependencies: nibabel, numpy, matplotlib.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write a PNG, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib            # noqa: E402
import numpy as np              # noqa: E402


def load_vol(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata(dtype=np.float32))


def mid_slices(vol: np.ndarray):
    """Central axial, coronal, sagittal slices (rotated for radiological view)."""
    x, y, z = (d // 2 for d in vol.shape[:3])
    axial = np.rot90(vol[:, :, z])
    coronal = np.rot90(vol[:, y, :])
    sagittal = np.rot90(vol[x, :, :])
    return axial, coronal, sagittal


def stats_line(name: str, vol: np.ndarray) -> str:
    nz = float((vol > 0).mean())
    return (f"{name:<42} shape {vol.shape}  "
            f"range [{vol.min():.3f}, {vol.max():.3f}]  brain {nz*100:4.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="QC montage for preprocessed OASIS-3 volumes.")
    ap.add_argument("--dir", default="${TAFNET_DATA}/oasis3_preprocessed")
    ap.add_argument("--n", type=int, default=4, help="Number of OASIS-3 volumes to show.")
    ap.add_argument("--reference", default=None,
                    help="Optional ADNI/OASIS-2 preprocessed volume for the top row.")
    ap.add_argument("--out", default="oasis3_qc_montage.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pp_dir = os.path.expanduser(os.path.expandvars(args.dir))
    files = sorted(glob.glob(os.path.join(pp_dir, "*_pp.nii.gz")))
    if not files:
        raise SystemExit(f"No *_pp.nii.gz volumes found in {pp_dir}")

    random.seed(args.seed)
    chosen = random.sample(files, min(args.n, len(files)))

    rows = []
    if args.reference:
        ref = os.path.expanduser(os.path.expandvars(args.reference))
        if os.path.exists(ref):
            rows.append(("REF: " + os.path.basename(ref), ref))
        else:
            print(f"[warn] reference not found, skipping: {ref}")
    rows += [(os.path.basename(p), p) for p in chosen]

    print("=" * 78)
    print("QC SUMMARY")
    print("=" * 78)
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    if len(rows) == 1:
        axes = axes[np.newaxis, :]

    for r, (name, path) in enumerate(rows):
        vol = load_vol(path)
        print(stats_line(name, vol))
        for c, (img, view) in enumerate(zip(mid_slices(vol),
                                            ("axial", "coronal", "sagittal"))):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(view)
            if c == 0:
                short = name if len(name) <= 24 else name[:21] + "..."
                ax.set_ylabel(short, fontsize=8)

    print("=" * 78)
    print("Check by eye: skull fully removed, brain centred, the reference row and "
          "the OASIS-3 rows share the SAME orientation (nose/posterior, "
          "left/right, top/bottom). A flipped axis between rows = orientation bug.")

    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\n[written] {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

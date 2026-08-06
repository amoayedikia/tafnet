#!/usr/bin/env python3
"""
preprocess_oasis3.py
====================
Run the IDENTICAL TAFNet preprocessing pipeline over the geometry-filtered
OASIS-3 volumes, then emit a model-ready preprocessed-pairs CSV for zero-shot
inference.

This script IMPORTS and CALLS the existing step functions from
src/tafnet/preprocessing -- it does not reimplement any step. OASIS-3 therefore
lands in the same MNI152 / 128^3 feature space as the ADNI training data and the
OASIS-2 validation data *by construction*:

    step1 brain extraction (ANTsPyNet "t1")
    step2 spatial normalisation (ANTs SyNRA -> ants.get_ants_data("mni"))
    step3 intensity normalisation (min-max, brain voxels -> [0,1])
    step4 Gaussian denoising (sigma = 0.5)
    step5 centre-crop / pad to 128^3, saved with identity affine

Place this file in the repo's `scripts/` directory and run it ON THE MAC, where
ANTs / ANTsPyNet are installed and the brain-extraction model weights are already
cached from the OASIS-2 run. The Toshiba drive must be mounted so the source
paths in the input CSV resolve.

Inputs (from prefilter_oasis3_pairs.py)
---------------------------------------
    --vols   oasis3_unique_volumes.csv        columns: src_path, out_path (the
                                              authoritative work list + output
                                              locations chosen at prefilter time)
    --pairs  oasis3_pairs_ready_filtered.csv  surviving labelled pairs

Outputs
-------
    <out_path per row>                  preprocessed 128^3 NIfTI volumes
    oasis3_preprocessing_log.csv        per-volume status, APPENDED after every
                                        scan (crash-safe progress record)
    oasis3_pairs_preprocessed.csv       the pair table with baseline/followup
                                        paths rewritten to the preprocessed
                                        volumes, ready for the zero-shot script

Crash-resilience
----------------
    * skips any output that already exists and is non-empty (auto-resume),
    * wraps every volume in try/except so one bad scan never aborts the run,
    * appends a log row after EACH volume and flushes, so a kill/resume loses at
      most the in-flight scan.
Safe to launch inside a tmux session for long unattended runs.

Usage
-----
    # from the repo root, inside tmux:
    python scripts/preprocess_oasis3.py \
        --vols  oasis3_unique_volumes.csv \
        --pairs oasis3_pairs_ready_filtered.csv

Dependencies: numpy, nibabel, pandas, ants (antspyx), antspynet, scipy.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# Make the repo's `src/` importable exactly as the other scripts do.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tafnet.preprocessing.pipeline import load_mni_template          # noqa: E402
from tafnet.preprocessing.steps import (                             # noqa: E402
    estimate_noise_jdac,
    step1_brain_extraction,
    step2_spatial_normalization,
    step3_intensity_normalization,
    step4_denoising,
    step5_resample_volume,
)


def _resolve(p):
    """Expand ${ENV_VAR} and ~ in a user-supplied path, returning a Path."""
    import os
    from pathlib import Path as _P
    return _P(os.path.expandvars(str(p))).expanduser()


# ---- pipeline parameters (must match configs/preprocessing.yaml) ------------
REGISTRATION_TYPE = "SyNRA"
NORMALIZATION_METHOD = "minmax"
APPLY_DENOISING = True
GAUSSIAN_SIGMA = 0.5
TARGET_SIZE = (128, 128, 128)
N_THREADS = 4

LOG_FIELDS = [
    "timestamp", "src_path", "out_path", "status", "error",
    "total_time_s", "noise_before", "noise_after", "final_shape",
]


def preprocess_one(src_path: str, out_path: str, template) -> dict:
    """Run all five steps on one volume. Returns a status dict. Never raises."""
    import tempfile

    stats = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "src_path": src_path,
        "out_path": out_path,
        "status": "fail",
        "error": "",
        "total_time_s": None,
        "noise_before": None,
        "noise_after": None,
        "final_shape": None,
    }
    t_start = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            brain_path = step1_brain_extraction(
                src_path, os.path.join(tmpdir, "brain.nii.gz")
            )
            mni_image = step2_spatial_normalization(
                brain_path, template, reg_type=REGISTRATION_TYPE
            )
            data = mni_image.numpy()
            data = step3_intensity_normalization(data, method=NORMALIZATION_METHOD)
            stats["noise_before"] = round(estimate_noise_jdac(data), 5)
            if APPLY_DENOISING:
                data = step4_denoising(data, sigma=GAUSSIAN_SIGMA)
                stats["noise_after"] = round(estimate_noise_jdac(data), 5)
            else:
                stats["noise_after"] = stats["noise_before"]
            data = step5_resample_volume(data, target_size=TARGET_SIZE)

            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            # Identity affine + float32, exactly as the core pipeline saves.
            nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), out_path)

            stats["status"] = "ok"
            stats["final_shape"] = "x".join(str(s) for s in data.shape)
    except Exception as exc:  # noqa: BLE001 — log and continue on any failure
        stats["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    stats["total_time_s"] = round(time.time() - t_start, 1)
    return stats


def append_log(log_path: str, row: dict) -> None:
    """Append a single status row, writing the header if the file is new."""
    new = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})
        f.flush()


def already_done(out_path: str) -> bool:
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def build_preprocessed_pairs(pairs_csv: str, src_to_out: dict, out_csv: str) -> None:
    """Rewrite the pair table's baseline/followup paths to the preprocessed
    volumes, keeping a pair only if BOTH preprocessed files exist on disk."""
    pairs = pd.read_csv(pairs_csv)
    keep_rows, dropped = [], 0
    for _, row in pairs.iterrows():
        b_src = os.path.expanduser(str(row["baseline_path"]))
        f_src = os.path.expanduser(str(row["followup_path"]))
        b_out, f_out = src_to_out.get(b_src), src_to_out.get(f_src)
        if not b_out or not f_out or not already_done(b_out) or not already_done(f_out):
            dropped += 1
            continue
        r = row.to_dict()
        r["baseline_src"], r["followup_src"] = b_src, f_src
        r["baseline_path"], r["followup_path"] = b_out, f_out
        keep_rows.append(r)

    out = pd.DataFrame(keep_rows)
    out.to_csv(out_csv, index=False)

    n = len(out)
    n_pos = int((out["label"] == 1).sum()) if n and "label" in out else 0
    n_neg = int((out["label"] == 0).sum()) if n and "label" in out else 0
    print(f"\n[pairs] {n} pairs are fully preprocessed and model-ready "
          f"({n_pos} pMCI / {n_neg} sMCI); {dropped} dropped (a volume failed).")
    if n and "label" in out:
        print(f"[pairs] converter subjects: "
              f"{out.loc[out['label'] == 1, 'OASISID'].nunique()}")
    print(f"[written] {out_csv}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocess OASIS-3 volumes for TAFNet.")
    ap.add_argument("--vols", default="oasis3_unique_volumes.csv",
                    help="Unique-volume work list from prefilter (src_path,out_path).")
    ap.add_argument("--pairs", default="oasis3_pairs_ready_filtered.csv",
                    help="Surviving labelled pair table from prefilter.")
    ap.add_argument("--log", default="oasis3_preprocessing_log.csv")
    ap.add_argument("--pairs-out", default="oasis3_pairs_preprocessed.csv")
    ap.add_argument("--no-resume", action="store_true",
                    help="Reprocess even if an output already exists.")
    args = ap.parse_args()

    vols_path = _resolve(args.vols)
    if not vols_path.exists():
        print(f"ERROR: vols file not found: {vols_path}", file=sys.stderr)
        return 1
    vols = pd.read_csv(vols_path)
    for col in ("src_path", "out_path"):
        if col not in vols.columns:
            print(f"ERROR: vols file missing column '{col}'.", file=sys.stderr)
            return 1

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(N_THREADS)

    print("=" * 70)
    print("LOADING MNI152 TEMPLATE  (ants.get_ants_data('mni'))")
    print("=" * 70)
    template = load_mni_template()
    print(f"  Shape:   {template.shape}")
    print(f"  Spacing: {template.spacing}")

    work = [
        (os.path.expanduser(s), os.path.expanduser(o))
        for s, o in zip(vols["src_path"].astype(str), vols["out_path"].astype(str))
    ]
    n_total = len(work)
    print(f"\nVolumes to consider: {n_total}")
    print(f"Pipeline: BE -> SyNRA -> minmax -> denoise(sigma={GAUSSIAN_SIGMA}) "
          f"-> crop {TARGET_SIZE}")
    print(f"Resume (skip existing): {not args.no_resume}")
    print(f"Per-volume log: {args.log}\n")

    src_to_out = dict(work)
    start = datetime.now()
    n_ok = n_skip = n_fail = 0

    for i, (src, out) in enumerate(work, 1):
        tag = os.path.basename(out)
        if not args.no_resume and already_done(out):
            n_skip += 1
            print(f"[{i}/{n_total}] SKIP (exists) {tag}")
            append_log(args.log, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "src_path": src, "out_path": out, "status": "skip",
            })
            continue

        if not os.path.exists(src):
            n_fail += 1
            print(f"[{i}/{n_total}] FAIL (source missing) {tag}")
            append_log(args.log, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "src_path": src, "out_path": out, "status": "fail",
                "error": "source file not found",
            })
            continue

        print(f"[{i}/{n_total}] {tag}")
        stats = preprocess_one(src, out, template)
        append_log(args.log, stats)
        if stats["status"] == "ok":
            n_ok += 1
            print(f"           [OK] {stats['total_time_s']}s  shape {stats['final_shape']}"
                  f"  noise {stats['noise_before']} -> {stats['noise_after']}")
        else:
            n_fail += 1
            print(f"           [X] {stats['error']}")

        if i % 10 == 0 or i == n_total:
            elapsed = (datetime.now() - start).total_seconds()
            done_now = n_ok + n_fail
            if done_now:
                rate = elapsed / done_now
                remaining = rate * (n_total - i)
                print(f"  -- {i}/{n_total} | ~{remaining/60:.0f} min remaining "
                      f"(ok {n_ok}, skip {n_skip}, fail {n_fail}) --")

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"  ok   : {n_ok}")
    print(f"  skip : {n_skip}")
    print(f"  fail : {n_fail}")
    print(f"  total time: {(datetime.now() - start).total_seconds()/60:.1f} min")

    # ---- assemble the model-ready preprocessed-pairs table ------------------
    pairs_path = _resolve(args.pairs)
    if pairs_path.exists():
        build_preprocessed_pairs(str(pairs_path), src_to_out, args.pairs_out)
    else:
        print(f"\n[warn] pairs file not found ({pairs_path}); "
              f"skipped writing {args.pairs_out}.")

    if n_fail:
        print(f"\n[note] {n_fail} volume(s) failed — inspect status=fail rows in "
              f"{args.log}. Any pair using a failed volume was dropped from "
              f"{args.pairs_out}; re-running resumes only the failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

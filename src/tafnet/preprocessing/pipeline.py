"""
End-to-end preprocessing pipeline driver.

The five stages are defined in steps.py; this module wires them together
into a single-scan and full-cohort driver with progress reporting,
auto-resume (skip already-done outputs), and a JSON report.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np

from .steps import (
    estimate_noise_jdac,
    extract_image_id_from_filename,
    get_subject_id_from_adni_path,
    step1_brain_extraction,
    step2_spatial_normalization,
    step3_intensity_normalization,
    step4_denoising,
    step5_resample_volume,
)


def load_mni_template():
    """Load the bundled ANTs MNI152 template. Side-effect: sets thread env."""
    import ants

    template_path = ants.get_ants_data("mni")
    return ants.image_read(template_path)


def find_input_scans(input_dir: str) -> List[Dict]:
    """
    Recursively scan a directory tree for NIfTI files and return a deduped
    list of {scan_id, subject_id, image_id, filepath} dicts (one per scan).
    """
    all_files = []
    for root, _dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".nii.gz") or f.endswith(".nii"):
                all_files.append(os.path.join(root, f))
    all_files = sorted(set(all_files))

    adni_root_basename = os.path.basename(os.path.normpath(input_dir))

    scans = []
    seen_ids = set()
    for filepath in all_files:
        subject_id = get_subject_id_from_adni_path(filepath, adni_root_basename)
        image_id = extract_image_id_from_filename(filepath)
        scan_id = f"{subject_id}_{image_id}"
        if scan_id in seen_ids:
            continue
        seen_ids.add(scan_id)
        scans.append({
            "scan_id": scan_id,
            "subject_id": subject_id,
            "image_id": image_id,
            "filepath": filepath,
        })
    return sorted(scans, key=lambda x: x["scan_id"])


def filter_already_done(scans: List[Dict], output_dir: str
                        ) -> Tuple[List[Dict], List[Dict]]:
    """Partition scans into (already_done, to_process) by output file existence."""
    done, todo = [], []
    for scan in scans:
        out = os.path.join(output_dir, f"{scan['scan_id']}.nii.gz")
        if os.path.exists(out):
            done.append(scan)
        else:
            todo.append(scan)
    return done, todo


def preprocess_single_scan(
    input_path: str,
    output_path: str,
    scan_id: str,
    config,
    mni_template,
) -> Dict:
    """Run all 5 preprocessing steps on one scan. Returns a stats dict."""
    proc = config.processing
    stats: Dict = {
        "scan_id": scan_id,
        "subject_id": get_subject_id_from_adni_path(
            input_path, os.path.basename(os.path.normpath(config.paths.raw_dir))
        ),
        "success": False,
        "error": None,
        "times": {},
        "noise_before": None,
        "noise_after": None,
    }

    try:
        with tempfile.TemporaryDirectory() as tmpdir:

            # ---- Step 1: Brain extraction
            t0 = time.time()
            print("    Step 1/5: Brain extraction (ANTsPyNet)...", end=" ", flush=True)
            brain_path = step1_brain_extraction(
                input_path, os.path.join(tmpdir, "brain.nii.gz"),
            )
            stats["times"]["brain_extraction"] = time.time() - t0
            print(f"done ({stats['times']['brain_extraction']:.1f}s)")

            # ---- Step 2: Spatial normalisation
            t0 = time.time()
            print("    Step 2/5: Spatial normalization (ANTs->MNI)...",
                  end=" ", flush=True)
            mni_image = step2_spatial_normalization(
                brain_path, mni_template, reg_type=proc.registration_type,
            )
            data = mni_image.numpy()
            stats["times"]["spatial_norm"] = time.time() - t0
            print(f"done ({stats['times']['spatial_norm']:.1f}s)")

            # ---- Step 3: Intensity normalisation
            t0 = time.time()
            print("    Step 3/5: Intensity normalization...", end=" ", flush=True)
            data = step3_intensity_normalization(data, method=proc.normalization_method)
            stats["noise_before"] = estimate_noise_jdac(data)
            stats["times"]["intensity_norm"] = time.time() - t0
            print(f"done ({stats['times']['intensity_norm']:.1f}s)")

            # ---- Step 4: Denoising
            if proc.apply_denoising:
                t0 = time.time()
                print(f"    Step 4/5: Denoising (sigma={proc.gaussian_sigma})...",
                      end=" ", flush=True)
                data = step4_denoising(data, sigma=proc.gaussian_sigma)
                stats["noise_after"] = estimate_noise_jdac(data)
                stats["times"]["denoising"] = time.time() - t0
                print(f"done ({stats['times']['denoising']:.1f}s)")
            else:
                print("    Step 4/5: Denoising... skipped")
                stats["noise_after"] = stats["noise_before"]

            # ---- Step 5: Resample
            t0 = time.time()
            target_size = tuple(proc.target_size)
            print(f"    Step 5/5: Resampling to {target_size}...",
                  end=" ", flush=True)
            data = step5_resample_volume(data, target_size=target_size)
            stats["times"]["resampling"] = time.time() - t0
            print(f"done ({stats['times']['resampling']:.1f}s)")

            # ---- Save
            print("    Saving...", end=" ", flush=True)
            nib.save(nib.Nifti1Image(data, np.eye(4)), output_path)
            print(f"done  ->  {os.path.basename(output_path)}")

            stats["success"] = True
            stats["final_shape"] = list(data.shape)
            stats["total_time"] = sum(stats["times"].values())

    except Exception as exc:    # noqa: BLE001
        stats["error"] = str(exc)
        print(f"FAILED: {exc}")
        traceback.print_exc()

    return stats


def run_pipeline(config) -> List[Dict]:
    """
    Full preprocessing driver.

    Walks config.paths.raw_dir, processes every NIfTI not already in
    config.paths.preprocessed_dir, writes a JSON report alongside the outputs.
    """
    paths = config.paths
    execn = config.execution
    proc = config.processing

    os.makedirs(paths.preprocessed_dir, exist_ok=True)
    if paths.save_intermediates:
        os.makedirs(os.path.join(paths.intermediate_dir, "brain_extracted"),
                    exist_ok=True)

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(proc.n_threads)

    print("=" * 70)
    print("LOADING MNI152 TEMPLATE")
    print("=" * 70)
    mni_template = load_mni_template()
    print(f"  Shape:   {mni_template.shape}")
    print(f"  Spacing: {mni_template.spacing}")

    print("\n" + "=" * 70)
    print("SCANNING INPUT DIRECTORY (RECURSIVE)")
    print("=" * 70)
    all_scans = find_input_scans(paths.raw_dir)
    print(f"\nTotal scans found:    {len(all_scans)}")
    print(f"Unique subjects:      "
          f"{len({s['subject_id'] for s in all_scans})}")

    if execn.skip_existing:
        done, todo = filter_already_done(all_scans, paths.preprocessed_dir)
        print(f"\nAlready done (skip):  {len(done)}")
        print(f"Remaining to process: {len(todo)}")
    else:
        todo = all_scans
        print(f"\nTo process:           {len(todo)}")

    if execn.dry_run:
        print("\n[DRY RUN — no processing]")
        for scan in todo[:10]:
            print(f"  Would process: {scan['scan_id']}")
        if len(todo) > 10:
            print(f"  ... and {len(todo) - 10} more")
        return []

    print("\n" + "=" * 70)
    print("RUNNING PREPROCESSING PIPELINE")
    print("=" * 70)
    start_time = datetime.now()
    print(f"\nStarting at {start_time.strftime('%H:%M:%S')}")
    print(f"Processing {len(todo)} scans...\n")

    all_stats: List[Dict] = []
    for i, scan in enumerate(todo):
        scan_id = scan["scan_id"]
        input_path = scan["filepath"]
        output_path = os.path.join(paths.preprocessed_dir, f"{scan_id}.nii.gz")

        print(f"\n[{i+1}/{len(todo)}] {scan_id}")
        print(f"  <- {os.path.relpath(input_path, paths.raw_dir)}")

        stats = preprocess_single_scan(
            input_path=input_path,
            output_path=output_path,
            scan_id=scan_id,
            config=config,
            mni_template=mni_template,
        )
        all_stats.append(stats)

        if stats["success"]:
            print(f"  [OK] {stats['total_time']:.1f}s  "
                  f"| noise {stats['noise_before']:.4f} -> {stats['noise_after']:.4f}")
        else:
            print(f"  [X]  Failed: {stats['error']}")

        if (i + 1) % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            avg = elapsed / (i + 1)
            rem = avg * (len(todo) - i - 1)
            print(f"\n  Progress: {i+1}/{len(todo)} | ~{rem/60:.0f} min remaining\n")

    total_time = (datetime.now() - start_time).total_seconds()
    successful = sum(1 for s in all_stats if s["success"])
    failed = len(all_stats) - successful

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Total time : {total_time/60:.1f} min")
    print(f"  Successful : {successful}/{len(all_stats)}")
    print(f"  Failed     : {failed}")

    _save_report(config, all_stats, total_time, successful, failed)
    _verify_outputs(config)
    return all_stats


def _save_report(config, all_stats, total_time, successful, failed) -> None:
    """Write preprocessing_report.json next to the preprocessed outputs."""
    if not all_stats:
        return
    proc = config.processing
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "raw_dir": config.paths.raw_dir,
            "preprocessed_dir": config.paths.preprocessed_dir,
            "brain_extraction": "ANTsPyNet T1",
            "registration_type": proc.registration_type,
            "normalization_method": proc.normalization_method,
            "apply_denoising": proc.apply_denoising,
            "gaussian_sigma": proc.gaussian_sigma,
            "target_size": list(proc.target_size),
        },
        "summary": {
            "total_scans": len(all_stats),
            "successful": successful,
            "failed": failed,
            "total_time_minutes": total_time / 60,
        },
        "scans": [
            {
                "scan_id": s["scan_id"],
                "subject_id": s["subject_id"],
                "success": s["success"],
                "error": s["error"],
                "noise_before": s["noise_before"],
                "noise_after": s["noise_after"],
                "times": s.get("times", {}),
            }
            for s in all_stats
        ],
    }
    report_path = os.path.join(config.paths.preprocessed_dir,
                               "preprocessing_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    nb = [s["noise_before"] for s in all_stats if s["noise_before"] is not None]
    na = [s["noise_after"] for s in all_stats if s["noise_after"] is not None]
    if nb:
        print("\n  [NOISE STATISTICS]")
        print(f"    Before: {np.mean(nb):.4f} +/- {np.std(nb):.4f}")
        print(f"    After:  {np.mean(na):.4f} +/- {np.std(na):.4f}")
        print("    JDAC reference: 0.037")


def _verify_outputs(config) -> None:
    """Sanity-check a handful of outputs for shape / value-range / brain content."""
    print("\n" + "=" * 70)
    print("VERIFICATION (first 5 outputs)")
    print("=" * 70)
    out_files = sorted(glob(os.path.join(config.paths.preprocessed_dir, "*.nii.gz")))
    if not out_files:
        print("\n[X] No output files yet — pipeline has not completed.")
        return
    target_size = tuple(config.processing.target_size)
    for path in out_files[:5]:
        data = nib.load(path).get_fdata()
        ok_shape = data.shape == target_size
        ok_range = 0.0 <= data.min() and data.max() <= 1.0
        ok_voxels = int(np.sum(data > 0)) > 100_000
        print(f"\n  {os.path.basename(path)}")
        print(f"    Shape:        {data.shape}  "
              f"{'[OK]' if ok_shape else f'[X] expected {target_size}'}")
        print(f"    Value range:  [{data.min():.3f}, {data.max():.3f}]  "
              f"{'[OK]' if ok_range else '[X] expected [0,1]'}")
        print(f"    Brain voxels: {int(np.sum(data > 0)):,}  "
              f"{'[OK]' if ok_voxels else '[X] too few'}")

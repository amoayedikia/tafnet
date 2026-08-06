#!/usr/bin/env python3
"""
prefilter_oasis3_pairs.py
=========================
Geometry pre-filter + unique-volume manifest builder for the OASIS-3 zero-shot
external-validation run of TAFNet.

Runs locally on the Mac. Header-only: it reads NIfTI headers with nibabel and
never loads voxel data, so it is fast and needs no ANTs / GPU. It does NOT
preprocess anything -- it decides *what* will be preprocessed and confirms the
converter count survives the geometry filter, before any heavy ANTs work.

What it does
------------
1. Reads the labelled, model-ready pair table from label_oasis3_pairs.py.
2. Collects the UNIQUE set of T1w volumes referenced across all pairs
   (deduplicating shared volumes -- a volume can be a baseline in one pair and
   a follow-up in another).
3. Inspects each unique volume's header (shape + voxel sizes) once and applies
   the geometry filter:
       - reject if any voxel dimension > MAX_VOXEL_MM   (thick-slice scouts)
       - reject if the smallest axis has < MIN_AXIS_DIM voxels (single-slice /
         very-thin acquisitions)
       - reject if the file is missing or unreadable.
4. A pair survives only if BOTH of its volumes pass. Drops are logged with
   reasons.
5. Writes three CSVs and prints a summary, including the surviving converter
   pair / subject counts and the 6-24 month interval-band count.

Outputs
-------
oasis3_volume_geometry.csv      one row per unique volume: subject, day, src
                                path, shape, zooms, pass flag, reason, and the
                                planned preprocessed output filename.
oasis3_pairs_ready_filtered.csv the input pair table, restricted to pairs whose
                                both volumes pass geometry. Same schema as input.
oasis3_unique_volumes.csv       the unique volumes belonging to surviving pairs,
                                with src_path and out_path -- this is the file the
                                preprocessing script will iterate over.

Usage
-----
    python prefilter_oasis3_pairs.py
    python prefilter_oasis3_pairs.py --pairs oasis3_t1w_pairs_ready.csv \
        --out-dir oasis3_preprocessed
    python prefilter_oasis3_pairs.py --max-voxel-mm 1.5 --min-axis-dim 60

Dependencies: pandas, nibabel.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import nibabel as nib
import pandas as pd


def _resolve(p):
    """Expand ${ENV_VAR} and ~ in a user-supplied path, returning a Path."""
    import os
    from pathlib import Path as _P
    return _P(os.path.expandvars(str(p))).expanduser()


# OASIS-3 filenames look like: sub-OAS30029_ses-d0131_T1w.nii.gz
FNAME_RE = re.compile(r"sub-(OAS3\d{4})_ses-d(\d+)_T1w", re.IGNORECASE)


def parse_subject_day(path: str) -> tuple[str, str]:
    """Extract (subject, day-string-as-written) from an OASIS-3 T1w filename.

    The day string is kept verbatim (e.g. '0131', '893') so the output name
    round-trips uniquely regardless of OASIS-3's inconsistent zero-padding.
    """
    m = FNAME_RE.search(os.path.basename(path))
    if m:
        return m.group(1), m.group(2)
    # Fallback: unrecognised name -> use a sanitised basename so we still get a
    # unique, traceable key rather than crashing.
    stem = os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")
    return "UNKNOWN", re.sub(r"[^A-Za-z0-9]+", "_", stem)


def planned_output_name(src_path: str) -> str:
    """Preprocessed output filename: same basename with _pp before the suffix."""
    base = os.path.basename(src_path)
    if base.endswith(".nii.gz"):
        return base[:-7] + "_pp.nii.gz"
    if base.endswith(".nii"):
        return base[:-4] + "_pp.nii.gz"
    return base + "_pp.nii.gz"


def inspect_volume(path: str, max_voxel_mm: float, min_axis_dim: int) -> dict:
    """Read a NIfTI header and apply the geometry filter. No voxel data loaded."""
    rec: dict = {
        "src_path": path,
        "exists": False,
        "shape": None,
        "zooms": None,
        "min_axis_dim": None,
        "max_voxel_mm": None,
        "pass": False,
        "reason": "",
    }
    p = Path(path)
    if not p.exists():
        rec["reason"] = "file not found"
        return rec
    rec["exists"] = True
    try:
        img = nib.load(str(p))           # lazy: header only, no dataobj access
        hdr = img.header
        shape = tuple(int(s) for s in img.shape[:3])
        zooms = tuple(round(float(z), 4) for z in hdr.get_zooms()[:3])
    except Exception as exc:  # noqa: BLE001 - want to log any read failure
        rec["reason"] = f"unreadable header: {type(exc).__name__}"
        return rec

    rec["shape"] = "x".join(str(s) for s in shape)
    rec["zooms"] = "x".join(f"{z:g}" for z in zooms)
    rec["min_axis_dim"] = min(shape) if shape else 0
    rec["max_voxel_mm"] = max(zooms) if zooms else None

    reasons = []
    if min(shape) < min_axis_dim:
        reasons.append(f"min axis {min(shape)} < {min_axis_dim}")
    if max(zooms) > max_voxel_mm:
        reasons.append(f"voxel {max(zooms):g}mm > {max_voxel_mm}mm")
    if reasons:
        rec["reason"] = "; ".join(reasons)
        rec["pass"] = False
    else:
        rec["reason"] = "ok"
        rec["pass"] = True
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Geometry pre-filter + unique-volume manifest for OASIS-3."
    )
    ap.add_argument("--pairs", default="oasis3_t1w_pairs_ready.csv",
                    help="Model-ready labelled pair table from label_oasis3_pairs.py")
    ap.add_argument("--out-dir", default="oasis3_preprocessed",
                    help="Directory the preprocessed volumes WILL be written to "
                         "(used only to compute out_path; nothing is written there here)")
    ap.add_argument("--max-voxel-mm", type=float, default=1.5,
                    help="Reject a volume if any voxel dimension exceeds this (mm)")
    ap.add_argument("--min-axis-dim", type=int, default=60,
                    help="Reject a volume if its smallest axis has fewer voxels than this")
    ap.add_argument("--geom-out", default="oasis3_volume_geometry.csv")
    ap.add_argument("--pairs-out", default="oasis3_pairs_ready_filtered.csv")
    ap.add_argument("--vols-out", default="oasis3_unique_volumes.csv")
    args = ap.parse_args()

    pairs_path = _resolve(args.pairs)
    if not pairs_path.exists():
        print(f"ERROR: pairs file not found: {pairs_path}", file=sys.stderr)
        return 1
    pairs = pd.read_csv(pairs_path)
    for col in ("OASISID", "baseline_path", "followup_path", "label"):
        if col not in pairs.columns:
            print(f"ERROR: pairs file missing required column '{col}'.",
                  file=sys.stderr)
            return 1

    n_pairs_in = len(pairs)
    print(f"Loaded {n_pairs_in} model-ready pairs from {pairs_path}")

    # ---- inspect each UNIQUE volume exactly once ----------------------------
    unique_src = pd.unique(
        pd.concat([pairs["baseline_path"], pairs["followup_path"]], ignore_index=True)
    )
    print(f"Inspecting {len(unique_src)} unique volumes "
          f"(vs {2 * n_pairs_in} path-slots before dedup)...")

    records = []
    for src in unique_src:
        rec = inspect_volume(str(src), args.max_voxel_mm, args.min_axis_dim)
        subj, day = parse_subject_day(str(src))
        rec["OASISID"] = subj
        rec["day"] = day
        rec["out_name"] = planned_output_name(str(src))
        rec["out_path"] = str(Path(args.out_dir) / rec["out_name"])
        records.append(rec)

    geom = pd.DataFrame.from_records(records)
    geom = geom[[
        "OASISID", "day", "shape", "zooms", "min_axis_dim", "max_voxel_mm",
        "exists", "pass", "reason", "src_path", "out_name", "out_path",
    ]]
    geom.to_csv(args.geom_out, index=False)

    pass_map = dict(zip(geom["src_path"], geom["pass"]))

    # ---- decide pair survival: BOTH volumes must pass -----------------------
    def pair_ok(row) -> bool:
        return bool(pass_map.get(str(row["baseline_path"]), False)
                    and pass_map.get(str(row["followup_path"]), False))

    pairs = pairs.copy()
    pairs["geometry_ok"] = pairs.apply(pair_ok, axis=1)
    surviving = pairs[pairs["geometry_ok"]].drop(columns=["geometry_ok"]).copy()
    dropped = pairs[~pairs["geometry_ok"]].copy()
    surviving.to_csv(args.pairs_out, index=False)

    # ---- unique volumes belonging to surviving pairs only -------------------
    keep_src = pd.unique(
        pd.concat([surviving["baseline_path"], surviving["followup_path"]],
                  ignore_index=True)
    )
    vols = geom[geom["src_path"].isin(keep_src)][
        ["OASISID", "day", "src_path", "out_name", "out_path", "shape", "zooms"]
    ].reset_index(drop=True)
    vols.to_csv(args.vols_out, index=False)

    # ---- summary ------------------------------------------------------------
    n_fail = int((~geom["pass"]).sum())
    print("\n" + "=" * 64)
    print("GEOMETRY FILTER SUMMARY")
    print("=" * 64)
    print(f"Unique volumes inspected : {len(geom)}")
    print(f"  passed                 : {int(geom['pass'].sum())}")
    print(f"  failed                 : {n_fail}")
    if n_fail:
        print("  failure reasons:")
        for reason, n in geom.loc[~geom["pass"], "reason"].value_counts().items():
            print(f"    - {reason}: {n}")

    print("-" * 64)
    print(f"Pairs in                 : {n_pairs_in}")
    print(f"Pairs surviving          : {len(surviving)}")
    print(f"Pairs dropped (geometry) : {len(dropped)}")
    if len(dropped):
        d_pos = int((dropped["label"] == 1).sum())
        d_neg = int((dropped["label"] == 0).sum())
        print(f"    of which converters (pMCI) : {d_pos}")
        print(f"    of which stable    (sMCI)  : {d_neg}")

    print("-" * 64)
    s_pos = surviving[surviving["label"] == 1]
    s_neg = surviving[surviving["label"] == 0]
    print(f"Surviving converter pairs (pMCI) : {len(s_pos)} "
          f"from {s_pos['OASISID'].nunique()} subjects")
    print(f"Surviving stable    pairs (sMCI) : {len(s_neg)} "
          f"from {s_neg['OASISID'].nunique()} subjects")
    print(f"Surviving unique volumes to preprocess : {len(vols)}")

    if "interval_months" in surviving.columns:
        band = surviving[(surviving["interval_months"] >= 6)
                         & (surviving["interval_months"] <= 24)]
        print(f"Surviving pairs in 6-24mo trained band : {len(band)} "
              f"({len(band[band['label'] == 1])} pMCI / "
              f"{len(band[band['label'] == 0])} sMCI)")

    print("=" * 64)
    print(f"\n[written] {args.geom_out}   (all unique volumes + geometry verdict)")
    print(f"[written] {args.pairs_out}   ({len(surviving)} surviving pairs)")
    print(f"[written] {args.vols_out}   ({len(vols)} volumes to preprocess)")
    if n_fail:
        print("\nReview the failed volumes in the geometry CSV before preprocessing; "
              "loosen --max-voxel-mm / --min-axis-dim only if a legitimate volume "
              "was caught.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

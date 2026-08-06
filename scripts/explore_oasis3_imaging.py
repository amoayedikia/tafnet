#!/usr/bin/env python3
"""
explore_oasis3_imaging.py
=========================
Explore the OASIS-3 NIfTI imaging tree to support longitudinal TAFNet validation.

Expected layout (BIDS-style, as distributed by OASIS-3):

    <root>/
        OAS30001_MR_d0129/
            anat2/NIFTI/sub-OAS30001_sess-d0129_T1w.nii.gz
            anat3/NIFTI/sub-OAS30001_sess-d0129_T2w.nii.gz
        OAS30001_MR_d3746/
            anat4/NIFTI/sub-OAS30001_sess-d3746_T1w.nii.gz
        OAS30002_MR_d0371/
            ...

Session folders are named OAS3xxxx_MR_dYYYY, where dYYYY is the number of days
from that subject's baseline. The same day offset is repeated in each filename
(sess-dYYYY), so subject + day gives the longitudinal axis directly.

What this script produces
-------------------------
Console report:
  - file inventory and scan-type breakdown
  - image geometry summary (shapes, voxel sizes, orientation, dtype)
  - per-subject longitudinal summary for the chosen scan type (default T1w):
    number of timepoints, total span in days/years, inter-scan intervals
  - pairing summary: how many subjects yield >=2 timepoints and how many
    consecutive pairs fall inside a configurable interval window
    (default 180-730 days, i.e. ~6-24 months, matching the ADNI training regime)

CSV outputs (written to --out-dir):
  - oasis3_scan_inventory.csv      one row per .nii.gz (subject, day, type, geometry, path)
  - oasis3_subject_longitudinal.csv one row per subject (timepoint counts, span, intervals)
  - oasis3_t1w_pairs.csv           one row per consecutive same-type pair (for building pairs)

Usage
-----
    python explore_oasis3_imaging.py
    python explore_oasis3_imaging.py --root "${OASIS3_ROOT}"
    python explore_oasis3_imaging.py --scan-type T1w --min-interval 180 --max-interval 730
    python explore_oasis3_imaging.py --no-headers          # fastest: skip geometry, names only
    python explore_oasis3_imaging.py --plot                # also write an interval histogram

Dependencies: pandas, numpy, nibabel. matplotlib only needed for --plot.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve(p):
    """Expand ${ENV_VAR} and ~ in a user-supplied path, returning a Path."""
    import os
    from pathlib import Path as _P
    return _P(os.path.expandvars(str(p))).expanduser()


try:
    import nibabel as nib
    HAVE_NIBABEL = True
except ImportError:
    HAVE_NIBABEL = False

# sub-OAS30001_sess-d3746_..._T1w.nii.gz  -> subject + day-offset
SUBJECT_DAY_RE = re.compile(r"sub-(OAS3\d{4})_(?:ses|sess)-d(\d+)", re.IGNORECASE)
# Optional BIDS entities we like to keep track of.
RUN_RE = re.compile(r"_run-(\w+?)(?:_|\.)", re.IGNORECASE)
ECHO_RE = re.compile(r"_echo-(\w+?)(?:_|\.)", re.IGNORECASE)
# Session folder name OAS30001_MR_d3746
SESSION_FOLDER_RE = re.compile(r"OAS3\d{4}_MR_d\d+", re.IGNORECASE)

DAYS_PER_MONTH = 30.4375
DAYS_PER_YEAR = 365.25


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def parse_scan_type(filename: str) -> str:
    """The BIDS suffix is the final token before the extension (T1w, T2w, FLAIR...)."""
    stem = filename
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem.split("_")[-1]


def find_named_parent(path: Path, pattern: re.Pattern) -> str | None:
    for parent in path.parents:
        if pattern.fullmatch(parent.name):
            return parent.name
    return None


def find_anat_folder(path: Path) -> str | None:
    for parent in path.parents:
        if parent.name.lower().startswith("anat"):
            return parent.name
    return None


def read_geometry(path: Path) -> dict:
    """Read header-only geometry. Does not load voxel data."""
    if not HAVE_NIBABEL:
        return {"shape": None, "voxel_mm": None, "orientation": None, "dtype": None}
    try:
        img = nib.load(str(path))
        shape = tuple(int(s) for s in img.shape)
        zooms = tuple(round(float(z), 3) for z in img.header.get_zooms()[:3])
        orient = "".join(nib.aff2axcodes(img.affine))
        dtype = str(img.header.get_data_dtype())
        return {"shape": shape, "voxel_mm": zooms, "orientation": orient, "dtype": dtype}
    except Exception as err:  # noqa: BLE001
        return {"shape": None, "voxel_mm": None, "orientation": None,
                "dtype": f"READ_ERROR:{err}"}


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
def build_inventory(root: Path, read_headers: bool, limit: int | None) -> pd.DataFrame:
    files = sorted(root.rglob("*.nii.gz")) + sorted(root.rglob("*.nii"))
    files = [f for f in files if f.is_file()]
    if limit:
        files = files[:limit]

    if not files:
        return pd.DataFrame()

    records = []
    total = len(files)
    for i, path in enumerate(files, 1):
        if i % 500 == 0 or i == total:
            print(f"  ...read {i}/{total} files", file=sys.stderr)

        name = path.name
        m = SUBJECT_DAY_RE.search(name)
        subject = m.group(1).upper() if m else None
        day = int(m.group(2)) if m else None

        run_m = RUN_RE.search(name)
        echo_m = ECHO_RE.search(name)

        rec = {
            "OASISID": subject,
            "session_day": day,
            "scan_type": parse_scan_type(name),
            "run": run_m.group(1) if run_m else None,
            "echo": echo_m.group(1) if echo_m else None,
            "anat_folder": find_anat_folder(path),
            "session_folder": find_named_parent(path, SESSION_FOLDER_RE),
            "file_mb": round(path.stat().st_size / 1024 / 1024, 2),
            "filepath": str(path),
        }
        if read_headers:
            rec.update(read_geometry(path))
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    return df


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def report_overview(df: pd.DataFrame) -> list[str]:
    lines = ["## Inventory overview", ""]
    lines.append(f"- Total NIfTI files: {len(df):,}")
    lines.append(f"- Unique subjects: {df['OASISID'].nunique():,}")
    lines.append(f"- Files with unparsable name (no subject/day): "
                 f"{df['OASISID'].isna().sum():,}")
    lines.append(f"- Total imaging volume on disk: {df['file_mb'].sum() / 1024:,.2f} GB")
    lines.append("")
    lines.append("- Scan-type breakdown (files):")
    for stype, n in df["scan_type"].value_counts().items():
        n_subj = df.loc[df["scan_type"] == stype, "OASISID"].nunique()
        lines.append(f"    - {stype}: {n:,} files across {n_subj:,} subjects")
    lines.append("")
    return lines


def report_geometry(df: pd.DataFrame, scan_type: str) -> list[str]:
    lines = [f"## Image geometry ({scan_type})", ""]
    sub = df[df["scan_type"] == scan_type]
    if "shape" not in df.columns or sub["shape"].isna().all():
        lines.append("- Geometry not available (run without --no-headers, "
                     "and ensure nibabel is installed).")
        lines.append("")
        return lines

    shapes = Counter(str(s) for s in sub["shape"].dropna())
    zooms = Counter(str(z) for z in sub["voxel_mm"].dropna())
    orients = Counter(str(o) for o in sub["orientation"].dropna())
    dtypes = Counter(str(d) for d in sub["dtype"].dropna())

    lines.append(f"- Distinct matrix sizes: {len(shapes)} (top 10):")
    for shape, n in shapes.most_common(10):
        lines.append(f"    - {shape}: {n:,}")
    lines.append(f"- Distinct voxel sizes mm: {len(zooms)} (top 10):")
    for z, n in zooms.most_common(10):
        lines.append(f"    - {z}: {n:,}")
    lines.append(f"- Orientation codes: " +
                 ", ".join(f"{o}={n}" for o, n in orients.most_common()))
    lines.append(f"- Data types: " +
                 ", ".join(f"{d}={n}" for d, n in dtypes.most_common()))
    lines.append("")
    lines.append("  Note: heterogeneous matrix/voxel sizes are expected across scanners; "
                 "your ANTs->MNI152->128^3 pipeline normalises these, so this is for "
                 "QC awareness rather than a blocker.")
    lines.append("")
    return lines


def build_subject_longitudinal(df: pd.DataFrame, scan_type: str) -> pd.DataFrame:
    sub = df[(df["scan_type"] == scan_type) & df["OASISID"].notna()
             & df["session_day"].notna()].copy()
    rows = []
    for sid, grp in sub.groupby("OASISID"):
        days = sorted(grp["session_day"].unique().tolist())
        intervals = [days[i + 1] - days[i] for i in range(len(days) - 1)]
        first, last = days[0], days[-1]
        span = last - first
        rows.append({
            "OASISID": sid,
            "n_timepoints": len(days),               # unique sessions with this scan type
            "n_scans": len(grp),                     # total files (multiple runs per session)
            "first_day": first,
            "last_day": last,
            "span_days": span,
            "span_years": round(span / DAYS_PER_YEAR, 2),
            "median_interval_days": int(np.median(intervals)) if intervals else None,
            "session_days": ";".join(str(d) for d in days),
            "intervals_days": ";".join(str(iv) for iv in intervals),
        })
    out = pd.DataFrame(rows).sort_values(
        ["n_timepoints", "span_days"], ascending=False).reset_index(drop=True)
    return out


def report_longitudinal(long_df: pd.DataFrame, scan_type: str) -> list[str]:
    lines = [f"## Longitudinal summary ({scan_type})", ""]
    if long_df.empty:
        lines.append("- No subjects with parsable sessions for this scan type.")
        lines.append("")
        return lines

    n_total = len(long_df)
    tp_counts = long_df["n_timepoints"].value_counts().sort_index()
    lines.append(f"- Subjects with at least one {scan_type}: {n_total:,}")
    lines.append("- Distribution of timepoints per subject:")
    for tp, n in tp_counts.items():
        lines.append(f"    - {tp} timepoint(s): {n:,} subjects")
    lines.append("")

    multi = long_df[long_df["n_timepoints"] >= 2]
    lines.append(f"- Subjects usable for longitudinal pairing (>=2 {scan_type}): "
                 f"{len(multi):,}")
    if not multi.empty:
        lines.append(f"    - span (years): min {multi['span_years'].min():.2f}, "
                     f"median {multi['span_years'].median():.2f}, "
                     f"max {multi['span_years'].max():.2f}")
        lines.append(f"    - timepoints: median {int(multi['n_timepoints'].median())}, "
                     f"max {int(multi['n_timepoints'].max())}")
    lines.append("")
    return lines


def build_pairs(df: pd.DataFrame, scan_type: str,
                min_interval: int, max_interval: int) -> pd.DataFrame:
    """One row per consecutive same-type session pair, with one volume picked per session."""
    sub = df[(df["scan_type"] == scan_type) & df["OASISID"].notna()
             & df["session_day"].notna()].copy()

    # Pick one representative file per (subject, session_day): deterministic by path.
    sub = sub.sort_values(["OASISID", "session_day", "filepath"])
    one_per_session = sub.groupby(["OASISID", "session_day"], as_index=False).agg(
        filepath=("filepath", "first"),
        n_runs=("filepath", "size"),
    )

    rows = []
    for sid, grp in one_per_session.groupby("OASISID"):
        grp = grp.sort_values("session_day").reset_index(drop=True)
        for i in range(len(grp) - 1):
            d0, d1 = int(grp.loc[i, "session_day"]), int(grp.loc[i + 1, "session_day"])
            interval = d1 - d0
            rows.append({
                "OASISID": sid,
                "day_baseline": d0,
                "day_followup": d1,
                "interval_days": interval,
                "interval_months": round(interval / DAYS_PER_MONTH, 1),
                "in_window": min_interval <= interval <= max_interval,
                "baseline_path": grp.loc[i, "filepath"],
                "followup_path": grp.loc[i + 1, "filepath"],
                "baseline_n_runs": int(grp.loc[i, "n_runs"]),
                "followup_n_runs": int(grp.loc[i + 1, "n_runs"]),
            })
    return pd.DataFrame(rows)


def report_pairs(pairs_df: pd.DataFrame, scan_type: str,
                 min_interval: int, max_interval: int) -> list[str]:
    lines = [f"## Pairing summary ({scan_type}, "
             f"window {min_interval}-{max_interval} days)", ""]
    if pairs_df.empty:
        lines.append("- No consecutive pairs found.")
        lines.append("")
        return lines

    iv = pairs_df["interval_days"]
    lines.append(f"- Total consecutive pairs: {len(pairs_df):,}")
    lines.append(f"- Subjects contributing pairs: {pairs_df['OASISID'].nunique():,}")
    lines.append(f"- Interval (days): min {iv.min()}, median {int(iv.median())}, "
                 f"max {iv.max()}")
    lines.append(f"- Interval (months): min {iv.min()/DAYS_PER_MONTH:.1f}, "
                 f"median {iv.median()/DAYS_PER_MONTH:.1f}, "
                 f"max {iv.max()/DAYS_PER_MONTH:.1f}")
    in_win = pairs_df["in_window"].sum()
    lines.append(f"- Pairs inside {min_interval}-{max_interval} day window: {in_win:,} "
                 f"({100*in_win/len(pairs_df):.1f}%), from "
                 f"{pairs_df.loc[pairs_df['in_window'], 'OASISID'].nunique():,} subjects")
    lines.append("")
    # Coarse interval histogram in text form.
    bins = [0, 180, 365, 545, 730, 1095, 1825, 100000]
    labels = ["<6mo", "6-12mo", "12-18mo", "18-24mo", "24-36mo", "3-5yr", ">5yr"]
    cats = pd.cut(iv, bins=bins, labels=labels, right=True, include_lowest=True)
    lines.append("- Interval distribution:")
    for label in labels:
        n = int((cats == label).sum())
        if n:
            lines.append(f"    - {label}: {n:,} pairs")
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    default_root = (
        "${OASIS3_ROOT}"
    )
    p = argparse.ArgumentParser(description="Explore OASIS-3 NIfTI imaging structure.")
    p.add_argument("--root", default=default_root, help="Imaging root with OAS3..._MR_d... folders")
    p.add_argument("--scan-type", default="T1w", help="Scan suffix to analyse longitudinally")
    p.add_argument("--min-interval", type=int, default=180, help="Min pair interval (days)")
    p.add_argument("--max-interval", type=int, default=730, help="Max pair interval (days)")
    p.add_argument("--no-headers", action="store_true", help="Skip NIfTI header reads (fast)")
    p.add_argument("--limit", type=int, default=None, help="Only scan first N files (debug)")
    p.add_argument("--out-dir", default=".", help="Where to write CSVs")
    p.add_argument("--plot", action="store_true", help="Write interval histogram PNG")
    args = p.parse_args()

    root = _resolve(args.root)
    if not root.exists():
        print(f"ERROR: root does not exist:\n  {root}", file=sys.stderr)
        return 1
    if not HAVE_NIBABEL and not args.no_headers:
        print("WARNING: nibabel not found; geometry will be skipped. "
              "Install it (pip install nibabel) or pass --no-headers.", file=sys.stderr)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building inventory...", file=sys.stderr)
    df = build_inventory(root, read_headers=not args.no_headers, limit=args.limit)
    if df.empty:
        print("No .nii.gz / .nii files found under root.", file=sys.stderr)
        return 1

    long_df = build_subject_longitudinal(df, args.scan_type)
    pairs_df = build_pairs(df, args.scan_type, args.min_interval, args.max_interval)

    out: list[str] = ["# OASIS-3 imaging exploration", "", f"Root: `{root}`", ""]
    out += report_overview(df)
    out += report_geometry(df, args.scan_type)
    out += report_longitudinal(long_df, args.scan_type)
    out += report_pairs(pairs_df, args.scan_type, args.min_interval, args.max_interval)
    print("\n".join(out))

    # Write CSVs.
    inv_path = out_dir / "oasis3_scan_inventory.csv"
    long_path = out_dir / "oasis3_subject_longitudinal.csv"
    pairs_path = out_dir / f"oasis3_{args.scan_type.lower()}_pairs.csv"
    df.to_csv(inv_path, index=False)
    long_df.to_csv(long_path, index=False)
    pairs_df.to_csv(pairs_path, index=False)
    print(f"\n[written] {inv_path}")
    print(f"[written] {long_path}")
    print(f"[written] {pairs_path}")

    if args.plot and not pairs_df.empty:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.hist(pairs_df["interval_days"] / DAYS_PER_MONTH, bins=40,
                    color="#3a6ea5", edgecolor="white")
            ax.axvspan(args.min_interval / DAYS_PER_MONTH,
                       args.max_interval / DAYS_PER_MONTH,
                       color="orange", alpha=0.15, label="target window")
            ax.set_xlabel("Inter-scan interval (months)")
            ax.set_ylabel("Number of consecutive pairs")
            ax.set_title(f"OASIS-3 {args.scan_type} consecutive-pair intervals")
            ax.legend()
            fig.tight_layout()
            png = out_dir / f"oasis3_{args.scan_type.lower()}_intervals.png"
            fig.savefig(png, dpi=150)
            print(f"[written] {png}")
        except ImportError:
            print("matplotlib not installed; skipping --plot.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

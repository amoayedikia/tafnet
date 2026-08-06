#!/usr/bin/env python3
"""
label_oasis3_pairs.py
=====================
Assign MCI-to-AD conversion labels to OASIS-3 consecutive T1w scan pairs,
mirroring the TAFNet task (MCI at baseline -> AD within a horizon).

Inputs
------
  --pairs        consecutive T1w pairs produced by explore_oasis3_imaging.py
                 (needs columns: OASISID, day_baseline, day_followup,
                  baseline_path, followup_path; interval columns optional)
  --cdr-csv      OASIS3_UDSb4_cdr.csv      (CDRTOT/CDRSUM/MMSE, keyed OASISID+days_to_visit)
  --dx-csv       OASIS3_UDSd1_diagnoses.csv (NORMCOG/DEMENTED/PROBAD/alzdis, same key)
  (or --clinical-root to auto-locate both under OASIS3_data_files/)

Labelling logic (defaults are paper-aligned)
--------------------------------------------
  1. Map each pair's baseline MR day to the nearest CDR visit within --tol-days.
     MR session day and clinical days_to_visit share the same per-subject day-0
     clock in OASIS-3, so this matching is valid.
  2. Baseline cognitive state from CDRTOT: 0 -> CN, 0.5 -> MCI, >=1 -> demented.
     Only CDR == --mci-cdr (0.5) baselines are eligible for the MCI->AD task;
     others are written out but flagged ineligible with a reason.
  3. Conversion search over CDR visits after baseline up to the horizon
     (--horizon-mode fixed: baseline + --horizon-months; followup: the follow-up
     scan day). First visit with CDRTOT >= --convert-cdr is the conversion event.
  4. AD etiology: by default require PROBAD==1 or alzdis==1 at the nearest d1
     visit to the conversion event. Conversions to non-AD dementia are EXCLUDED
     (label NA + reason), not mislabelled sMCI. --no-ad-etiology disables this
     and treats any dementia conversion as positive.
  5. Non-converters: labelled sMCI (0). label_confident is True only if the
     subject has clinical follow-up reaching the full horizon; otherwise the
     negative is censored and you may wish to drop it.

Output
------
  --out          full labelled table (every pair, with flags and reasons)
  --out-ready    model-ready subset: eligible MCI baseline, label in {0,1},
                 confident label, baseline clinical gap within tolerance.

Label encoding: label = 1 (pMCI / AD converter), 0 (sMCI / stable), NaN (ineligible/uncertain).

Usage
-----
  python label_oasis3_pairs.py \
      --pairs oasis3_t1w_pairs_m6-m24.csv \
      --clinical-root "${OASIS3_ROOT}/OASIS3_data_files"

Dependencies: pandas, numpy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve(p):
    """Expand ${ENV_VAR} and ~ in a user-supplied path, returning a Path."""
    import os
    from pathlib import Path as _P
    return _P(os.path.expandvars(str(p))).expanduser()


DAYS_PER_MONTH = 30.4375


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def read_csv_robust(path: Path) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as err:  # noqa: BLE001
            last_err = err
    raise RuntimeError(f"Could not read {path}: {last_err}")


def locate_clinical(clinical_root: Path) -> tuple[Path, Path]:
    """Find the CDR (UDSb4) and diagnoses (UDSd1) CSVs under a clinical root."""
    cdr = next(iter(sorted(clinical_root.rglob("*UDSb4*cdr*.csv"))), None)
    dx = next(iter(sorted(clinical_root.rglob("*UDSd1*diagn*.csv"))), None)
    if cdr is None:
        raise FileNotFoundError(f"No UDSb4 CDR csv found under {clinical_root}")
    if dx is None:
        raise FileNotFoundError(f"No UDSd1 diagnoses csv found under {clinical_root}")
    return cdr, dx


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing expected columns: {missing}\n"
                       f"Columns present: {list(df.columns)}")


# --------------------------------------------------------------------------- #
# Matching helpers
# --------------------------------------------------------------------------- #
def nearest_visit(visits: pd.DataFrame, day_col: str, target: float, tol: int):
    """Return (row, gap_days) of the visit nearest `target` within `tol`, else (None, None)."""
    if visits is None or visits.empty:
        return None, None
    diffs = (visits[day_col] - target).abs()
    idx = diffs.idxmin()
    gap = int(diffs.loc[idx])
    if gap <= tol:
        return visits.loc[idx], gap
    return None, None


def classify_cdr(cdr_value: float, mci_cdr: float) -> str:
    if pd.isna(cdr_value):
        return "Unknown"
    if cdr_value == 0.0:
        return "CN"
    if cdr_value == mci_cdr:
        return "MCI"
    if cdr_value >= 1.0:
        return "Dementia"
    return f"Other(CDR={cdr_value})"


def ad_confirmed_at(dx_visits: pd.DataFrame, day: float, tol: int) -> bool | None:
    """True/False if a nearby d1 visit confirms AD etiology; None if no nearby visit."""
    row, _ = nearest_visit(dx_visits, "days_to_visit", day, tol)
    if row is None:
        return None
    probad = row.get("PROBAD", np.nan)
    alzdis = row.get("alzdis", np.nan)
    possad = row.get("POSSAD", np.nan)
    flags = [probad, alzdis, possad]
    return any((not pd.isna(v)) and float(v) == 1.0 for v in flags)


# --------------------------------------------------------------------------- #
# Core labelling
# --------------------------------------------------------------------------- #
def label_pairs(pairs: pd.DataFrame, cdr: pd.DataFrame, dx: pd.DataFrame,
                tol: int, mci_cdr: float, convert_cdr: float,
                horizon_mode: str, horizon_months: int,
                require_ad: bool) -> pd.DataFrame:

    horizon_days = int(round(horizon_months * DAYS_PER_MONTH))
    cdr_by_subj = {sid: g.sort_values("days_to_visit").reset_index(drop=True)
                   for sid, g in cdr.groupby("OASISID")}
    dx_by_subj = {sid: g.sort_values("days_to_visit").reset_index(drop=True)
                  for sid, g in dx.groupby("OASISID")}

    out_rows = []
    for _, pr in pairs.iterrows():
        sid = pr["OASISID"]
        d_base = float(pr["day_baseline"])
        d_follow = float(pr["day_followup"])

        rec = {
            "OASISID": sid,
            "day_baseline": int(d_base),
            "day_followup": int(d_follow),
            "interval_months": round((d_follow - d_base) / DAYS_PER_MONTH, 1),
            "baseline_cdr": np.nan,
            "baseline_mmse": np.nan,
            "baseline_dx": None,
            "baseline_clinical_gap_days": np.nan,
            "eligible_mci_baseline": False,
            "label": np.nan,
            "label_reason": None,
            "converted_dementia": False,
            "conversion_day": np.nan,
            "conversion_cdr": np.nan,
            "ad_etiology_confirmed": None,
            "followup_coverage_days": np.nan,
            "followup_within_horizon": (d_follow - d_base) <= horizon_days,
            "label_confident": False,
            "baseline_path": pr.get("baseline_path"),
            "followup_path": pr.get("followup_path"),
        }

        cdr_sub = cdr_by_subj.get(sid)
        dx_sub = dx_by_subj.get(sid)

        if cdr_sub is None:
            rec["label_reason"] = "no CDR record for subject"
            out_rows.append(rec)
            continue

        # 1. Baseline cognitive state.
        base_visit, gap = nearest_visit(cdr_sub, "days_to_visit", d_base, tol)
        if base_visit is None:
            rec["label_reason"] = f"no CDR visit within {tol}d of baseline"
            out_rows.append(rec)
            continue
        base_cdr = float(base_visit["CDRTOT"]) if not pd.isna(base_visit["CDRTOT"]) else np.nan
        rec["baseline_cdr"] = base_cdr
        rec["baseline_mmse"] = base_visit.get("MMSE", np.nan)
        rec["baseline_dx"] = classify_cdr(base_cdr, mci_cdr)
        rec["baseline_clinical_gap_days"] = gap

        if rec["baseline_dx"] != "MCI":
            rec["label_reason"] = f"baseline not MCI (CDR={base_cdr})"
            out_rows.append(rec)
            continue
        rec["eligible_mci_baseline"] = True

        # 2. Conversion horizon.
        if horizon_mode == "fixed":
            horizon_end = d_base + horizon_days
        else:  # "followup"
            horizon_end = d_follow

        future = cdr_sub[(cdr_sub["days_to_visit"] > d_base)
                         & (cdr_sub["days_to_visit"] <= horizon_end)]
        converters = future[future["CDRTOT"] >= convert_cdr]

        if not converters.empty:
            conv = converters.sort_values("days_to_visit").iloc[0]
            rec["converted_dementia"] = True
            rec["conversion_day"] = int(conv["days_to_visit"])
            rec["conversion_cdr"] = float(conv["CDRTOT"])
            ad_ok = ad_confirmed_at(dx_sub, conv["days_to_visit"], tol)
            rec["ad_etiology_confirmed"] = ad_ok

            if not require_ad:
                rec["label"] = 1.0
                rec["label_reason"] = "converter (any dementia; etiology not required)"
                rec["label_confident"] = True
            elif ad_ok is True:
                rec["label"] = 1.0
                rec["label_reason"] = "AD converter (CDR>=conv + d1 AD etiology)"
                rec["label_confident"] = True
            elif ad_ok is False:
                rec["label"] = np.nan
                rec["label_reason"] = "converted to non-AD dementia (excluded)"
            else:  # ad_ok is None
                rec["label"] = np.nan
                rec["label_reason"] = "dementia conversion, no d1 visit to confirm etiology"
            out_rows.append(rec)
            continue

        # 3. Non-converter: assess follow-up adequacy.
        last_day = float(cdr_sub["days_to_visit"].max())
        coverage = last_day - d_base
        rec["followup_coverage_days"] = int(coverage)
        rec["label"] = 0.0
        if last_day >= horizon_end:
            rec["label_confident"] = True
            rec["label_reason"] = "stable through horizon (sMCI)"
        else:
            rec["label_confident"] = False
            rec["label_reason"] = "stable but follow-up shorter than horizon (censored)"
        out_rows.append(rec)

    return pd.DataFrame(out_rows)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def summarise(labeled: pd.DataFrame) -> None:
    print("\n# Labelling summary\n")
    print(f"- Total pairs processed: {len(labeled):,}")
    print("- Baseline cognitive state (per pair):")
    for dx_state, n in labeled["baseline_dx"].value_counts(dropna=False).items():
        print(f"    - {dx_state}: {n:,}")
    elig = labeled[labeled["eligible_mci_baseline"]]
    print(f"\n- Eligible MCI-baseline pairs: {len(elig):,} "
          f"from {elig['OASISID'].nunique():,} subjects")

    lab = elig.dropna(subset=["label"])
    pos = int((lab["label"] == 1.0).sum())
    neg = int((lab["label"] == 0.0).sum())
    print(f"- Labelled (eligible, non-NA): {len(lab):,}  -> pMCI={pos:,}, sMCI={neg:,}")
    if len(lab):
        print(f"    - class balance pMCI: {100*pos/len(lab):.1f}%")
    conf = lab[lab["label_confident"]]
    print(f"- Confident labels (model-ready): {len(conf):,}  "
          f"(pMCI={int((conf['label']==1).sum()):,}, "
          f"sMCI={int((conf['label']==0).sum()):,})")

    excl = elig[elig["label"].isna()]
    if len(excl):
        print("\n- Eligible-but-excluded reasons:")
        for reason, n in excl["label_reason"].value_counts().items():
            print(f"    - {reason}: {n:,}")
    print("")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description="Label OASIS-3 consecutive T1w pairs for MCI->AD.")
    p.add_argument("--pairs", default="oasis3_t1w_pairs_m6-m24.csv")
    p.add_argument("--clinical-root",
                   default="${OASIS3_ROOT}/OASIS3_data_files")
    p.add_argument("--cdr-csv", default=None, help="Override autodetected UDSb4 CDR csv")
    p.add_argument("--dx-csv", default=None, help="Override autodetected UDSd1 diagnoses csv")
    p.add_argument("--tol-days", type=int, default=365,
                   help="Max gap when matching an MR day to a clinical visit")
    p.add_argument("--mci-cdr", type=float, default=0.5, help="CDRTOT value treated as MCI baseline")
    p.add_argument("--convert-cdr", type=float, default=1.0, help="CDRTOT threshold for conversion")
    p.add_argument("--horizon-mode", choices=["fixed", "followup"], default="fixed")
    p.add_argument("--horizon-months", type=int, default=36)
    p.add_argument("--no-ad-etiology", action="store_true",
                   help="Do not require d1 AD confirmation (any dementia counts as conversion)")
    p.add_argument("--out", default="oasis3_t1w_pairs_labeled.csv")
    p.add_argument("--out-ready", default="oasis3_t1w_pairs_ready.csv")
    args = p.parse_args()

    pairs_path = _resolve(args.pairs)
    if not pairs_path.exists():
        print(f"ERROR: pairs file not found: {pairs_path}", file=sys.stderr)
        return 1
    pairs = read_csv_robust(pairs_path)
    require_columns(pairs, ["OASISID", "day_baseline", "day_followup"], "pairs file")

    if args.cdr_csv and args.dx_csv:
        cdr_path, dx_path = _resolve(args.cdr_csv), _resolve(args.dx_csv)
    else:
        cdr_path, dx_path = locate_clinical(_resolve(args.clinical_root))
    print(f"CDR file:        {cdr_path}", file=sys.stderr)
    print(f"Diagnoses file:  {dx_path}", file=sys.stderr)

    cdr = read_csv_robust(cdr_path)
    dx = read_csv_robust(dx_path)
    require_columns(cdr, ["OASISID", "days_to_visit", "CDRTOT"], "CDR file")
    require_columns(dx, ["OASISID", "days_to_visit"], "diagnoses file")
    for col in ("PROBAD", "alzdis", "POSSAD"):
        if col not in dx.columns:
            print(f"WARNING: diagnoses file has no '{col}' column; "
                  f"AD-etiology confirmation will be weaker.", file=sys.stderr)

    labeled = label_pairs(
        pairs, cdr, dx,
        tol=args.tol_days, mci_cdr=args.mci_cdr, convert_cdr=args.convert_cdr,
        horizon_mode=args.horizon_mode, horizon_months=args.horizon_months,
        require_ad=not args.no_ad_etiology,
    )

    labeled.to_csv(args.out, index=False)
    ready = labeled[
        labeled["eligible_mci_baseline"]
        & labeled["label"].notna()
        & labeled["label_confident"]
    ].copy()
    ready.to_csv(args.out_ready, index=False)

    summarise(labeled)
    print(f"[written] {args.out}  (all pairs with flags)")
    print(f"[written] {args.out_ready}  ({len(ready):,} model-ready labelled pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

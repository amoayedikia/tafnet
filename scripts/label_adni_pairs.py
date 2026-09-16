#!/usr/bin/env python3
"""
Build pMCI / sMCI labels for the ADNI scan pairs from DXSUM.csv.

Mirrors scripts/label_oasis3_pairs.py, which implements what TAF-Net Section 3.1
describes but was never applied to ADNI. Replaces the enrolment-`Group` label in
src/tafnet/data/datasets.py, which is constant per subject and therefore encodes
"enrolled as AD", not conversion.

Procedure
  1. Baseline clinical state: the DXSUM visit nearest the FIRST scan, within
     --tol-days. The pair is eligible only if that visit's DIAGNOSIS is MCI.
  2. Conversion search: the first DXSUM visit after baseline, within
     --horizon-months of the first scan, whose DIAGNOSIS is Dementia.
  3. Converters (label 1) record the conversion date, days-to-conversion, and
     the dementia aetiology recorded at that visit.
  4. Non-converters (label 0) are confident only if clinical follow-up reaches
     the full horizon; otherwise they are censored.
  5. Optional MMSE slope over the horizon, as an external correlate.

Decisions baked in as defaults (override with the flags):
  * Conversions dated BEFORE the follow-up scan are dropped   (--keep-early-dx)
    Those pairs are detection, not prediction.
  * Censored negatives are excluded from the model-ready set  (--keep-censored)
  * ANY dementia counts as a conversion, regardless of aetiology
    (--require-ad-etiology restricts to AD). Aetiology is always RECORDED, so
    the choice can be reversed without re-running.

Outputs
  --out        every pair, with label, status and diagnostics (audit trail)
  --out-ready  the model-ready subset: eligible, confident, not dropped

Label encoding: 1 = pMCI (converter), 0 = sMCI (stable), blank = ineligible.

    python3 scripts/label_adni_pairs.py \
        --pairs pairs_6_24m.csv \
        --dxsum "$ADNI_CLINICAL/DXSUM.csv" \
        --mmse  "$ADNI_CLINICAL/MMSE.csv" \
        --anchor pair \
        --out adni_pairs_labeled_pair.csv --out-ready adni_pairs_ready_pair.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import date, datetime

DAYS_PER_MONTH = 30.4375

# Aetiology of a dementia visit -----------------------------------------------
AD_DXDDUE = "dementia due to alzheimer's disease"
OTHER_DXDDUE = "dementia due to other etiology"


def parse_date(s: str):
    s = (s or "").strip().strip('"')
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def clean(s) -> str:
    s = (s or "").strip().strip('"')
    return "" if s.upper() == "NA" else s


def classify_etiology(visit: dict):
    """Return (etiology, detail) for a Dementia visit: AD / non-AD / unspecified."""
    dxddue = clean(visit.get("DXDDUE")).lower()
    dxodes = clean(visit.get("DXODES"))
    dxothdem = clean(visit.get("DXOTHDEM"))
    dxad = clean(visit.get("DXAD"))
    dxapp = clean(visit.get("DXAPP"))

    # Explicit non-AD signals win — they are the ones that change the label
    # under --require-ad-etiology.
    if dxddue == OTHER_DXDDUE:
        return "non-AD", dxodes or "other etiology"
    if dxothdem.lower() == "yes":
        return "non-AD", dxodes or "other dementia"
    if dxodes and dxodes.lower() not in ("other (specify)",):
        return "non-AD", dxodes

    if dxddue == AD_DXDDUE or dxad.lower() == "yes" or dxapp in ("Probable", "Possible"):
        return "AD", dxapp or "AD"
    if dxodes:
        return "unspecified", dxodes
    return "unspecified", ""


def ols_slope(xs, ys):
    """Least-squares slope of ys on xs. None if fewer than 3 points or no spread."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def load_dxsum(path):
    """subject -> [visit dicts sorted by exam date]."""
    by_subj = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d = parse_date(r.get("EXAMDATE"))
            dx = clean(r.get("DIAGNOSIS"))
            if d is None or not dx:
                continue
            r["_date"], r["_dx"] = d, dx
            by_subj[clean(r.get("PTID"))].append(r)
    for v in by_subj.values():
        v.sort(key=lambda r: r["_date"])
    return by_subj


def load_mmse(path, dxsum_by_subj):
    """subject -> [(date, score)], dated by joining VISCODE2 through DXSUM."""
    if not path or not os.path.exists(path):
        return {}
    viscode_date = {}
    for subj, visits in dxsum_by_subj.items():
        for v in visits:
            vc = clean(v.get("VISCODE2")) or clean(v.get("VISCODE"))
            if vc:
                viscode_date.setdefault((subj, vc), v["_date"])
    out = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            subj = clean(r.get("PTID"))
            vc = clean(r.get("VISCODE2")) or clean(r.get("VISCODE"))
            raw = clean(r.get("MMSCORE"))
            if not (subj and vc and raw):
                continue
            try:
                score = float(raw)
            except ValueError:
                continue
            d = parse_date(r.get("EXAMDATE")) or viscode_date.get((subj, vc))
            if d is not None:
                out[subj].append((d, score))
    for v in out.values():
        v.sort(key=lambda t: t[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs_6_24m.csv")
    ap.add_argument("--dxsum", required=True)
    ap.add_argument("--mmse", default=None)
    ap.add_argument("--tol-days", type=int, default=365,
                    help="max gap between first scan and the baseline clinical visit")
    ap.add_argument("--horizon-months", type=int, default=36)
    ap.add_argument("--mci-baseline", default="MCI")
    ap.add_argument("--convert-dx", default="Dementia")
    ap.add_argument("--anchor", choices=["subject", "pair"], default="subject",
                    help="What 'baseline' means. 'subject': the subject's FIRST "
                         "scan anchors the MCI check and the 36-month horizon, so "
                         "every pair from an MCI-at-entry subject is used (this is "
                         "how the 464-subject cohort was defined). 'pair': each "
                         "pair's own first scan must be MCI and carries its own "
                         "horizon — stricter, and drops later pairs of subjects "
                         "who have already converted.")
    ap.add_argument("--keep-early-dx", action="store_true",
                    help="keep pairs whose diagnosis precedes the follow-up scan "
                         "(default: drop them — detection, not prediction)")
    ap.add_argument("--keep-censored", action="store_true",
                    help="keep negatives whose follow-up is shorter than the horizon "
                         "(default: exclude from the model-ready set)")
    ap.add_argument("--require-ad-etiology", action="store_true",
                    help="count only AD dementia as conversion "
                         "(default: any dementia counts)")
    ap.add_argument("--out", default="adni_pairs_labeled.csv")
    ap.add_argument("--out-ready", default="adni_pairs_ready.csv")
    a = ap.parse_args()

    horizon_days = int(round(a.horizon_months * DAYS_PER_MONTH))
    dxsum = load_dxsum(a.dxsum)
    mmse = load_mmse(a.mmse, dxsum)
    print(f"DXSUM: {len(dxsum):,} subjects with dated diagnoses")
    if mmse:
        print(f"MMSE : {len(mmse):,} subjects with dated scores")

    with open(a.pairs, newline="") as fh:
        pairs = list(csv.DictReader(fh))
    print(f"pairs : {len(pairs):,}   (anchor: {a.anchor})\n")

    # Under --anchor subject, the MCI check and the horizon are both measured
    # from the subject's earliest scan, not from each pair's own first scan.
    subject_anchor = {}
    for p in pairs:
        s, d = clean(p["subject"]), parse_date(p["date1"])
        if d is not None and (s not in subject_anchor or d < subject_anchor[s]):
            subject_anchor[s] = d

    rows = []
    for p in pairs:
        subj = clean(p["subject"])
        d1, d2 = parse_date(p["date1"]), parse_date(p["date2"])
        anchor = subject_anchor.get(subj, d1) if a.anchor == "subject" else d1
        rec = {
            "subject": subj, "scan1": clean(p["scan1"]), "scan2": clean(p["scan2"]),
            "date1": p["date1"], "date2": p["date2"],
            "interval_days": p.get("interval_days", ""),
            "baseline_dx": "", "baseline_visit_date": "", "baseline_gap_days": "",
            "label": "", "status": "",
            "conversion_date": "", "days_to_conversion": "",
            "etiology": "", "etiology_detail": "",
            "last_visit_date": "", "followup_days": "", "label_confident": "",
            "dx_before_scan2": "",
            "mmse_baseline": "", "mmse_slope_per_year": "", "mmse_n": "",
        }
        visits = dxsum.get(subj, [])
        if not visits or d1 is None or anchor is None:
            rec["status"] = "no_clinical_data"
            rows.append(rec)
            continue

        # 1. Baseline visit: nearest to the anchor scan, within tolerance
        base = min(visits, key=lambda v: abs((v["_date"] - anchor).days))
        gap = abs((base["_date"] - anchor).days)
        rec.update(baseline_dx=base["_dx"],
                   baseline_visit_date=base["_date"].isoformat(),
                   baseline_gap_days=gap)
        if gap > a.tol_days:
            rec["status"] = "baseline_visit_out_of_tolerance"
            rows.append(rec)
            continue
        if base["_dx"] != a.mci_baseline:
            rec["status"] = "not_mci_baseline"
            rows.append(rec)
            continue

        # 2. Conversion search, within the horizon measured from the first scan
        conv = None
        for v in visits:
            if v["_date"] <= base["_date"]:
                continue
            if (v["_date"] - anchor).days > horizon_days:
                break
            if v["_dx"] == a.convert_dx:
                etio, detail = classify_etiology(v)
                if a.require_ad_etiology and etio == "non-AD":
                    continue
                conv = (v, etio, detail)
                break

        last_visit = visits[-1]["_date"]
        rec["last_visit_date"] = last_visit.isoformat()
        rec["followup_days"] = (last_visit - anchor).days

        if conv is not None:
            v, etio, detail = conv
            early = d2 is not None and v["_date"] < d2
            rec.update(label=1,
                       conversion_date=v["_date"].isoformat(),
                       days_to_conversion=(v["_date"] - anchor).days,
                       etiology=etio, etiology_detail=detail,
                       label_confident=True,
                       dx_before_scan2=early)
            rec["status"] = "early_dx" if early else "ok"
        else:
            confident = rec["followup_days"] >= horizon_days
            rec.update(label=0, label_confident=confident, dx_before_scan2=False)
            rec["status"] = "ok" if confident else "censored"

        # 3. MMSE over the horizon, as an external correlate
        pts = [((d - anchor).days, s) for d, s in mmse.get(subj, [])
               if 0 <= (d - anchor).days <= horizon_days]
        if pts:
            rec["mmse_baseline"] = pts[0][1]
            rec["mmse_n"] = len(pts)
            slope = ols_slope([x for x, _ in pts], [y for _, y in pts])
            if slope is not None:
                rec["mmse_slope_per_year"] = round(slope * 365.25, 3)
        rows.append(rec)

    # ---- model-ready subset -------------------------------------------------
    def is_ready(r):
        if r["status"] == "ok":
            return True
        if r["status"] == "early_dx" and a.keep_early_dx:
            return True
        if r["status"] == "censored" and a.keep_censored:
            return True
        return False

    ready = [r for r in rows if is_ready(r)]

    for path, data in ((a.out, rows), (a.out_ready, ready)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(data)

    # ---- summary ------------------------------------------------------------
    def subj_counts(data):
        pos = {r["subject"] for r in data if r["label"] == 1}
        neg = {r["subject"] for r in data if r["label"] == 0} - pos
        return pos, neg

    print("--- all pairs, by status ---")
    for k, v in sorted(Counter(r["status"] for r in rows).items()):
        print(f"  {k:<32}: {v:,}")

    print(f"\n--- model-ready set  ({a.out_ready}) ---")
    pos, neg = subj_counts(ready)
    npos = sum(1 for r in ready if r["label"] == 1)
    print(f"  pairs                : {len(ready):,}")
    print(f"  positive pairs       : {npos:,} "
          f"({100*npos/max(len(ready),1):.1f}%)")
    print(f"  subjects             : {len(pos | neg):,}")
    print(f"  converters (pMCI)    : {len(pos):,}")
    print(f"  stable     (sMCI)    : {len(neg):,}")

    etio = Counter(r["etiology"] for r in ready if r["label"] == 1)
    if etio:
        print("\n  converter aetiology  : " +
              ", ".join(f"{k} {v}" for k, v in sorted(etio.items())))
        if not a.require_ad_etiology and etio.get("non-AD"):
            print(f"    NOTE: {etio['non-AD']} non-AD converter pair(s) are counted "
                  f"as conversions.\n"
                  f"    Section 3.1 must say 'MCI-to-dementia', not 'MCI-to-AD'.")

    d2c = [r["days_to_conversion"] for r in ready if r["label"] == 1]
    if d2c:
        d2c.sort()
        print(f"\n  days to conversion   : median {d2c[len(d2c)//2]:,}  "
              f"min {d2c[0]:,}  max {d2c[-1]:,}")

    sl = [r["mmse_slope_per_year"] for r in ready if r["mmse_slope_per_year"] != ""]
    if sl:
        sl.sort()
        print(f"  MMSE slope/yr        : n {len(sl):,}  median {sl[len(sl)//2]:+.2f}"
              f"  p05 {sl[len(sl)//20]:+.2f}  p95 {sl[-len(sl)//20]:+.2f}")

    ivals = [int(r["interval_days"]) for r in ready if str(r["interval_days"]).isdigit()]
    if ivals:
        bins = Counter()
        for d in ivals:
            bins[f"{int(round(d/30.4375)):>2d}m"] += 1
        print("\n  interval distribution: " +
              ", ".join(f"{k} {v}" for k, v in sorted(bins.items(),
                                                      key=lambda t: int(t[0][:-1]))))

    print(f"\n[written] {a.out}  ({len(rows):,} rows)")
    print(f"[written] {a.out_ready}  ({len(ready):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

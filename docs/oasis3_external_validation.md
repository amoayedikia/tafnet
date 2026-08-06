# OASIS-3 external validation

Zero-shot transfer of the ADNI-trained TAFNet checkpoint to OASIS-3, with no
cohort-specific adaptation. OASIS-3 probes transfer at *matched* acquisition
(3T, as in ADNI), so any residual gap is attributable to site and cohort effects
rather than confounded with a change in field strength — the complement to the
OASIS-2 evaluation, which shifts field strength, label derivation and scanner
generation simultaneously.

Zero-shot is also the only well-posed evaluation here: 25 converter subjects are
too few to support a subject-level adaptation split, and the controlled A/B on
OASIS-2 already established that fine-tuning yields no measurable AUC
improvement at this data scale (Δ = +0.008, p = 0.31).

## Why OASIS-3 needs its own chain

OASIS-2 pairs are rebuilt from a CDR-group trajectory by the core dataset class.
OASIS-3 cannot use that path, for two reasons:

1. **The time axis is explicit.** Session folders are named `OAS3xxxx_MR_dYYYY`,
   where `dYYYY` is days from that subject's baseline. There is no visit code to
   normalise — the interval is read directly, and it is irregular by design.
2. **Labels need clinical joins.** Conversion is derived from the UDS b4 (CDR)
   and d1 (diagnosis) forms, matched to each scan by day offset, with AD
   etiology confirmed at the conversion event.

So `04_oasis3_zeroshot.py` consumes explicit pre-labelled pairs rather than
rebuilding them, which preserves the labelling decisions described below.
Tensor preparation reuses the core `_load_nifti_volume`, so the model sees
byte-identical inputs to the ADNI and OASIS-2 runs: a (1, 128, 128, 128) float32
volume in [0, 1].

## Labelling logic

Implemented in `scripts/label_oasis3_pairs.py`. Defaults are paper-aligned.

1. Match each pair's baseline MR day to the nearest CDR visit within
   `--tol-days`. MR session day and clinical `days_to_visit` share the same
   per-subject day-0 clock, which makes this valid.
2. Baseline state from `CDRTOT`: 0 → CN, 0.5 → MCI, ≥1 → demented. Only
   CDR = 0.5 baselines are eligible; others are written out and flagged
   ineligible with a reason.
3. Search CDR visits after baseline up to the horizon (default 36 months). The
   first visit with `CDRTOT ≥ --convert-cdr` is the conversion event.
4. **AD etiology is confirmed**, requiring `PROBAD == 1` or `alzdis == 1` at the
   nearest d1 visit to the conversion event. Conversions to non-AD dementia are
   *excluded* (label NA), not mislabelled as stable. `--no-ad-etiology` disables
   this.
5. Non-converters are labelled sMCI (0). `label_confident` is True only when the
   subject has clinical follow-up reaching the full horizon; censored negatives
   are flagged so they can be dropped.

Label encoding: `1` = pMCI (AD converter), `0` = sMCI (stable), `NaN` =
ineligible or uncertain.

## Geometry filter

`prefilter_oasis3_pairs.py` reads NIfTI headers only — no voxel data — so it is
fast and needs neither ANTs nor a GPU. It rejects thick-slice scouts (any voxel
dimension above the maximum) and very-thin acquisitions (smallest axis below the
minimum voxel count), plus missing or unreadable files. A pair survives only if
both volumes pass. This runs *before* any heavy ANTs work, so the converter
count is confirmed to survive the filter before preprocessing time is spent.

## Preprocessing

`preprocess_oasis3.py` imports and calls the existing step functions from
`src/tafnet/preprocessing` — it reimplements nothing. OASIS-3 therefore lands in
the same MNI152 / 128³ feature space as ADNI and OASIS-2 by construction:

1. Brain extraction (ANTsPyNet `t1`)
2. Spatial normalisation (ANTs SyNRA → `ants.get_ants_data("mni")`)
3. Intensity normalisation (min-max over brain voxels → [0, 1])
4. Gaussian denoising (σ = 0.5)
5. Centre-crop / pad to 128³, saved with identity affine

It is crash-resilient: existing non-empty outputs are skipped, every volume is
wrapped in try/except, and a log row is appended and flushed after each scan.

`qc_montage_oasis3.py` then renders axial/coronal/sagittal mid-slices. The one
failure mode the numeric pipeline cannot catch is a silent orientation flip —
pass `--reference` with a known-good ADNI or OASIS-2 volume to compare directly.

## Cohort

| Quantity | Value |
|---|---|
| Model-ready pairs | 125 |
| Unique subjects | 101 |
| Converter pairs / subjects | 26 / 25 |
| Prevalence | 0.208 |
| Unique preprocessed volumes | 227 |
| Interval range | 0.2 – 85.1 months (median 29.4) |

Nineteen subjects contribute multiple pairs; two contribute both a converter and
a stable pair. This non-independence is handled by subject-clustered bootstrap
CIs and a one-pair-per-subject sensitivity analysis.

## Expected results

```
Pooled (125 pairs / 101 subjects)
  AUC          0.733   [0.609, 0.847]   subject-clustered bootstrap, 2000 resamples
  Sensitivity  0.308
  Specificity  0.919
  F1           0.381
  Accuracy     0.792

One pair per subject (101 pairs)
  AUC          0.744   [0.610, 0.861]
```

Stratified by scan interval:

| Interval (months) | Pairs (pos/neg) | AUC |
|---|---|---|
| < 6 | 9 (4/5) | 1.000 † |
| 6–24 * | 28 (9/19) | 0.743 |
| 24–60 | 76 (13/63) | 0.692 |
| > 60 | 12 (0/12) | — |
| Pooled | 125 (26/99) | 0.733 |

\* Trained interval regime on ADNI. † Computed on 9 pairs; unstable and reported
only for completeness.

## Reading the numbers

The interval-stratified row is the central control. Restricted to the trained
6–24-month band the AUC is 0.743, essentially identical to the pooled estimate —
confirming the headline is not carried by long-interval pairs, which would
otherwise be the natural concern. Discrimination at 24–60 months (0.692) is
modestly lower, consistent with extrapolation beyond the trained window rather
than reliance on it.

The `> 60` bin contains no converters, leaving AUC undefined. This is a property
of the cohort, not a pipeline failure.

The default 0.5 threshold gives high specificity (0.919) and correspondingly low
sensitivity (0.308), matching the profile on ADNI (0.939) and OASIS-2 (0.949).
AUC measures ranking; a strong ranker operated at a conservative threshold is
not a poor detector. The OASIS-2 analysis showed sensitivity can be raised from
0.473 to 0.703 by shifting the decision boundary without retraining.

With 25 converter subjects the CI is wide by construction. Lead with the pooled
AUC; the lower bound (0.609) lies above chance, so discrimination is
statistically resolved despite the modest converter count.

# TAFNet — A Two-Scan Deep Learning Model for Predicting Dementia in Mild Cognitive Impairment

Reference implementation of TAFNet, a hybrid CNN–Transformer model that predicts
conversion from mild cognitive impairment (MCI) to dementia from a pair of
T1-weighted MRI scans of the same person.

The repository covers the full pipeline: pair labelling from clinical
diagnoses, longitudinal preprocessing, encoder pretraining, 5-fold
participant-level cross-validation with a participant-level held-out test set
against five benchmarks, the ablations, the gate analysis, and the statistical
analyses and figures reported in the paper.

> **No data is included in this repository.** ADNI is distributed under a Data
> Use Agreement that prohibits redistribution. See
> [Data availability](#data-availability).

## Results

Cohort: 604 scan pairs (6–24 months apart) from 325 ADNI participants with MCI
at the first scan; 133 pairs (22.0%) convert to dementia within 36 months of the
first scan. 15% of participants are held out before cross-validation. All
intervals are participant-clustered bootstrap percentile intervals.

| Method | Scans | CV AUC, mean of folds (SD) | CV AUC, pooled [95% CI] | Held-out AUC [95% CI] |
|---|---|---|---|---|
| ResNet3D-18 | 1 | 0.725 (0.055) | 0.648 [0.571, 0.721] | 0.743 [0.526, 0.918] |
| DenseNet3D-121 | 1 | 0.752 (0.033) | 0.605 [0.531, 0.675] | 0.725 [0.506, 0.909] |
| TAFNet-InitialOnly | 1 | 0.797 (0.045) | 0.770 [0.701, 0.834] | 0.827 [0.694, 0.937] |
| Siamese-Subtract | 2 | 0.724 (0.076) | 0.702 [0.640, 0.762] | 0.769 [0.610, 0.909] |
| CNN-LSTM | 2 | 0.820 (0.037) | 0.793 [0.728, 0.854] | 0.859 [0.751, 0.951] |
| **TAFNet** | 2 | 0.814 (0.040) | 0.812 [0.753, 0.866] | 0.852 [0.737, 0.949] |

Cross-validation: 519 pairs / 276 participants / 113 converters. Held-out:
85 / 49 / 20, scored by the mean of the five fold models. At the
validation-derived Youden threshold (0.239), held-out sensitivity is 0.850 and
specificity 0.723; the held-out set was never used to choose a threshold.

What the results support:

* **Two scans beat one.** TAFNet outperforms the single-scan networks
  (pooled cross-validation difference +0.206 against DenseNet3D-121 and +0.164
  against ResNet3D-18, both p < 0.001). With everything else held fixed, adding
  the follow-up scan gives +0.041 (95% CI [+0.011, +0.073], p = 0.009).
* **Learned fusion beats subtraction** (+0.109, 95% CI [+0.038, +0.181],
  p = 0.002).
* These differences are significant in cross-validation and have the same
  direction, without reaching significance, on the smaller held-out set.
* **TAFNet and CNN-LSTM are comparable**, with neither consistently ahead:
  +0.018 [−0.003, +0.039] pooled cross-validation, −0.007 [−0.027, +0.003]
  held-out. A formal equivalence test (margin 0.03 AUC) is met on the held-out
  set and not on the pooled cross-validation predictions, so we claim neither
  superiority nor equivalence.
* **Ablations.** Any single fusion branch performs as well as all three, and
  neither the granularity of the gate nor the baseline residual changes the
  result. The gate coefficients are unstable across fold models and do not
  explain individual predictions.

## Architecture

Three stages, described in full in the manuscript:

1. **Siamese 3D CNN encoder** — five blocks (16→32→64→128→128) with Dynamic
   Contextual Channel Attention (DCCA) after blocks 1 to 4, mapping a 128³
   volume to a (128, 8, 8, 8) bottleneck. Weights are shared across timepoints.
   Pretrained on cross-sectional CN vs dementia volumes (1,948 volumes, 531
   participants not in the longitudinal cohort), then frozen.
2. **Temporal Fusion Module** — three branches (temporal difference,
   cross-temporal multi-head attention with 4 heads, concatenation +
   projection) mixed by an **Adaptive Temporal Gate** that outputs one
   (α, β, γ) per participant from the pooled baseline and follow-up features. A
   baseline residual adds the baseline features back to the fused output.
3. **Classification head** — GAP → FC(128, 64) → ReLU → FC(64, 1) → sigmoid.

3,068,980 parameters in total (Temporal Fusion Module 116,099, of which the
gate 16,643); with the encoder frozen, the longitudinal phase trains 124,420.
Gate mode (`patient` / `position`), the baseline residual and the active
branches are config flags for ablation (`architecture.*` in
`configs/default.yaml`).

Implemented in [`src/tafnet/models/`](src/tafnet/models/).

## Installation

```bash
git clone https://github.com/amoayedikia/tafnet.git && cd tafnet
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# dcm2niix is a binary, not a Python package (only needed if starting from DICOM)
sudo apt-get install -y dcm2niix     # Ubuntu / Debian
# brew install dcm2niix              # macOS
```

Preprocessing needs ANTsPy, ANTsPyNet and TemplateFlow, which download model
weights and the MNI template on first use. Training needs a CUDA GPU.

## Preprocessing

`scripts/preprocess_v2.py` is the pipeline used for the paper. It is
longitudinal and keeps the whole brain:

1. N4 bias-field correction of each scan.
2. An unbiased within-subject template from all of a participant's scans (rigid).
3. Brain extraction once, on the template; one mask per participant.
4. Affine registration of the template to MNI152NLin2009cAsym.
5. The two transformations composed, so each scan is resampled once onto a
   128³ grid at 1.6 mm (204.8 mm field of view).
6. Intensity scaling between the 1st and 99th percentiles inside the brain mask.
7. Quality control: no brain voxels on any face of the box, within-subject and
   template-to-MNI alignment.

No smoothing and no cropping are applied.

The earlier pipeline (`scripts/01_preprocess.py`, `src/tafnet/preprocessing/`)
was **not** used for these results: its final centre crop removed part of the
cortex. It is kept only because the OASIS scripts import its steps.

## Configuration

Settings live in YAML under `configs/`; any field can be changed from the
command line with `--override key.path=value`.

* `configs/default.yaml` — ADNI paths, architecture, training schedules,
  benchmark on/off flags. Key paths:
  * `paths.data_dir` — preprocessed volumes, named `<subject>_<image_id>.nii.gz`
  * `paths.pairs_csv` — labelled pairs from `label_adni_pairs.py`
  * `paths.phase4_csv` — encoder pretraining labels
    (`subject, image_id, baseline_dx, label`; CN = 0, dementia = 1)
  * `paths.holdout_frac` — held-out participant fraction (default 0.15)
  * `paths.output_dir` — checkpoints, per-fold cache and results

## Pipeline

```bash
# 1. Conversion labels from DXSUM: MCI at the first scan, dementia within 36 months,
#    censored negatives and diagnoses before the follow-up scan excluded
python scripts/label_adni_pairs.py \
    --pairs pairs_6_24m.csv \
    --dxsum "$ADNI_CLINICAL/DXSUM.csv" --mmse "$ADNI_CLINICAL/MMSE.csv" \
    --anchor pair \
    --out adni_pairs_labeled_pair.csv --out-ready adni_pairs_ready_pair.csv

# 2. DICOM -> NIfTI (one file per ADNI Image Data ID), then longitudinal preprocessing
python scripts/convert_adni.py --root /path/to/ADNI --out nifti --jobs 4
python scripts/preprocess_v2.py --manifest nifti/manifest.csv \
    --out preprocessed_v2 --work work_v2 --jobs 4

# 3. Encoder pretraining, then 5-fold CV + held-out test for TAFNet and the benchmarks.
#    The paper's runs used these two overrides.
python scripts/02_train.py --config configs/default.yaml --no-drive-check \
    --override phase56.batch_size=4 --override training.accumulation_steps=1

# 4. The eight ablations (paper Section 5.4); see the header of the script
bash scripts/run_ablations.sh

# 5. Per-participant gate coefficients (paper Section 5.6)
python scripts/06_gate_analysis.py --config configs/default.yaml \
    --results-dir "$TAFNET_RESULTS" --out "$TAFNET_RESULTS/gate_analysis"

# 6. Inference cost of TAFNet and CNN-LSTM (paper Section 6.3)
PYTHONPATH=src python scripts/time_inference.py
```

`pairs_6_24m.csv` lists consecutive T1w scans per participant
(`subject, scan1, scan2, date1, date2, interval_days`). Each completed fold is
cached under `<output_dir>/folds/` and a re-run resumes from there; the cache is
invalidated automatically when the cohort, split or architecture changes. Pass
`--skip-phase4` to reuse an existing `phase4_encoder_best.pth`.

## Statistical analyses and figures

The scripts in [`analysis/`](analysis/) reproduce every table and figure of the
results section from the per-fold files written by `02_train.py`. They read
their inputs from environment variables (see `analysis/common.py`):

```bash
export TAFNET_RESULTS=/path/to/results            # output_dir of 02_train.py
export TAFNET_PAIRS_CSV=/path/to/adni_pairs_ready_pair.csv
export TAFNET_SIDECAR_CSV=/path/to/sidecar_summary.csv   # image_id, vendor, series
export TAFNET_ANALYSIS_OUT=analysis_out
cd analysis
```

| Script | Reproduces |
|---|---|
| `auc_comparisons.py` | AUCs, paired differences, equivalence interval (Sections 5.2, 5.3) |
| `ablations.py` | The eight ablation variants (Section 5.4) |
| `operating_points.py` | Validation-derived thresholds (Section 5.5) |
| `calibration.py` | Brier score, ECE, recalibration (Section 5.5) |
| `gate_checks.py` | Gate stability and adjusted associations (Section 5.6) |
| `secondary_analyses.py` | DeLong tests, convergent and sensitivity analyses (Section 5.7) |
| `subgroups.py` | Acquisition subgroups (Section 5.7) |
| `figures.py` | Figures 5 to 10 (run `auc_comparisons.py` first) |

All bootstrap intervals use 10,000 participant-clustered resamples with seed
20260907 (2,000 in `06_gate_analysis.py` and `gate_checks.py`).

## Repository layout

```
tafnet/
├── analysis/                    # statistics and figures of the paper
├── configs/                     # YAML configuration
├── docs/
├── scripts/                     # CLI entry points
│   ├── label_adni_pairs.py      # ADNI conversion labels from DXSUM
│   ├── convert_adni.py          # DICOM -> NIfTI
│   ├── preprocess_v2.py         # longitudinal preprocessing (used for the paper)
│   ├── 02_train.py              # encoder pretraining, CV and held-out test
│   ├── 03_evaluate.py           # rebuild tables and plots from a finished run
│   ├── 06_gate_analysis.py      # per-participant gate analysis
│   ├── run_ablations.sh         # the eight ablation variants
│   └── time_inference.py        # inference cost
└── src/tafnet/
    ├── config.py                # YAML loader with --override support
    ├── data/                    # labelled-pair dataset, held-out split, transforms
    ├── evaluation/              # metrics, reporting, plots
    ├── models/                  # encoder, TAFNet, pretraining head, benchmarks
    ├── training/                # encoder pretraining, CV + held-out training
    └── utils/                   # seeding, device, drive-mount checks
```

The OASIS-2 and OASIS-3 scripts (`scripts/*oasis*`, `scripts/04_*`, `05_*`,
`07_*`, `08_*`, `src/tafnet/external/`, `docs/oasis*`) and the v1 preprocessing
files remain from earlier work. They are not part of the paper and have not been
run with the current models.

## Reproducibility

* Global seed 42 in every script (`tafnet.utils.seed.set_seed`); held-out split
  `test_frac=0.15, random_state=42`, drawn at participant level.
* The exact config for a run is written to `<output_dir>/run_config.yaml`.
* `02_train.py` writes one JSON + checkpoint per method and fold under
  `<output_dir>/folds/`; the analysis scripts read those JSON files. The summary
  files it also writes keep the historical names `comprehensive_results_v4.json`
  and `predictions_v4.json`.
* Volumes that fail to load raise an error instead of being replaced with zeros.
* Preprocessing, DICOM conversion and training folds all resume from where they
  stopped.

## Data availability

ADNI: [adni.loni.usc.edu](https://adni.loni.usc.edu). Access requires an
approved application; no data, labels or derived subject-level files are
committed here. Data used in preparation of this work were obtained from the
Alzheimer's Disease Neuroimaging Initiative (ADNI) database; ADNI investigators
contributed to the design and implementation of ADNI and/or provided data but
did not participate in the analysis or writing.

## Citation

Fin S., Moayedikia A., White D. J., Wiil U. K., Troncoso A. *A Two-Scan Deep
Learning Model for Predicting Dementia in Mild Cognitive Impairment.* Manuscript
in preparation, 2026.

```bibtex
@unpublished{fin2026tafnet,
  title  = {A Two-Scan Deep Learning Model for Predicting Dementia in Mild Cognitive Impairment},
  author = {Fin, Sara and Moayedikia, Alireza and White, David J. and Wiil, Uffe Kock and Troncoso, Alicia},
  note   = {Manuscript in preparation},
  year   = {2026}
}
```

An earlier version of this work, with a different preprocessing pipeline and
different results, is available as arXiv:2605.28397.

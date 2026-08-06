# TAFNet — Adaptive Temporal Gating of Longitudinal MRI for Alzheimer's Prediction

Reference implementation of the Temporal Adaptive Fusion Network (TAFNet), a
hybrid CNN–Transformer architecture that predicts MCI-to-AD conversion from a
pair of longitudinal T1-weighted structural MRI scans.

The repository covers the full research pipeline: preprocessing, encoder
pretraining, 5-fold subject-level cross-validation against six benchmarks, and
zero-shot external validation on two independent cohorts (OASIS-2 and OASIS-3).

> **No data is included in this repository.** ADNI and OASIS are distributed
> under Data Use Agreements that prohibit redistribution. See
> [`data/README.md`](data/README.md) for how to obtain the cohorts and
> regenerate every manifest the pipeline consumes.

## Results

| Cohort | Field | Label derivation | Pairs (conv. subj.) | Prev. | AUC | Sens | Spec |
|---|---|---|---|---|---|---|---|
| ADNI (5-fold CV) | 3T | Clinically adjudicated | 529 (84) | 0.263 | 0.916 ± 0.044 | 0.449 | 0.939 |
| OASIS-2 (zero-shot) | 1.5T | CDR-based | 291 (13) | 0.065 | 0.788 | 0.474 | 0.949 |
| OASIS-3 (zero-shot) | 3T | CDR + UDS AD confirmation | 125 (25) | 0.208 | 0.733 [0.609, 0.847] | 0.308 | 0.919 |

Sensitivity and specificity are at the default 0.5 threshold. The OASIS-3
interval is a subject-clustered bootstrap 95% CI; the ADNI figure is the
cross-fold standard deviation.

OASIS-3 discrimination stratified by scan interval — the control confirming the
headline is not carried by long-interval pairs:

| Interval (months) | Pairs (pos/neg) | AUC |
|---|---|---|
| < 6 | 9 (4/5) | 1.000 † |
| 6–24 * | 28 (9/19) | 0.743 |
| 24–60 | 76 (13/63) | 0.692 |
| > 60 | 12 (0/12) | — |
| **Pooled** | **125 (26/99)** | **0.733** |

\* Trained interval regime. † Computed on 9 pairs; unstable and not interpretable.

## Architecture

Three stages, described in full in the accompanying manuscript:

1. **Siamese 3D CNN encoder** — five-block encoder (16→32→64→128→128) with
   Dense Context Channel Attention, mapping each 128³ volume to a
   (128, 8, 8, 8) bottleneck. Weights are shared across timepoints.
2. **Temporal Fusion Module** — three branches (temporal difference,
   cross-temporal attention, concatenation + projection) mixed by an Adaptive
   Temporal Gate.
3. **Classification head** — GAP → FC(128, 64) → ReLU → FC(64, 1) → sigmoid.

Implemented in [`src/tafnet/models/`](src/tafnet/models/).

## Installation

```bash
git clone <this-repo> tafnet && cd tafnet
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# dcm2niix is a binary, not a Python package (only needed if starting from DICOM)
sudo apt-get install -y dcm2niix     # Ubuntu / Debian
# brew install dcm2niix              # macOS
```

Preprocessing requires ANTsPy and ANTsPyNet, which pull large model weights on
first use. Training requires a CUDA GPU; CPU falls back to FP32 and is not
practical for full training.

## Configuration

Two environment variables resolve the data roots referenced by the OASIS-3
scripts. Both are expanded at runtime alongside `~`:

```bash
export OASIS3_ROOT="/path/to/OASIS3"          # raw OASIS-3 download
export TAFNET_DATA="/path/to/working-data"    # preprocessed volumes, checkpoints, results
```

Everything else lives in YAML under `configs/`, and any field can be patched
from the command line with `--override key.path=value`:

* `configs/preprocessing.yaml` — raw/intermediate/output dirs, ANTs settings,
  intensity normalisation mode, target shape.
* `configs/default.yaml` — ADNI CSV and volume paths, visit-pair definitions,
  architecture hyperparameters, training schedules, benchmark on/off flags.
* `configs/oasis2.yaml` — external-validation paths and the architecture block,
  which must match the checkpoint being loaded.

## Pipeline

### ADNI — training

```bash
python scripts/00_dicom_to_nifti.py --config configs/preprocessing.yaml
python scripts/01_preprocess.py     --config configs/preprocessing.yaml
python scripts/02_train.py          --config configs/default.yaml
python scripts/03_evaluate.py       --config configs/default.yaml
```

`02_train.py` runs Phase 4 encoder pretraining (cross-sectional CN vs AD) then
5-fold subject-level CV of TAFNet plus six benchmarks. `03_evaluate.py` replays
the results table, paired statistical tests, ROC curves and AUC box plots from a
finished run without retraining.

### OASIS-2 — external validation

```bash
python scripts/04_oasis2_zeroshot.py    --config configs/oasis2.yaml
python scripts/05_oasis2_finetune_cv.py --config configs/oasis2.yaml
```

See [`docs/oasis2_external_validation.md`](docs/oasis2_external_validation.md).

### OASIS-3 — external validation

A six-step chain; each step writes a CSV consumed by the next. Steps 1–5 run
locally on CPU, step 6 takes a few minutes for 125 pairs.

```bash
# 1. Inventory the clinical CSVs (UDS forms) — orientation only
python scripts/explore_oasis3.py --root "$OASIS3_ROOT/OASIS3_data_files"

# 2. Inventory the imaging tree, build consecutive T1w pairs
python scripts/explore_oasis3_imaging.py --root "$OASIS3_ROOT" \
    --scan-type T1w --min-interval 180 --max-interval 730

# 3. Assign conversion labels from CDR + UDS diagnoses
python scripts/label_oasis3_pairs.py \
    --pairs oasis3_t1w_pairs_m6-m24.csv \
    --clinical-root "$OASIS3_ROOT/OASIS3_data_files"

# 4. Header-only geometry filter; decides what gets preprocessed
python scripts/prefilter_oasis3_pairs.py --pairs oasis3_t1w_pairs_ready.csv

# 5. Run the identical TAFNet preprocessing chain over surviving volumes
python scripts/preprocess_oasis3.py \
    --vols oasis3_unique_volumes.csv \
    --pairs oasis3_pairs_ready_filtered.csv

# 5b. Visual QC — the one failure mode the numeric pipeline cannot catch
python scripts/qc_montage_oasis3.py --dir "$TAFNET_DATA/oasis3_preprocessed" \
    --reference "$TAFNET_DATA/oasis2_preprocessed/<known_good>.nii.gz"

# 6. Zero-shot inference + interval-stratified AUC + bootstrap CI
python scripts/04_oasis3_zeroshot.py \
    --config configs/oasis2.yaml \
    --pairs-csv oasis3_pairs_preprocessed.csv \
    --checkpoint "$TAFNET_DATA/tafnet_v4_fold1_best.pth" \
    --output-dir "$TAFNET_DATA/oasis3_results"
```

See [`docs/oasis3_external_validation.md`](docs/oasis3_external_validation.md)
for the labelling logic, cohort counts, and expected numbers.

## Repository layout

```
tafnet/
├── configs/                     # YAML configuration
├── data/README.md               # how to obtain cohorts (no data committed)
├── docs/
│   ├── gcp_setup.md             # GCP VM + rclone walkthrough
│   ├── oasis2_external_validation.md
│   └── oasis3_external_validation.md
├── scripts/                     # CLI entry points, numbered by pipeline stage
└── src/tafnet/
    ├── config.py                # YAML loader with --override support
    ├── data/                    # Dataset classes and torchio transforms
    ├── evaluation/              # Metrics, reporting, plot helpers
    ├── external/                # External-cohort helpers
    ├── models/                  # Encoder, TAFNet, Phase-4 head, benchmarks
    ├── preprocessing/           # ANTs / ANTsPyNet preprocessing steps
    ├── training/                # Phase-4 pretraining + CV training loops
    └── utils/                   # Seeding, drive-mount checks
```

## Reproducibility

* Global seed fixed in every script (`tafnet.utils.seed.set_seed`, default 42).
* The exact config for a run is written to `<output_dir>/run_config.yaml`.
* `02_train.py` dumps `comprehensive_results_v4.json` and `predictions_v4.json`,
  which `03_evaluate.py` consumes to recreate plots without retraining.
* Preprocessing and DICOM conversion are resumable — existing outputs are
  skipped, and `preprocess_oasis3.py` appends a log row after every volume so a
  kill/resume loses at most one scan.

## Data availability

ADNI: [adni.loni.usc.edu](https://adni.loni.usc.edu) · OASIS-2 and OASIS-3:
[oasis-brains.org](https://www.oasis-brains.org). Both require an approved
application. Data used in preparation of this work were obtained from the
Alzheimer's Disease Neuroimaging Initiative (ADNI) database; ADNI investigators
contributed to design and implementation and/or provided data but did not
participate in analysis or writing.

## Citation

Moayedikia A., Fin S., Troncoso A., Wiil U. K. *Adaptive Temporal Gating of
Longitudinal Magnetic Resonance Imaging for Alzheimer's Prediction.* Manuscript
under review.

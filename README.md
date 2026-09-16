# TAFNet — Adaptive Temporal Gating of Longitudinal Magnetic Resonance Imaging for Dementia Prediction

[![arXiv](https://img.shields.io/badge/arXiv-2605.28397-b31b1b.svg)](https://arxiv.org/abs/2605.28397)

Reference implementation of the Temporal Adaptive Fusion Network (TAFNet), a
hybrid CNN–Transformer model that predicts conversion from mild cognitive
impairment (MCI) to dementia from a pair of longitudinal T1-weighted MRI scans.

The repository covers the full pipeline: pair labelling from clinical
diagnoses, preprocessing, encoder pretraining, 5-fold subject-level
cross-validation with a subject-level held-out test set against five
benchmarks, per-patient gate analysis, and zero-shot scoring of an external
cohort (OASIS-3).

> **No data is included in this repository.** ADNI and OASIS are distributed
> under Data Use Agreements that prohibit redistribution. See
> [Data availability](#data-availability).

## Results

### ADNI

Cohort: 604 scan pairs (6–24 months apart) from 325 participants with MCI at
the first scan; 133 pairs (22.0%) convert to dementia within 36 months. 15% of
participants are held out before cross-validation.

| Partition | Pairs / participants / converters | AUC [95% CI] | Sens | Spec |
|---|---|---|---|---|
| Cross-validation (5-fold) | 519 / 276 / 113 | 0.832 ± 0.042 (fold mean); 0.821 pooled | — | — |
| Held-out test (5-fold ensemble) | 85 / 49 / 20 | **0.846** [0.729, 0.946] | 0.850 | 0.754 |

Held-out sensitivity and specificity are at the validation-derived Youden
threshold (0.191); the held-out set was never used to choose it. Intervals are
participant-clustered bootstrap percentile intervals.

Benchmarks (same folds and held-out set):

| Method | Timepoints | Held-out AUC [95% CI] | CV pooled AUC |
|---|---|---|---|
| ResNet3D-18 | 1 | 0.723 [0.514, 0.904] | 0.649 |
| DenseNet3D-121 | 1 | 0.744 [0.534, 0.921] | 0.721 |
| Siamese-Subtract | 2 | 0.721 [0.564, 0.847] | 0.682 |
| TAFNet-InitialOnly | 1 | 0.819 [0.688, 0.927] | 0.798 |
| CNN-LSTM | 2 | 0.847 [0.729, 0.946] | 0.815 |
| **TAFNet** | 2 | **0.846** [0.729, 0.946] | **0.821** |

TAFNet and CNN-LSTM beat the single-timepoint CNNs (significant in
cross-validation), and learned fusion beats plain subtraction (significant on
both partitions). TAFNet and CNN-LSTM perform equivalently (held-out ΔAUC
−0.001, 95% CI [−0.017, +0.014]).

### OASIS-3 (zero-shot, exploratory)

The five ADNI fold models applied without retraining; score = mean probability
across folds. 125 pairs from 101 participants, 26 converter pairs (25
participants), prevalence 0.208. Labels: CDR with UDS confirmation of AD
aetiology, 36-month horizon.

| Analysis | AUC [95% CI] |
|---|---|
| All pairs (pooled) | 0.760 [0.66, 0.85] |
| One pair per participant (earliest) | 0.766 [0.66, 0.86] |

| Scan interval (months) | Pairs (pos/neg) | AUC |
|---|---|---|
| < 6 | 9 (4/5) | 1.000 † |
| 6–24 * | 28 (9/19) | 0.713 |
| 24–60 | 76 (13/63) | 0.781 |
| > 60 | 12 (0/12) | — |

\* Interval range used in ADNI training. † 9 pairs; not interpretable.

The score distribution shifts on transfer, so the ADNI threshold does not carry
over (at 0.191: sensitivity 0.692, specificity 0.657). Re-deriving the cut-off
in OASIS-3 gives 0.077 (bootstrap 95% CI 0.028–0.569); with the cut-off chosen
on other participants in repeated 5-fold cross-validation, sensitivity is 0.818
and specificity 0.531. Intervals are participant-clustered.

OASIS-2 has not yet been re-evaluated with the v4 models; its scripts remain in
the repository for the earlier checkpoint.

## Architecture

Three stages, described in full in the manuscript:

1. **Siamese 3D CNN encoder** — five blocks (16→32→64→128→128), each followed
   by Dynamic Contextual Channel Attention (DCCA), mapping a 128³ volume to a
   (128, 8, 8, 8) bottleneck. Weights are shared across timepoints. Pretrained
   on cross-sectional CN vs dementia volumes (1,948 volumes, 531 participants
   not in the longitudinal cohort), then frozen.
2. **Temporal Fusion Module** — three branches (temporal difference,
   cross-temporal multi-head attention with 4 heads, concatenation +
   projection) mixed by an **Adaptive Temporal Gate** that outputs one
   (α, β, γ) per patient from the pooled baseline and follow-up features. A
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

Preprocessing needs ANTsPy and ANTsPyNet, which download model weights on first
use. Training needs a CUDA GPU. Inference with trained models (for example the
OASIS-3 scoring below) runs on CPU in a few seconds per pair.

## Configuration

Settings live in YAML under `configs/`; any field can be changed from the
command line with `--override key.path=value`.

* `configs/preprocessing.yaml` — raw/intermediate/output dirs, ANTs settings,
  intensity normalisation, target shape.
* `configs/default.yaml` — ADNI paths, architecture, training schedules,
  benchmark on/off flags. Key paths:
  * `paths.data_dir` — preprocessed volumes, named `<subject>_<image_id>.nii.gz`
  * `paths.pairs_csv` — labelled pairs from `label_adni_pairs.py`
  * `paths.phase4_csv` — Phase 4 pretraining labels
    (`subject, image_id, baseline_dx, label`; CN = 0, dementia = 1)
  * `paths.holdout_frac` — held-out participant fraction (default 0.15)
  * `paths.output_dir` — checkpoints, per-fold cache and results
* `configs/oasis2.yaml` — external-validation paths for the OASIS-2 scripts.

Environment variables used by the OASIS-3 and labelling examples:

```bash
export OASIS3_ROOT="/path/to/OASIS3"          # raw OASIS-3 download
export TAFNET_DATA="/path/to/working-data"    # preprocessed volumes, checkpoints, results
export ADNI_CLINICAL="/path/to/ADNIMERGE"     # DXSUM.csv, MMSE.csv
```

## Pipeline

### ADNI — labels, preprocessing, training

```bash
# 1. Conversion labels from DXSUM: MCI at the first scan, dementia within 36 months,
#    censored negatives and diagnoses before the follow-up scan excluded
python scripts/label_adni_pairs.py \
    --pairs pairs_6_24m.csv \
    --dxsum "$ADNI_CLINICAL/DXSUM.csv" --mmse "$ADNI_CLINICAL/MMSE.csv" \
    --anchor pair \
    --out adni_pairs_labeled_pair.csv --out-ready adni_pairs_ready_pair.csv

# 2. DICOM -> NIfTI -> 128^3 preprocessed volumes
python scripts/00_dicom_to_nifti.py --config configs/preprocessing.yaml
python scripts/01_preprocess.py     --config configs/preprocessing.yaml

# 3. Phase 4 encoder pretraining, then 5-fold CV + held-out test for TAFNet and benchmarks
python scripts/02_train.py --config configs/default.yaml --no-drive-check

# 4. Rebuild tables, paired tests and plots from a finished run
python scripts/03_evaluate.py --config configs/default.yaml --no-drive-check

# 5. Per-patient gate coefficients vs clinical progression
python scripts/06_gate_analysis.py --config configs/default.yaml \
    --results-dir "$TAFNET_DATA/results" --out "$TAFNET_DATA/results/gate_analysis"
```

`pairs_6_24m.csv` lists consecutive T1w scans per participant
(`subject, scan1, scan2, date1, date2, interval_days`). Each completed fold is
cached under `<output_dir>/folds/` and a re-run resumes from there; the cache is
invalidated automatically when the cohort, split or architecture changes. Pass
`--skip-phase4` to reuse an existing `phase4_encoder_best.pth`.

### OASIS-3 — cohort and preprocessing

Each step writes a CSV consumed by the next.

```bash
# 1. Inventory the clinical CSVs (UDS forms) — orientation only
python scripts/explore_oasis3.py --root "$OASIS3_ROOT/OASIS3_data_files"

# 2. Inventory the imaging tree, build consecutive T1w pairs
python scripts/explore_oasis3_imaging.py --root "$OASIS3_ROOT" \
    --scan-type T1w --min-interval 180 --max-interval 730

# 3. Conversion labels from CDR + UDS diagnoses
python scripts/label_oasis3_pairs.py \
    --pairs oasis3_t1w_pairs_m6-m24.csv \
    --clinical-root "$OASIS3_ROOT/OASIS3_data_files"

# 4. Header-only geometry filter; decides what gets preprocessed
python scripts/prefilter_oasis3_pairs.py --pairs oasis3_t1w_pairs_ready.csv

# 5. The same TAFNet preprocessing chain as ADNI
python scripts/preprocess_oasis3.py \
    --vols oasis3_unique_volumes.csv \
    --pairs oasis3_pairs_ready_filtered.csv

# 5b. Visual QC
python scripts/qc_montage_oasis3.py --dir "$TAFNET_DATA/oasis3_preprocessed" \
    --reference "$TAFNET_DATA/oasis2_preprocessed/<known_good>.nii.gz"
```

See [`docs/oasis3_external_validation.md`](docs/oasis3_external_validation.md)
for the labelling logic and cohort counts.

### OASIS-3 — scoring with the v4 models

```bash
# 6. Score every pair with the five TAFNet-Full fold models (CPU is fine).
#    --check verifies files, loads each checkpoint strictly and scores one pair.
python scripts/07_score_oasis3_v4.py \
    --pairs  pairs.csv \
    --vol-dir  "$TAFNET_DATA/oasis3_preprocessed" \
    --ckpt-dir "$TAFNET_DATA/results/folds" \
    --check
python scripts/07_score_oasis3_v4.py \
    --pairs  pairs.csv \
    --vol-dir  "$TAFNET_DATA/oasis3_preprocessed" \
    --ckpt-dir "$TAFNET_DATA/results/folds"

# 7. Choose and evaluate an OASIS-3 cut-off (participant-clustered)
python scripts/08_tune_threshold_oasis3.py \
    --scores results_oasis3_v4/OASIS3_125_tafnet_v4_scores.csv
```

`pairs.csv` needs `OASISID, baseline_session, followup_session` (e.g.
`OAS30007_d2722`), plus `pair_label_converter` for the evaluation. Step 6 writes
per-fold probabilities, the ensemble score, fold spread and per-patient gate
weights, and resumes if interrupted. `04_oasis3_zeroshot.py` is the earlier
single-checkpoint version of this step.

### OASIS-2 — earlier checkpoint

```bash
python scripts/04_oasis2_zeroshot.py    --config configs/oasis2.yaml
python scripts/05_oasis2_finetune_cv.py --config configs/oasis2.yaml
```

See [`docs/oasis2_external_validation.md`](docs/oasis2_external_validation.md).
Not yet re-run with the v4 models.

## Repository layout

```
tafnet/
├── configs/                     # YAML configuration
├── docs/
│   ├── gcp_setup.md             # GCP VM + rclone walkthrough
│   ├── oasis2_external_validation.md
│   └── oasis3_external_validation.md
├── scripts/                     # CLI entry points
│   ├── label_adni_pairs.py      # ADNI conversion labels from DXSUM
│   ├── 00–03_*.py               # ADNI preprocessing, training, evaluation
│   ├── 04–05_*.py               # external validation (earlier checkpoint)
│   ├── 06_gate_analysis.py      # per-patient gate analysis
│   ├── 07–08_*.py               # OASIS-3 scoring with v4 models, cut-off tuning
│   └── *oasis3*.py              # OASIS-3 cohort building and preprocessing
└── src/tafnet/
    ├── config.py                # YAML loader with --override support
    ├── data/                    # labelled-pair dataset, held-out split, transforms
    ├── evaluation/              # metrics, reporting, plots
    ├── external/                # external-cohort helpers
    ├── models/                  # encoder, TAFNet, Phase 4 head, benchmarks
    ├── preprocessing/           # ANTs / ANTsPyNet preprocessing steps
    ├── training/                # Phase 4 pretraining, CV + held-out training
    └── utils/                   # seeding, device, drive-mount checks
```

## Reproducibility

* Global seed 42 in every script (`tafnet.utils.seed.set_seed`); held-out split
  `test_frac=0.15, random_state=42`, drawn at participant level.
* The exact config for a run is written to `<output_dir>/run_config.yaml`.
* `02_train.py` writes `comprehensive_results_v4.json`, `predictions_v4.json`
  and one JSON + checkpoint per method and fold; `03_evaluate.py` rebuilds plots
  without retraining.
* Volumes that fail to load raise an error instead of being replaced with zeros.
* Preprocessing, DICOM conversion, training folds and OASIS-3 scoring all
  resume from where they stopped.

## Data availability

ADNI: [adni.loni.usc.edu](https://adni.loni.usc.edu) · OASIS-2 and OASIS-3:
[oasis-brains.org](https://www.oasis-brains.org). Both require an approved
application; no data, labels or derived subject-level files are committed here.
Data used in preparation of this work were obtained from the Alzheimer's Disease
Neuroimaging Initiative (ADNI) database; ADNI investigators contributed to the
design and implementation of ADNI and/or provided data but did not participate
in the analysis or writing.

## Citation

Fin S., Moayedikia A., Troncoso Lora A., Wiil U. K. *Adaptive Temporal Gating of Longitudinal Magnetic Resonance Imaging for Dementia Prediction.*
arXiv:2605.28397, 2026. https://arxiv.org/abs/2605.28397

```bibtex
@article{fin2026tafnet,
  title   = {Adaptive Temporal Gating of Longitudinal Magnetic Resonance Imaging for Dementia Prediction},
  author  = {Fin, Sara and Moayedikia, Alireza and Troncoso Lora, Alicia and Wiil, Uffe Kock},
  journal = {arXiv preprint arXiv:2605.28397},
  year    = {2026}
}
```

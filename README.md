# TAFNet — Adaptive Temporal Gating of Longitudinal MRI for Alzheimer's Prediction

A clean, runnable Python package extracted from the original TAFNet research
notebooks. The package targets a GCP VM with a GPU and reads ADNI data
(DICOM and / or `.nii.gz`) from Google Drive via an `rclone` mount.

The code below is a faithful conversion of two notebooks from the upstream
research repository:

| Module(s)                                | Origin notebook                                              |
|------------------------------------------|--------------------------------------------------------------|
| `src/tafnet/preprocessing/`              | `ADNI_Complete_Preprocessing_Pipeline_v5.ipynb`              |
| `src/tafnet/{models,training,evaluation,data}/` | `TAFNet_v4_Comprehensive.ipynb`                       |

Nothing about the model, the training recipe, or the preprocessing chain has
been altered — only the packaging (CLI scripts + YAML config + library layout)
is new.


## What the package does

1. **DICOM → NIfTI** conversion of raw ADNI downloads (`scripts/00_dicom_to_nifti.py`).
2. **Preprocessing**: brain extraction (ANTsPyNet) → spatial normalisation to
   MNI152 with ANTs SyNRA → intensity normalisation → Gaussian denoising →
   centre-crop / pad to 128³ (`scripts/01_preprocess.py`).
3. **Training**: Phase 4 encoder pretraining (cross-sectional baseline-vs-AD)
   followed by 5-fold subject-level CV training of TAFNet plus six benchmarks
   — TAFNet-NoLong, ResNet3D-18, DenseNet3D-121, Siamese-CNN, CNN-LSTM
   (`scripts/02_train.py`).
4. **Evaluation replay**: regenerate the results table, paired statistical
   tests, ROC curves, and AUC box plots from a completed run without
   retraining (`scripts/03_evaluate.py`).

On the ADNI cohort used in the manuscript (~319 MCI subjects, 84 pMCI /
235 sMCI, 529 longitudinal pairs at 6 / 12 / 24-month gaps), the TAFNet-Full
configuration achieves an AUC of ≈0.916 with specificity ≈0.939 at the 0.5
threshold.


## Quickstart

```bash
# 1. Clone and enter the project
git clone <your-fork-url> tafnet && cd tafnet

# 2. Install Python deps (a fresh venv is recommended)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install the dcm2niix binary (only needed if you start from DICOM)
sudo apt-get install -y dcm2niix     # Ubuntu / Debian

# 4. Mount your Google Drive at /mnt/drive
#    See docs/gcp_setup.md for the rclone setup.

# 5. Adjust paths to taste, then run the pipeline:
$EDITOR configs/default.yaml configs/preprocessing.yaml

python scripts/00_dicom_to_nifti.py --config configs/preprocessing.yaml
python scripts/01_preprocess.py     --config configs/preprocessing.yaml
python scripts/02_train.py          --config configs/default.yaml
python scripts/03_evaluate.py       --config configs/default.yaml

# 6. (Optional) OASIS-2 external validation of the trained checkpoint.
#    See docs/oasis2_external_validation.md for inputs and expected numbers.
python scripts/04_oasis2_zeroshot.py    --config configs/oasis2.yaml
python scripts/05_oasis2_finetune_cv.py --config configs/oasis2.yaml
```

Every script accepts `--override key.path=value` to patch the config from the
command line, and `--no-drive-check` to bypass the rclone-mount sanity check
(useful for local smoke tests with synthetic data).


## Configuration

Two YAML files under `configs/` hold every tunable parameter:

* **`configs/preprocessing.yaml`** — raw DICOM dir, intermediate NIfTI dir,
  preprocessed output dir, ANTs SyNRA settings, intensity-normalisation mode,
  target shape, etc.
* **`configs/default.yaml`** — paths to the ADNI CSV and preprocessed volumes,
  visit-pair definitions, encoder/architecture hyperparameters, Phase 4 and
  Phase 5/6 training hyperparameters, and per-benchmark on/off flags.

Default paths assume `/mnt/drive` is your rclone mount; override them with
`--override paths.drive_root=...` (or by editing the YAML directly).


## Directory layout

```
tafnet/
├── configs/
│   ├── default.yaml             # training / evaluation config
│   └── preprocessing.yaml       # DICOM-to-NIfTI + preprocessing config
├── scripts/
│   ├── 00_dicom_to_nifti.py     # raw DICOM -> .nii.gz (dcm2niix wrapper)
│   ├── 01_preprocess.py         # brain extraction, registration, normalisation
│   ├── 02_train.py              # Phase 4 + 5-fold CV across all models
│   └── 03_evaluate.py           # replay plots / stats from a finished run
├── src/tafnet/
│   ├── config.py                # YAML loader with --override support
│   ├── data/                    # Dataset classes and torchio transforms
│   ├── models/                  # Encoder, TAFNet, Phase-4 head, benchmarks
│   ├── preprocessing/           # ANTs / ANTsPyNet preprocessing steps
│   ├── training/                # Phase-4 pretraining + CV training loops
│   ├── evaluation/              # Metrics, reporting, plot helpers
│   └── utils/                   # Seeding, Drive-mount checks
├── docs/
│   └── gcp_setup.md             # GCP VM + rclone setup walkthrough
└── requirements.txt
```


## Reproducibility

* Global seed is fixed in every script (`tafnet.utils.seed.set_seed`,
  default `42`).
* The exact config used for a training run is written to
  `<output_dir>/run_config.yaml`.
* `scripts/02_train.py` also dumps `comprehensive_results_v4.json` and
  `predictions_v4.json`, which `scripts/03_evaluate.py` consumes to recreate
  ROC and box plots without rerunning anything.


## Notes

* The original notebooks are not part of this package; only the verified
  paper-version code paths were ported. If you need to reference them they
  remain in the upstream `tafnet-main` archive.
* The training scripts use `torch.amp` (the new, non-deprecated API) with an
  explicit `device_type="cuda"`; running on CPU will fall back to FP32 but is
  not practical for full ADNI training.
* The DICOM-to-NIfTI step is **resumable**: re-running `00_dicom_to_nifti.py`
  skips series directories that already have a `.nii.gz` output.
* For configuration overrides the parser handles `true / false / null / int /
  float`; everything else is a string. Quote shell values as usual.

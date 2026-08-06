# OASIS-2 External Validation

This document describes how to reproduce the OASIS-2 external-validation results
for TAFNet: zero-shot transfer of the ADNI-trained checkpoint, and the
warm-start fine-tuning A/B comparison under subject-level 5-fold CV.

It assumes the ADNI training pipeline (`scripts/00`–`scripts/03`) has already
produced a TAFNet checkpoint, and that OASIS-2 has been preprocessed with the
*same* pipeline as ADNI (`scripts/01_preprocess.py`), i.e. skull-stripped,
MNI-registered, intensity-normalised, denoised, resampled to 128³.

## Inputs

| Item | Default location | Notes |
|---|---|---|
| Inventory CSV | `~/oasis2/oasis2_inventory.csv` | Columns `Subject, Visit, Image Data ID, Group` |
| Preprocessed NIfTIs | `~/oasis2/oasis2_preprocessed/` | Named `{Subject}_{Image Data ID}.nii.gz` |
| ADNI checkpoint | `~/oasis2/checkpoints/tafnet_v4_fold1_best.pth` | **Full** model state_dict |

The inventory CSV uses the **same schema** the core dataset reads for ADNI.
OASIS-2 specifics:

- `Visit` is one of `v1, v2, v3, v4, v5`.
- `Group` is CDR-derived: `CN` (CDR 0), `MCI` (CDR 0.5), `AD` (CDR ≥ 1.0).
- Conversion labelling is implicit in `Group`: a pair is a converter (label 1)
  when its **follow-up** visit's `Group` is `AD`, exactly as for ADNI. No
  OASIS-2-specific dataset code is needed — the core
  `MultiTimepointLongitudinalDataset` handles it.

All forward visit pairs across `v1..v5` are enumerated by default
(`tafnet.external.DEFAULT_OASIS2_VISIT_PAIRS`).

## 1. Zero-shot transfer

```bash
python scripts/04_oasis2_zeroshot.py --config configs/oasis2.yaml --no-drive-check
```

Writes to `paths.oasis2_output_dir`:

- `oasis2_zeroshot_predictions.json`  — `{y_true, y_pred}` for ROC / analysis
- `oasis2_zeroshot_metrics.json`      — AUC, Sens, Spec, F1, Accuracy

Expected (full 291-pair cohort): **AUC ≈ 0.788**, Sens ≈ 0.47, Spec ≈ 0.95.

MCI-only sub-cohort:

```bash
python scripts/04_oasis2_zeroshot.py --config configs/oasis2.yaml --no-drive-check \
  --override paths.oasis2_csv=~/oasis2/oasis2_inventory_mci_only.csv \
  --override paths.oasis2_output_dir=~/oasis2/results_mci_only
```

## 2. Warm-start fine-tuning A/B (5-fold CV)

```bash
python scripts/05_oasis2_finetune_cv.py --config configs/oasis2.yaml --no-drive-check
```

Per fold, the same checkpoint is (a) scored zero-shot on the test partition,
(b) fine-tuned on the train partition with the **encoder frozen** (only the
Temporal Fusion Module + classifier adapt; LR 1e-5; class-weighted BCE;
10 epochs, no early stopping; encoder kept in eval mode so BatchNorm stats do
not drift), then (c) re-scored on the **same** test partition.

Subject-level folds come from `get_subject_level_split_indices` — the same
splitter used by the main training code — so a subject never appears in both
train and test. There is no separate split-CSV step.

Writes to `paths.oasis2_output_dir/results_finetune/`:

```
results_finetune/
├── fold{1..5}/
│   ├── tafnet_oasis2_fold{i}_best.pth      # fine-tuned weights (final epoch)
│   ├── zeroshot_test_predictions.json      # baseline on this fold's test set
│   └── finetuned_test_predictions.json     # fine-tuned on the same test set
└── comparison.json                         # per-fold + pooled + Wilcoxon
```

Expected: mean ΔAUC ≈ **+0.008** (Wilcoxon p ≈ 0.31 — not significant), with a
pooled operating-point shift (sensitivity up, specificity down) driven by the
heavy positive-class weight. The headline external-validation number is the
**pooled** AUC, not the fold-averaged mean (folds carry only 2–5 converters
each, so single mis-ranked cases swing per-fold AUC).

## Quick smoke test

```bash
python scripts/05_oasis2_finetune_cv.py --config configs/oasis2.yaml --no-drive-check \
  --override oasis2_finetune.epochs=2 --override training.num_folds=2
```

## Notes

- The checkpoint is loaded **whole** (`load_full_checkpoint`), not via
  `TAFNet.load_pretrained_encoder` (which is encoder-only).
- `architecture` in `configs/oasis2.yaml` must match the checkpoint
  (`[16,32,64,128,128]`, DCCA on, feature_dim 128, num_heads 8). If you change
  the encoder for a retrain, update both.
- The classifier emits logits: inference applies sigmoid; fine-tuning uses
  `BCEWithLogitsLoss`.

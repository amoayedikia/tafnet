"""
Datasets for TAFNet.

CrossSectionalDataset
    All individual CN and AD scans (used for Phase 4 encoder pretraining).

MultiTimepointLongitudinalDataset
    Subject-paired scans across multiple visit combinations
    ((bl,m06), (bl,m12), (bl,m24), (m06,m12), (m12,m24), (y1,y2)).
    Provides subject-level stratified K-fold splits to prevent leakage.
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from .transforms import build_train_transform


# ---------------------------------------------------------------------------
# Visit normalisation (shared by all datasets)
# ---------------------------------------------------------------------------

VISIT_ALIASES = {
    "bl":  ["bl", "baseline", "sc", "screening"],
    "m06": ["m06", "m6", "month6"],
    "m12": ["m12", "month12"],
    "m24": ["m24", "month24"],
    "m36": ["m36", "month36"],
    "m48": ["m48", "month48"],
    "y1":  ["y1", "year1"],
    "y2":  ["y2", "year2"],
}


def normalize_visit(v: str) -> str:
    v = v.lower().strip().replace(" ", "")
    for canonical, aliases in VISIT_ALIASES.items():
        if v in aliases:
            return canonical
    return v


def _load_nifti_volume(path: str, fallback_shape: Tuple[int, int, int] = (128, 128, 128)
                      ) -> torch.Tensor:
    """Load a NIfTI, min-max normalise to [0,1] if needed, return (1,D,H,W) tensor."""
    try:
        data = nib.load(path).get_fdata(dtype=np.float32)
        if data.max() > 1.0:
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        return torch.from_numpy(data).unsqueeze(0)
    except Exception:  # noqa: BLE001 — match original silent fallback
        return torch.zeros((1, *fallback_shape), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Phase 4 dataset (cross-sectional CN vs AD)
# ---------------------------------------------------------------------------

class CrossSectionalDataset(Dataset):
    """
    Loads ALL individual CN and AD scans (no pairing required).
    Used for Phase 4 encoder pretraining.

    Labels: CN=0, AD=1. MCI scans are dropped here.
    """

    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        is_training: bool = False,
        verify_files: bool = True,
    ):
        self.data_dir = data_dir
        self.is_training = is_training
        self.transform = build_train_transform() if is_training else None

        df = pd.read_csv(csv_path)
        label_map = {"CN": 0, "AD": 1, "MCI": None,
                     "EMCI": None, "LMCI": None, "SMC": None}
        df["label"] = df["Group"].map(label_map)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        self.samples: List[dict] = []
        labels: List[int] = []
        missing = 0
        for _, row in df.iterrows():
            subj = row["Subject"]
            image_id = row["Image Data ID"]
            path = os.path.join(data_dir, f"{subj}_{image_id}.nii.gz")
            if verify_files:
                if not os.path.exists(path) or os.path.getsize(path) == 0:
                    missing += 1
                    continue
            self.samples.append({
                "path": path,
                "label": int(row["label"]),
                "subject": subj,
            })
            labels.append(int(row["label"]))

        self.labels = np.array(labels)
        n_pos = int(self.labels.sum())
        n_neg = len(self.labels) - n_pos

        print("\nPHASE 4 DATASET (Cross-Sectional)")
        print(f"  Loaded: {len(self.samples)} scans | Missing: {missing}")
        print(f"  Labels: CN={n_neg}, AD={n_pos}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = _load_nifti_volume(s["path"])
        label = torch.tensor([s["label"]], dtype=torch.float32)
        if self.transform is not None:
            subject = tio.Subject(image=tio.ScalarImage(tensor=img))
            img = self.transform(subject)["image"].data
        return img, label


# ---------------------------------------------------------------------------
# Phase 5/6 dataset (multi-timepoint longitudinal)
# ---------------------------------------------------------------------------

class MultiTimepointLongitudinalDataset(Dataset):
    """
    Longitudinal MRI pairs with multi-timepoint pooling.

    Yields (initial_scan, followup_scan, label) where the label is taken from
    the follow-up visit (CN/sMCI=0, AD/pMCI=1).

    Provides `get_subject_level_split_indices()` for stratified K-fold CV with
    strict subject-level isolation — a single subject never appears in both
    train and val.
    """

    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        visit_pairs: Sequence[Tuple[str, str, int]],
        is_training: bool = False,
        verify_files: bool = True,
    ):
        self.data_dir = data_dir
        self.is_training = is_training
        self.transform = build_train_transform() if is_training else None

        df = pd.read_csv(csv_path)
        # CN/sMCI=0, AD/pMCI=1 (any MCI subtype treated as non-converter at scan time)
        # WARNING - DO NOT USE THIS PATH FOR ADNI.
        # `Group` is the ENROLMENT research group and is constant across a
        # subject's visits (0 of 971 subjects change it), so "label at
        # follow-up" below is just the enrolment label. Training on it gives
        # AD-enrolled vs rest, not MCI-to-dementia conversion, and reproduces
        # the published 529 pairs / 319 subjects / 84 positives from a cohort
        # that is actually 157 MCI + 115 CN + 47 AD.
        # Use data.LabelledPairDataset with the output of
        # label_adni_pairs.py instead. This class is retained only for
        # OASIS-2, whose Group column is CDR-derived per visit.
        label_map = {"CN": 0, "AD": 1, "MCI": 0,
                     "EMCI": 0, "LMCI": 0, "SMC": 0}
        df["label"] = df["Group"].map(label_map)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        # subject -> visit_code -> {image_id, label}
        subject_visits: dict = defaultdict(dict)
        for _, row in df.iterrows():
            subj = row["Subject"]
            v = str(row["Visit"]).lower().strip().replace(" ", "")
            subject_visits[subj][v] = {
                "image_id": row["Image Data ID"],
                "label": int(row["label"]),
            }

        def find_visit(visits_dict, target):
            target_norm = normalize_visit(target)
            for v, payload in visits_dict.items():
                if normalize_visit(v) == target_norm:
                    return payload
            return None

        self.samples: List[dict] = []
        labels: List[int] = []
        subjects: List[str] = []
        pair_counts: dict = defaultdict(int)
        missing = 0

        for subj, visits in subject_visits.items():
            for (v1, v2, _gap) in visit_pairs:
                t1_data = find_visit(visits, v1)
                t2_data = find_visit(visits, v2)
                if t1_data is None or t2_data is None:
                    continue

                t1_path = os.path.join(data_dir, f"{subj}_{t1_data['image_id']}.nii.gz")
                t2_path = os.path.join(data_dir, f"{subj}_{t2_data['image_id']}.nii.gz")

                if verify_files:
                    t1_ok = os.path.exists(t1_path) and os.path.getsize(t1_path) > 0
                    t2_ok = os.path.exists(t2_path) and os.path.getsize(t2_path) > 0
                    if not (t1_ok and t2_ok):
                        missing += 1
                        continue

                self.samples.append({
                    "initial_path": t1_path,
                    "followup_path": t2_path,
                    "label": t2_data["label"],   # label at follow-up
                    "subject": subj,
                    "pair_type": f"{v1}_{v2}",
                })
                labels.append(t2_data["label"])
                subjects.append(subj)
                pair_counts[f"{v1}->{v2}"] += 1

        self.labels = np.array(labels)
        self.subjects = np.array(subjects)
        n_pos = int(self.labels.sum())
        n_neg = len(self.labels) - n_pos
        n_subjects = len(set(self.subjects))

        print("\nMULTI-TIMEPOINT LONGITUDINAL DATASET")
        print(f"  Total pairs: {len(self.samples)} | Missing: {missing}")
        print(f"  Unique subjects: {n_subjects}")
        print(f"  Labels: Non-converter={n_neg}, Converter={n_pos}")
        print("  Pair breakdown:")
        for pair_type, count in sorted(pair_counts.items()):
            print(f"    {pair_type}: {count}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        t1_img = _load_nifti_volume(s["initial_path"])
        t2_img = _load_nifti_volume(s["followup_path"])
        label = torch.tensor([s["label"]], dtype=torch.float32)

        if self.transform is not None:
            subject = tio.Subject(
                initial=tio.ScalarImage(tensor=t1_img),
                followup=tio.ScalarImage(tensor=t2_img),
            )
            transformed = self.transform(subject)
            t1_img = transformed["initial"].data
            t2_img = transformed["followup"].data

        return t1_img, t2_img, label

    # ---- subject-level split ------------------------------------------------

    def get_subject_level_split_indices(
        self, n_splits: int = 5, random_state: int = 42
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Subject-level stratified K-fold.

        Strata are computed as the majority label across each subject's pairs.
        Returns list of (train_indices, val_indices) into self.samples.
        """
        unique_subjects = list(set(self.subjects.tolist()))
        subject_labels = []
        for subj in unique_subjects:
            mask = self.subjects == subj
            subject_labels.append(int(np.round(self.labels[mask].mean())))
        subject_labels = np.array(subject_labels)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state)
        folds = []
        for tr_idx, va_idx in skf.split(unique_subjects, subject_labels):
            tr_set = {unique_subjects[i] for i in tr_idx}
            va_set = {unique_subjects[i] for i in va_idx}
            tr = [i for i, s in enumerate(self.subjects) if s in tr_set]
            va = [i for i, s in enumerate(self.subjects) if s in va_set]
            folds.append((tr, va))
        return folds


# ---------------------------------------------------------------------------
# Subset helpers used by training to materialise per-fold train/val datasets
# ---------------------------------------------------------------------------

def subset_cross_sectional(
    parent: CrossSectionalDataset,
    indices: Iterable[int],
    csv_path: str,
    data_dir: str,
    is_training: bool,
) -> CrossSectionalDataset:
    """Build a child CrossSectionalDataset restricted to the given indices."""
    indices = list(indices)
    child = CrossSectionalDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        is_training=is_training,
        verify_files=False,
    )
    child.samples = [parent.samples[i] for i in indices]
    child.labels = parent.labels[indices]
    return child


def subset_longitudinal(
    parent: MultiTimepointLongitudinalDataset,
    indices: Iterable[int],
    csv_path: str,
    data_dir: str,
    visit_pairs: Sequence[Tuple[str, str, int]],
    is_training: bool,
) -> MultiTimepointLongitudinalDataset:
    """Build a child MultiTimepointLongitudinalDataset restricted to indices."""
    indices = list(indices)
    child = MultiTimepointLongitudinalDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        visit_pairs=visit_pairs,
        is_training=is_training,
        verify_files=False,
    )
    child.samples = [parent.samples[i] for i in indices]
    child.labels = parent.labels[indices]
    child.subjects = parent.subjects[indices]
    return child

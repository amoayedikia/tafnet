"""
Longitudinal pair dataset driven by an explicit, pre-computed label file.

Replaces MultiTimepointLongitudinalDataset for ADNI. That class derives its
label from the collection CSV's `Group` column:

    label_map = {"CN": 0, "AD": 1, "MCI": 0, "EMCI": 0, "LMCI": 0, "SMC": 0}
    ...
    "label": t2_data["label"],   # label at follow-up

`Group` is the enrolment research group and is constant across a subject's
visits (verified: 0 of 971 subjects change Group), so "label at follow-up" is
just the enrolment label and the trained task was AD-enrolled vs rest — not
MCI-to-dementia conversion. See claude/tafnet-critique-evidence.md §2.

This class instead consumes the output of `label_adni_pairs.py`, which derives
conversion from DXSUM.csv the way Section 3.1 describes: MCI at baseline,
forward search for dementia within a 36-month horizon, censored negatives
excluded, pairs whose diagnosis precedes the follow-up scan dropped.

Expected CSV columns (extra columns are carried through, not required):
    subject, scan1, scan2, label            [required]
    interval_days, days_to_conversion, mmse_slope_per_year, etiology  [optional]

Volumes are read from `<data_dir>/<subject>_<scan>.nii.gz`.
"""
from __future__ import annotations

import copy
import os
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torchio as tio
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

from .transforms import build_train_transform


class VolumeLoadError(RuntimeError):
    """Raised when a volume cannot be read. Never silently substituted."""


def load_volume_strict(path: str) -> torch.Tensor:
    """
    Load a NIfTI as (1, D, H, W), min-max normalised to [0,1] if needed.

    Unlike `datasets._load_nifti_volume`, this raises instead of returning an
    all-zero volume. A blank brain that keeps its label is a silent label error
    that inflates or deflates AUC with no trace in the logs (audit B2).
    """
    import nibabel as nib

    try:
        data = nib.load(path).get_fdata(dtype=np.float32)
    except Exception as err:  # noqa: BLE001
        raise VolumeLoadError(f"{path}: {type(err).__name__}: {err}") from err
    if data.size == 0 or not np.isfinite(data).all():
        raise VolumeLoadError(f"{path}: empty or non-finite volume")
    if data.max() > 1.0:
        rng = data.max() - data.min()
        data = (data - data.min()) / (rng + 1e-8)
    return torch.from_numpy(data).unsqueeze(0)


class LabelledPairDataset(Dataset):
    """
    Longitudinal scan pairs with labels read from a pre-computed CSV.

    Yields (initial_scan, followup_scan, label). `label` is the pair's
    conversion label: 1 = pMCI, 0 = sMCI.
    """

    def __init__(
        self,
        pairs_csv: str,
        data_dir: str,
        is_training: bool = False,
        verify_files: bool = True,
        label_col: str = "label",
        quiet: bool = False,
    ) -> None:
        self.pairs_csv = pairs_csv
        self.data_dir = data_dir
        self.is_training = is_training
        self.label_col = label_col
        self.transform = build_train_transform() if is_training else None

        df = pd.read_csv(pairs_csv)
        required = {"subject", "scan1", "scan2", label_col}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"{pairs_csv} is missing columns: {sorted(missing_cols)}")

        df = df.dropna(subset=[label_col])
        df[label_col] = df[label_col].astype(int)
        bad = set(df[label_col].unique()) - {0, 1}
        if bad:
            raise ValueError(f"{pairs_csv}: labels must be 0/1, found {sorted(bad)}")

        carry = [c for c in ("interval_days", "days_to_conversion",
                             "mmse_slope_per_year", "etiology", "date1", "date2")
                 if c in df.columns]

        self.samples: List[dict] = []
        labels: List[int] = []
        subjects: List[str] = []
        missing = 0

        for row in df.itertuples(index=False):
            subj = str(getattr(row, "subject"))
            t1 = os.path.join(data_dir, f"{subj}_{getattr(row, 'scan1')}.nii.gz")
            t2 = os.path.join(data_dir, f"{subj}_{getattr(row, 'scan2')}.nii.gz")
            if verify_files and not (
                os.path.exists(t1) and os.path.getsize(t1) > 0
                and os.path.exists(t2) and os.path.getsize(t2) > 0
            ):
                missing += 1
                continue
            rec = {"initial_path": t1, "followup_path": t2,
                   "label": int(getattr(row, label_col)), "subject": subj}
            for c in carry:
                rec[c] = getattr(row, c)
            self.samples.append(rec)
            labels.append(rec["label"])
            subjects.append(subj)

        if not self.samples:
            raise RuntimeError(
                f"No usable pairs from {pairs_csv} with data_dir={data_dir}. "
                f"{missing} pair(s) had a missing volume — check --data-dir."
            )

        self.labels = np.asarray(labels)
        self.subjects = np.asarray(subjects)

        if not quiet:
            self._report(missing)

    # ---- reporting ---------------------------------------------------------

    def _report(self, missing: int) -> None:
        n_pos = int(self.labels.sum())
        n_neg = len(self.labels) - n_pos
        pos_subj = {s for s, y in zip(self.subjects, self.labels) if y == 1}
        neg_subj = set(self.subjects.tolist()) - pos_subj
        print("\nLABELLED PAIR DATASET")
        print(f"  Source          : {self.pairs_csv}")
        print(f"  Pairs           : {len(self.samples):,}"
              + (f"   (skipped {missing:,} with a missing volume)" if missing else ""))
        print(f"  Labels (pairs)  : sMCI={n_neg:,}  pMCI={n_pos:,}  "
              f"({100*n_pos/len(self.labels):.1f}% positive)")
        print(f"  Subjects        : {len(pos_subj | neg_subj):,}  "
              f"(converter={len(pos_subj):,}, stable-only={len(neg_subj):,})")
        both = sum(1 for s in pos_subj
                   if 0 in self.labels[self.subjects == s].tolist())
        if both:
            print(f"  NOTE            : {both} subject(s) contribute both a "
                  f"positive and a negative pair — expected under per-pair "
                  f"horizon anchoring; folds still split on subject.")

    # ---- torch Dataset -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        t1_img = load_volume_strict(s["initial_path"])
        t2_img = load_volume_strict(s["followup_path"])
        label = torch.tensor([s["label"]], dtype=torch.float32)

        if self.transform is not None:
            # One tio.Subject so both volumes get identical sampled parameters
            subject = tio.Subject(
                initial=tio.ScalarImage(tensor=t1_img),
                followup=tio.ScalarImage(tensor=t2_img),
            )
            transformed = self.transform(subject)
            t1_img = transformed["initial"].data
            t2_img = transformed["followup"].data

        return t1_img, t2_img, label

    # ---- splits ------------------------------------------------------------

    def _subject_strata(self, subjects: Sequence[str]) -> np.ndarray:
        """A subject is a converter if ANY of its pairs is positive."""
        return np.array([int(self.labels[self.subjects == s].max())
                         for s in subjects])

    def get_subject_level_split_indices(
        self, n_splits: int = 5, random_state: int = 42,
        subjects_subset: Sequence[str] | None = None,
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Subject-level stratified K-fold. No subject spans train and val.

        Strata use max, not majority: a subject with any converter pair is
        stratified as a converter, so the rare class is spread across folds.
        """
        pool = (sorted(set(self.subjects.tolist())) if subjects_subset is None
                else sorted(set(subjects_subset)))
        strata = self._subject_strata(pool)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state)
        folds = []
        for tr_idx, va_idx in skf.split(pool, strata):
            tr_set = {pool[i] for i in tr_idx}
            va_set = {pool[i] for i in va_idx}
            folds.append((
                [i for i, s in enumerate(self.subjects) if s in tr_set],
                [i for i, s in enumerate(self.subjects) if s in va_set],
            ))
        return folds

    def get_holdout_split(
        self, test_frac: float = 0.15, random_state: int = 42,
    ) -> Tuple[List[int], List[str]]:
        """
        Carve a subject-level held-out test set off the front.

        Returns (test_indices, remaining_subjects). Feed the second value to
        `get_subject_level_split_indices(subjects_subset=...)` so CV runs only
        on the remainder. Without this, reported metrics are the best-epoch
        value on the partition used for early stopping (audit B1).
        """
        pool = sorted(set(self.subjects.tolist()))
        strata = self._subject_strata(pool)
        keep, test = train_test_split(
            pool, test_size=test_frac, random_state=random_state, stratify=strata,
        )
        test_set = set(test)
        test_idx = [i for i, s in enumerate(self.subjects) if s in test_set]
        return test_idx, sorted(keep)

    # ---- subsetting --------------------------------------------------------

    def subset(self, indices: Iterable[int],
               is_training: bool = False) -> "LabelledPairDataset":
        """
        Child dataset over the given indices. Does not re-read the CSV or
        re-stat the filesystem, unlike `subset_longitudinal`.
        """
        indices = list(indices)
        child = copy.copy(self)
        child.samples = [self.samples[i] for i in indices]
        child.labels = self.labels[indices]
        child.subjects = self.subjects[indices]
        child.is_training = is_training
        child.transform = build_train_transform() if is_training else None
        return child

    # ---- integrity ---------------------------------------------------------

    def assert_no_subject_leakage(self, *index_sets: Iterable[int]) -> None:
        """Raise if any subject appears in more than one of the given splits."""
        sets = [set(self.subjects[list(ix)].tolist()) for ix in index_sets]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                overlap = sets[i] & sets[j]
                if overlap:
                    raise AssertionError(
                        f"subject leakage between split {i} and {j}: "
                        f"{sorted(overlap)[:5]}"
                        + (" ..." if len(overlap) > 5 else "")
                    )

    def label_summary(self) -> dict:
        """Counts for the paper's cohort table, with matching denominators."""
        pos_subj = {s for s, y in zip(self.subjects, self.labels) if y == 1}
        all_subj = set(self.subjects.tolist())
        return {
            "pairs": len(self.labels),
            "pairs_positive": int(self.labels.sum()),
            "pairs_positive_pct": round(100 * float(self.labels.mean()), 1),
            "subjects": len(all_subj),
            "subjects_converter": len(pos_subj),
            "subjects_stable": len(all_subj - pos_subj),
            "subjects_positive_pct": round(100 * len(pos_subj) / len(all_subj), 1),
            "etiology": dict(Counter(
                s.get("etiology") for s in self.samples if s["label"] == 1
            )) if "etiology" in (self.samples[0] if self.samples else {}) else {},
        }


class LabelledVolumeDataset(Dataset):
    """
    Single-volume dataset from an explicit label CSV. Used for Phase 4 encoder
    pretraining (CN vs dementia).

    Replaces `CrossSectionalDataset` for ADNI, which labels from the collection
    CSV's enrolment `Group` column and therefore (a) carries the same defect as
    A1 and (b) offers no guarantee that its subjects are disjoint from the
    longitudinal task cohort — audit A2, the Phase 4 leak.

    The label file is built from `preprocess_list.csv` rows with role ==
    'phase4', whose baseline diagnosis comes from DXSUM. Those 531 subjects are
    disjoint from the task cohort by construction, because the task cohort
    requires baseline MCI. `exclude_subjects` re-checks that at load time
    rather than trusting it.

    Required columns: subject, image_id, label.
    Volumes are read from `<data_dir>/<subject>_<image_id>.nii.gz`.
    """

    def __init__(
        self,
        labels_csv: str,
        data_dir: str,
        is_training: bool = False,
        verify_files: bool = True,
        exclude_subjects: Iterable[str] | None = None,
        quiet: bool = False,
    ) -> None:
        self.labels_csv = labels_csv
        self.data_dir = data_dir
        self.is_training = is_training
        self.transform = build_train_transform() if is_training else None

        df = pd.read_csv(labels_csv)
        required = {"subject", "image_id", "label"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"{labels_csv} is missing columns: {sorted(missing_cols)}")

        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        excluded = (set() if exclude_subjects is None
                    else {str(x) for x in exclude_subjects})
        leaked = excluded & set(df["subject"].astype(str))
        if leaked:
            raise AssertionError(
                f"{len(leaked)} Phase 4 subject(s) also appear in the task cohort — "
                f"this is the audit A2 leak: {sorted(leaked)[:5]}"
            )

        self.samples: List[dict] = []
        labels: List[int] = []
        subjects: List[str] = []
        missing = 0
        for row in df.itertuples(index=False):
            subj = str(getattr(row, "subject"))
            path = os.path.join(data_dir, f"{subj}_{getattr(row, 'image_id')}.nii.gz")
            if verify_files and not (
                os.path.exists(path) and os.path.getsize(path) > 0
            ):
                missing += 1
                continue
            self.samples.append({"path": path, "label": int(getattr(row, "label")),
                                 "subject": subj})
            labels.append(int(getattr(row, "label")))
            subjects.append(subj)

        if not self.samples:
            raise RuntimeError(
                f"No usable volumes from {labels_csv} with data_dir={data_dir} "
                f"({missing} missing)."
            )

        self.labels = np.asarray(labels)
        self.subjects = np.asarray(subjects)

        if not quiet:
            n_pos = int(self.labels.sum())
            print("\nLABELLED VOLUME DATASET (Phase 4)")
            print(f"  Source      : {self.labels_csv}")
            print(f"  Volumes     : {len(self.samples):,}"
                  + (f"   (skipped {missing:,} missing)" if missing else ""))
            print(f"  Labels      : CN={len(self.labels)-n_pos:,}  dementia={n_pos:,}")
            print(f"  Subjects    : {len(set(subjects)):,}")
            if excluded:
                print(f"  Disjoint from the {len(excluded):,}-subject task cohort: verified")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = load_volume_strict(s["path"])
        label = torch.tensor([s["label"]], dtype=torch.float32)
        if self.transform is not None:
            img = self.transform(tio.ScalarImage(tensor=img)).data
        return img, label

    def get_subject_level_split(
        self, test_frac: float = 0.2, random_state: int = 42,
    ) -> Tuple[List[int], List[int]]:
        """Subject-level stratified train/val split. Returns (train_idx, val_idx)."""
        pool = sorted(set(self.subjects.tolist()))
        strata = np.array([int(self.labels[self.subjects == s].max()) for s in pool])
        train_s, val_s = train_test_split(
            pool, test_size=test_frac, random_state=random_state, stratify=strata,
        )
        train_set, val_set = set(train_s), set(val_s)
        return ([i for i, s in enumerate(self.subjects) if s in train_set],
                [i for i, s in enumerate(self.subjects) if s in val_set])

    def subset(self, indices: Iterable[int],
               is_training: bool = False) -> "LabelledVolumeDataset":
        indices = list(indices)
        child = copy.copy(self)
        child.samples = [self.samples[i] for i in indices]
        child.labels = self.labels[indices]
        child.subjects = self.subjects[indices]
        child.is_training = is_training
        child.transform = build_train_transform() if is_training else None
        return child

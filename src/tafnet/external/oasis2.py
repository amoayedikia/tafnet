"""
OASIS-2 external-validation helpers for TAFNet.

These functions wrap the *existing* core modules so the OASIS-2 zero-shot and
fine-tuning scripts stay thin:

    - load_full_checkpoint   : load a FULL TAFNet state_dict (e.g.
                               tafnet_v4_fold1_best.pth) into a TAFNet, robust
                               to a few common wrapping conventions. NOTE: this
                               differs from TAFNet.load_pretrained_encoder,
                               which loads *encoder-only* weights — the OASIS-2
                               checkpoint is the whole model.
    - build_tafnet_from_cfg  : construct a TAFNet matching the architecture
                               block of a loaded config.
    - build_oasis2_dataset   : construct a MultiTimepointLongitudinalDataset on
                               an OASIS-2 inventory CSV. OASIS-2 ships visits
                               v1..v5 and CDR-derived Group labels (CN/MCI/AD);
                               the dataset's visit normaliser passes v1..v5
                               through unchanged and labels each pair by its
                               follow-up Group, exactly as for ADNI.
    - run_inference          : forward pass over a loader -> (y_true, y_proba).
    - pos_weight_from_labels : neg/pos ratio for class-weighted BCE.

Expected OASIS-2 inventory CSV schema (same columns the core dataset reads for
ADNI):

    Subject, Visit, Image Data ID, Group

where Visit in {v1, v2, v3, v4, v5}, Group in {CN, MCI, AD} (CDR-derived), and
each preprocessed NIfTI is named ``{Subject}_{Image Data ID}.nii.gz`` inside
``data_dir``.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import MultiTimepointLongitudinalDataset
from ..models import TAFNet


# All forward visit pairs across the OASIS-2 v1..v5 timepoints. The third
# element (minimum-month gap) is metadata only — the core dataset ignores it
# when pairing — so it is set to 0 here because OASIS-2 inter-visit intervals
# vary per subject and are not encoded in the visit code.
DEFAULT_OASIS2_VISIT_PAIRS: List[Tuple[str, str, int]] = [
    ("v1", "v2", 0), ("v1", "v3", 0), ("v1", "v4", 0), ("v1", "v5", 0),
    ("v2", "v3", 0), ("v2", "v4", 0), ("v2", "v5", 0),
    ("v3", "v4", 0), ("v3", "v5", 0),
    ("v4", "v5", 0),
]


def load_full_checkpoint(
    model: TAFNet,
    checkpoint_path: str,
    device: str = "cpu",
    strict: bool = True,
) -> TAFNet:
    """
    Load a FULL TAFNet state_dict into ``model`` (in place) and return it.

    Handles three storage conventions:
        1. a bare ``state_dict``  (the tafnet_v4_fold1_best.pth case),
        2. ``{"model_state_dict": ...}``,
        3. ``{"state_dict": ...}``.

    A leading ``"module."`` prefix (from DataParallel) is stripped if present.
    """
    obj = torch.load(checkpoint_path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj

    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
             for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"  [warn] missing keys when loading checkpoint: {list(missing)}")
    if unexpected:
        print(f"  [warn] unexpected keys when loading checkpoint: {list(unexpected)}")
    print(f"  Loaded FULL TAFNet checkpoint from: {checkpoint_path}")
    return model


def build_tafnet_from_cfg(
    cfg,
    use_longitudinal: bool = True,
    freeze_encoder: bool = False,
) -> TAFNet:
    """
    Build a TAFNet whose architecture matches ``cfg.architecture``.

    The defaults mirror configs/default.yaml so the model shape lines up with
    the ADNI-trained checkpoint:
        encoder_channels=[16,32,64,128,128], use_dcca=true, feature_dim=128,
        num_heads=8, dropout=0.3.
    """
    arch = cfg.architecture
    return TAFNet(
        encoder_channels=tuple(arch.encoder_channels),
        use_dcca=bool(arch.use_dcca),
        feature_dim=int(arch.feature_dim),
        num_heads=int(arch.num_heads),
        dropout=float(arch.dropout),
        use_longitudinal=use_longitudinal,
        freeze_encoder=freeze_encoder,
    )


def build_oasis2_dataset(
    csv_path: str,
    data_dir: str,
    visit_pairs: Sequence[Tuple[str, str, int]] | None = None,
    is_training: bool = False,
    verify_files: bool = True,
) -> MultiTimepointLongitudinalDataset:
    """Construct the OASIS-2 longitudinal dataset (defaults to all v1..v5 pairs)."""
    if visit_pairs is None:
        visit_pairs = DEFAULT_OASIS2_VISIT_PAIRS
    return MultiTimepointLongitudinalDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        visit_pairs=list(visit_pairs),
        is_training=is_training,
        verify_files=verify_files,
    )


@torch.no_grad()
def run_inference(
    model: TAFNet,
    loader: DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run ``model`` over ``loader`` and return (y_true, y_pred_proba) as 1-D arrays.

    The classifier emits logits, so a sigmoid is applied here to obtain
    probabilities. The model is set to eval mode for the duration of the call.
    """
    was_training = model.training
    model.eval()
    model.to(device)

    y_true: List[float] = []
    y_proba: List[float] = []
    for t1, t2, label in loader:
        t1 = t1.to(device)
        t2 = t2.to(device)
        logits = model(t1, t2)
        proba = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        y_proba.extend(proba.tolist())
        y_true.extend(label.detach().cpu().numpy().flatten().tolist())

    if was_training:
        model.train()
    return np.asarray(y_true, dtype=float), np.asarray(y_proba, dtype=float)


def pos_weight_from_labels(labels: Sequence[int]) -> float:
    """Return neg/pos ratio for BCEWithLogitsLoss ``pos_weight`` (>=1 typical)."""
    labels = np.asarray(list(labels)).astype(int).flatten()
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0:
        return 1.0
    return float(n_neg) / float(n_pos)

"""
Individual preprocessing step functions, ported from
ADNI_Complete_Preprocessing_Pipeline_v5.ipynb.

Five steps, applied in order:
    1. Brain extraction (ANTsPyNet T1 model)
    2. Spatial normalisation (ANTs registration to MNI152, SyNRA default)
    3. Intensity normalisation (min-max | percentile | zscore)
    4. Light Gaussian denoising (sigma=0.5 by default)
    5. Centre-crop / pad to 128 x 128 x 128

Importing this module triggers `import ants` and `import antspynet`, which
take a few seconds. If those aren't installed, importing will fail loudly.
"""
from __future__ import annotations

import os
import re
from typing import Tuple

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Path / ID parsing
# ---------------------------------------------------------------------------

def get_subject_id_from_adni_path(filepath: str, adni_root_basename: str) -> str:
    """
    Extract the top-level ADNI subject folder name (e.g. '136_S_0429') from
    a deeply nested path like .../ADNI/136_S_0429/MPR.../date/series/file.nii.
    """
    parts = os.path.normpath(filepath).split(os.sep)
    try:
        idx = parts.index(adni_root_basename)
        return parts[idx + 1]
    except (ValueError, IndexError):
        return os.path.basename(filepath).replace(".nii.gz", "").replace(".nii", "")


def extract_image_id_from_filename(filepath: str) -> str:
    """
    Extract the ADNI Image ID (e.g. 'I40657') from a filename. Falls back to
    a date string YYYYMMDD if no Image ID pattern is present.
    """
    filename = os.path.basename(filepath)
    m = re.search(r"(I\d+)", filename)
    if m:
        return m.group(1)
    parts = filepath.split(os.sep)
    for part in parts:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", part)
        if m:
            return m.group(1).replace("-", "")
    return "scan"


# ---------------------------------------------------------------------------
# Step 1: Brain extraction (ANTsPyNet)
# ---------------------------------------------------------------------------

def step1_brain_extraction(input_path: str, output_path: str) -> str:
    """
    Brain-extract a T1 MRI using ANTsPyNet's deep-learning model.

    First call downloads ~50 MB of model weights to a local cache (one-time
    per VM). Threshold the probability map at 0.5 to obtain the binary mask.
    """
    import ants
    import antspynet

    image = ants.image_read(input_path)
    prob_mask = antspynet.brain_extraction(image, modality="t1", verbose=False)
    brain_mask = ants.threshold_image(
        prob_mask, low_thresh=0.5, high_thresh=1.0, inval=1, outval=0,
    )
    brain_image = image * brain_mask
    ants.image_write(brain_image, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Step 2: Spatial normalisation (ANTs)
# ---------------------------------------------------------------------------

def step2_spatial_normalization(input_path: str, template, reg_type: str = "SyNRA"):
    """Register a brain-extracted volume to the MNI152 template."""
    import ants

    moving = ants.image_read(input_path)
    registration = ants.registration(
        fixed=template, moving=moving, type_of_transform=reg_type,
    )
    return registration["warpedmovout"]


# ---------------------------------------------------------------------------
# Step 3: Intensity normalisation
# ---------------------------------------------------------------------------

def step3_intensity_normalization(data: np.ndarray, method: str = "minmax"
                                  ) -> np.ndarray:
    """
    Normalise intensities to [0, 1] using brain-voxel statistics.

    `method` ∈ {"minmax", "percentile", "zscore"}. minmax matches JDAC.
    """
    brain_mask = data > 0
    brain_voxels = data[brain_mask]
    if len(brain_voxels) == 0:
        return data

    if method == "minmax":
        vmin, vmax = brain_voxels.min(), brain_voxels.max()
        if vmax - vmin < 1e-8:
            return np.zeros_like(data)
        normalized = (data - vmin) / (vmax - vmin)
    elif method == "percentile":
        vmin = np.percentile(brain_voxels, 1)
        vmax = np.percentile(brain_voxels, 99)
        if vmax - vmin < 1e-8:
            return np.zeros_like(data)
        normalized = (data - vmin) / (vmax - vmin)
    elif method == "zscore":
        mean = brain_voxels.mean()
        std = brain_voxels.std()
        if std < 1e-8:
            return np.zeros_like(data)
        normalized = (data - mean) / std
        normalized = normalized - normalized.min()
        normalized = normalized / normalized.max()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    normalized = np.clip(normalized, 0, 1)
    normalized[~brain_mask] = 0
    return normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4: Gaussian denoising
# ---------------------------------------------------------------------------

def step4_denoising(data: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Light Gaussian denoising; preserves edges at sigma=0.5."""
    brain_mask = data > 0
    denoised = ndimage.gaussian_filter(data, sigma=sigma)
    denoised[~brain_mask] = 0
    return denoised.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5: Centre crop / pad to fixed shape
# ---------------------------------------------------------------------------

def step5_resample_volume(data: np.ndarray,
                          target_size: Tuple[int, int, int] = (128, 128, 128)
                          ) -> np.ndarray:
    """Centre-crop or zero-pad each axis to match target_size exactly."""
    current = np.array(data.shape)
    target = np.array(target_size)
    diff = target - current

    if np.any(diff > 0):
        pad_before = np.maximum(diff // 2, 0)
        pad_after = np.maximum(diff - pad_before, 0)
        data = np.pad(
            data, list(zip(pad_before, pad_after)),
            mode="constant", constant_values=0,
        )

    current = np.array(data.shape)
    if np.any(current > target):
        start = (current - target) // 2
        end = start + target
        data = data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    return data.astype(np.float32)


def estimate_noise_jdac(data: np.ndarray) -> float:
    """Noise estimate sigma_e ~= sqrt(Var(grad x)) (JDAC method)."""
    gx = np.diff(data, axis=0)
    gy = np.diff(data, axis=1)
    gz = np.diff(data, axis=2)
    return float(np.sqrt((np.var(gx) + np.var(gy) + np.var(gz)) / 3))

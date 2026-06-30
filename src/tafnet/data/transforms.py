"""TorchIO transform builders shared by all datasets."""
from __future__ import annotations

import torchio as tio


def build_train_transform() -> tio.Compose:
    """
    Light geometric + intensity augmentation used during training.

    Matched to the v4 Comprehensive notebook exactly: LR flip, small affine
    perturbation, low-amplitude noise, light Gaussian blur. These are safe
    for 3D T1-weighted structural MRI.
    """
    return tio.Compose([
        tio.RandomFlip(axes=("LR",)),
        tio.RandomAffine(scales=0.05, degrees=5, translation=2),
        tio.RandomNoise(std=0.02),
        tio.RandomBlur(std=(0, 0.5)),
    ])

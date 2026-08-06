"""Seeding and device selection."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: cuda  ({name}, {mem_gb:.1f} GB)")
        return "cuda"
    print("Device: cpu  (no CUDA GPU detected)")
    return "cpu"

"""
External-validation helpers.

Currently provides OASIS-2 zero-shot transfer and warm-start fine-tuning
utilities that reuse the core TAFNet dataset, model, and metrics modules.
"""
from .oasis2 import (
    DEFAULT_OASIS2_VISIT_PAIRS,
    build_oasis2_dataset,
    build_tafnet_from_cfg,
    load_full_checkpoint,
    pos_weight_from_labels,
    run_inference,
)

__all__ = [
    "DEFAULT_OASIS2_VISIT_PAIRS",
    "build_oasis2_dataset",
    "build_tafnet_from_cfg",
    "load_full_checkpoint",
    "pos_weight_from_labels",
    "run_inference",
]

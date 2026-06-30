"""Preprocessing: DICOM->NIfTI plus the 5-step ADNI pipeline."""
from .dicom_to_nifti import convert_all, convert_series, find_dicom_series_dirs
from .pipeline import (
    find_input_scans,
    load_mni_template,
    preprocess_single_scan,
    run_pipeline,
)

__all__ = [
    "convert_all",
    "convert_series",
    "find_dicom_series_dirs",
    "find_input_scans",
    "load_mni_template",
    "preprocess_single_scan",
    "run_pipeline",
]

"""
Shared paths and loaders for the analysis scripts.

Every script reads its inputs from the locations below, set by environment
variables, and writes to TAFNET_ANALYSIS_OUT. No data is stored in this
repository.

    TAFNET_RESULTS       output_dir of scripts/02_train.py. Must contain
                         folds/<method>_fold<k>.json; for the ablation and gate
                         scripts also ablation/<variant>/folds/ and
                         gate_analysis/ (from scripts/06_gate_analysis.py).
    TAFNET_PAIRS_CSV     labelled pairs from scripts/label_adni_pairs.py
    TAFNET_SIDECAR_CSV   one row per scan: image_id, vendor, series
                         (only for subgroups.py)
    TAFNET_ANALYSIS_OUT  where tables and figures are written (default ./analysis_out)
"""
import json, os
import numpy as np

RESULTS = os.environ.get("TAFNET_RESULTS", "results")
PAIRS_CSV = os.environ.get("TAFNET_PAIRS_CSV", "adni_pairs_ready_pair.csv")
SIDECAR_CSV = os.environ.get("TAFNET_SIDECAR_CSV", "sidecar_summary.csv")
OUT = os.environ.get("TAFNET_ANALYSIS_OUT", "analysis_out")
METHODS = ["ResNet3D-18", "DenseNet3D-121", "Siamese-Subtract",
           "CNN-LSTM", "TAFNet-InitialOnly", "TAFNet-Full"]


def out_path(*parts):
    """Path under TAFNET_ANALYSIS_OUT; the directory is created if needed."""
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    return p


def fold_json(method, k, variant=None):
    """Per-fold result file of the main run, or of one ablation variant."""
    d = (os.path.join(RESULTS, "folds") if variant is None
         else os.path.join(RESULTS, "ablation", variant, "folds"))
    return os.path.join(d, f"{method}_fold{k}.json")


def load_oof_and_heldout(methods=METHODS):
    """
    Out-of-fold and held-out predictions assembled from the fold files.

    Returns (oof, bundle):
      oof[m]    = {"y_true": [...], "y_pred": [...]}   validation pairs, folds 1..5 in order
      bundle    = {"test_y_true": [...], "test_y_pred_per_fold": {m: 5 x n_test}}
    """
    oof, per_fold, y_test = {}, {}, None
    for m in methods:
        yt, yp, ft = [], [], []
        for k in range(1, 6):
            j = json.load(open(fold_json(m, k)))
            yt += list(j["predictions"]["y_true"]); yp += list(j["predictions"]["y_pred"])
            ft.append(list(j["test_y_pred"]))
            if y_test is None:
                y_test = list(j["test_y_true"])
            assert np.array_equal(np.asarray(j["test_y_true"], float), np.asarray(y_test, float))
        oof[m] = {"y_true": yt, "y_pred": yp}; per_fold[m] = ft
    return oof, {"test_y_true": y_test, "test_y_pred_per_fold": per_fold}

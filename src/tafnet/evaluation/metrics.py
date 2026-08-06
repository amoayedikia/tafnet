"""
Binary classification metrics for MCI-to-AD conversion.

`compute_all_metrics`     — AUC, sensitivity, specificity, F1, accuracy, etc.
`aggregate_fold_metrics`  — mean and std across CV folds, with totals.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_all_metrics(
    y_true: Iterable, y_pred_proba: Iterable, threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute AUC, sensitivity, specificity, F1, accuracy, precision, NPV, CM."""
    y_true = np.asarray(list(y_true)).flatten()
    y_pred_proba = np.asarray(list(y_pred_proba)).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        return {
            "AUC": 0.5, "Sensitivity": 0.0, "Specificity": 0.0,
            "F1": 0.0, "Accuracy": 0.0, "Precision": 0.0, "NPV": 0.0,
            "TP": 0, "TN": 0, "FP": 0, "FN": 0,
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = 0.5

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "AUC": float(auc),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "F1": float(f1),
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "NPV": float(npv),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def aggregate_fold_metrics(fold_metrics_list: List[Dict]) -> Dict:
    """Mean ± std across folds, plus per-fold list and total confusion matrix."""
    keys = ["AUC", "Sensitivity", "Specificity",
            "F1", "Accuracy", "Precision", "NPV"]
    aggregated: Dict = {}
    for key in keys:
        values = [m[key] for m in fold_metrics_list]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))
        aggregated[f"{key}_per_fold"] = [float(v) for v in values]

    aggregated["TP_total"] = int(sum(m["TP"] for m in fold_metrics_list))
    aggregated["TN_total"] = int(sum(m["TN"] for m in fold_metrics_list))
    aggregated["FP_total"] = int(sum(m["FP"] for m in fold_metrics_list))
    aggregated["FN_total"] = int(sum(m["FN"] for m in fold_metrics_list))
    return aggregated

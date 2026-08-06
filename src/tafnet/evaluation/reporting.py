"""
Results summary, ROC plots, and AUC box plots.

Three entry points, all called from scripts/02_train.py after CV completes:
    generate_results_summary  — printable table, statistical tests, JSON dump
    plot_roc_curves           — one figure with all methods
    plot_boxplots             — AUC distribution box plot
"""
from __future__ import annotations

import json
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import roc_auc_score, roc_curve


def generate_results_summary(
    all_results: Dict, all_predictions: Dict, output_dir: str,
) -> Dict:
    """Print summary table + statistical comparison to TAFNet-Full, save JSON."""
    print("\n" + "=" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 70)

    header = ("\n  Method                  | AUC           | Sens          | "
              "Spec          | F1            | Acc")
    print(header)
    print("  " + "-" * 105)
    for method, res in all_results.items():
        print(f"  {method:24s} | "
              f"{res['AUC_mean']:.3f}±{res['AUC_std']:.3f}  | "
              f"{res['Sensitivity_mean']:.3f}±{res['Sensitivity_std']:.3f}  | "
              f"{res['Specificity_mean']:.3f}±{res['Specificity_std']:.3f}  | "
              f"{res['F1_mean']:.3f}±{res['F1_std']:.3f}  | "
              f"{res['Accuracy_mean']:.3f}±{res['Accuracy_std']:.3f}")

    if "TAFNet-Full" in all_results:
        print("\n  Statistical Comparison (TAFNet-Full vs others):")
        tafnet_aucs = all_results["TAFNet-Full"]["AUC_per_fold"]
        for method, res in all_results.items():
            if method == "TAFNet-Full":
                continue
            other_aucs = res["AUC_per_fold"]
            try:
                _, p_wilcox = wilcoxon(tafnet_aucs, other_aucs, alternative="greater")
            except Exception:    # noqa: BLE001
                p_wilcox = 1.0
            try:
                t_stat, p_ttest = ttest_rel(tafnet_aucs, other_aucs)
                p_ttest = p_ttest / 2 if t_stat > 0 else 1 - p_ttest / 2
            except Exception:    # noqa: BLE001
                p_ttest = 1.0
            delta = float(np.mean(tafnet_aucs) - np.mean(other_aucs))
            sig = ("***" if p_wilcox < 0.001
                   else "**" if p_wilcox < 0.01
                   else "*" if p_wilcox < 0.05 else "")
            print(f"    vs {method:20s}: delta={delta:+.4f}, p={p_wilcox:.4f} {sig}")

    json_results = {}
    for method, res in all_results.items():
        json_results[method] = {
            "AUC_mean": float(res["AUC_mean"]),
            "AUC_std": float(res["AUC_std"]),
            "AUC_per_fold": [float(x) for x in res["AUC_per_fold"]],
            "Sensitivity_mean": float(res["Sensitivity_mean"]),
            "Sensitivity_std": float(res["Sensitivity_std"]),
            "Specificity_mean": float(res["Specificity_mean"]),
            "Specificity_std": float(res["Specificity_std"]),
            "F1_mean": float(res["F1_mean"]),
            "F1_std": float(res["F1_std"]),
            "Accuracy_mean": float(res["Accuracy_mean"]),
            "Accuracy_std": float(res["Accuracy_std"]),
            "TP_total": int(res["TP_total"]),
            "TN_total": int(res["TN_total"]),
            "FP_total": int(res["FP_total"]),
            "FN_total": int(res["FN_total"]),
        }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "comprehensive_results_v4.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    # Also dump raw predictions so that ROC curves can be replayed offline
    # (predictions_v4.json keyed by method, values {"y_true": [...], "y_pred": [...]}).
    preds_path = os.path.join(output_dir, "predictions_v4.json")
    json_preds = {
        method: {
            "y_true": [float(v) for v in p["y_true"]],
            "y_pred": [float(v) for v in p["y_pred"]],
        }
        for method, p in all_predictions.items()
    }
    with open(preds_path, "w") as f:
        json.dump(json_preds, f)
    print(f"  Predictions saved: {preds_path}")
    return json_results


def plot_roc_curves(all_predictions: Dict, output_dir: str) -> None:
    """One ROC plot overlaying all methods. TAFNet-Full drawn thicker/solid."""
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_predictions)))

    for idx, (method, preds) in enumerate(all_predictions.items()):
        fpr, tpr, _ = roc_curve(preds["y_true"], preds["y_pred"])
        auc_val = roc_auc_score(preds["y_true"], preds["y_pred"])
        lw = 3 if "TAFNet-Full" in method else 2
        ls = "-" if "TAFNet" in method else "--"
        plt.plot(fpr, tpr, color=colors[idx], lw=lw, linestyle=ls,
                 label=f"{method} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves: MCI-to-AD Conversion Prediction", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    out = os.path.join(output_dir, "roc_curves_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ROC curves saved: {out}")


def plot_boxplots(all_results: Dict, output_dir: str) -> None:
    """AUC box plot across folds, one box per method."""
    methods = list(all_results.keys())
    auc_data = [all_results[m]["AUC_per_fold"] for m in methods]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(auc_data, patch_artist=True, labels=methods)
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("AUC Distribution Across 5-Fold Cross-Validation", fontsize=14)
    ax.set_ylim([0.5, 1.05])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)

    out = os.path.join(output_dir, "auc_boxplot_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Box plots saved: {out}")

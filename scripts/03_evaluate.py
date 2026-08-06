"""
Re-generate the results table, statistical tests, ROC curves, and AUC box plots
from a *completed* training run, without re-running any model.

Reads from <output_dir>:
    comprehensive_results_v4.json   (required; per-method AUC/Sens/Spec/F1/Acc + per-fold AUCs)
    predictions_v4.json             (optional; raw y_true / y_pred per method — needed for ROC)

Writes back into the same <output_dir>:
    roc_curves_comparison.png       (only if predictions file is present)
    auc_boxplot_comparison.png

Typical usage on the GCP VM:
    python scripts/03_evaluate.py --config configs/default.yaml
    python scripts/03_evaluate.py --output-dir /mnt/drive/MyDrive/TAFNet_results_v4
    python scripts/03_evaluate.py --output-dir ./results --no-drive-check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

# Make `src/` importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tafnet.config import load_config                              # noqa: E402
from tafnet.evaluation.reporting import plot_roc_curves, plot_boxplots  # noqa: E402
from tafnet.utils.drive import verify_drive_mount                  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay TAFNet evaluation plots/statistics.")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config; output_dir is taken from paths.output_dir.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory containing comprehensive_results_v4.json. "
                        "Overrides config if both are supplied.")
    p.add_argument("--override", action="append", default=[],
                   help="Dotted config overrides, e.g. --override paths.output_dir=/tmp/run1")
    p.add_argument("--no-drive-check", action="store_true",
                   help="Skip the rclone Drive-mount sanity check.")
    return p.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir is not None:
        return args.output_dir
    if args.config is None:
        raise SystemExit(
            "Must provide either --output-dir or --config "
            "(so output_dir can be read from paths.output_dir)."
        )
    cfg = load_config(args.config, overrides=args.override)
    return cfg.paths.output_dir


def print_summary_table(results: dict) -> None:
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY (from saved JSON)")
    print("=" * 70)
    header = ("\n  Method                  | AUC           | Sens          | "
              "Spec          | F1            | Acc")
    print(header)
    print("  " + "-" * 105)
    for method, res in results.items():
        print(f"  {method:24s} | "
              f"{res['AUC_mean']:.3f}±{res['AUC_std']:.3f}  | "
              f"{res['Sensitivity_mean']:.3f}±{res['Sensitivity_std']:.3f}  | "
              f"{res['Specificity_mean']:.3f}±{res['Specificity_std']:.3f}  | "
              f"{res['F1_mean']:.3f}±{res['F1_std']:.3f}  | "
              f"{res['Accuracy_mean']:.3f}±{res['Accuracy_std']:.3f}")


def print_statistical_comparison(results: dict) -> None:
    if "TAFNet-Full" not in results:
        print("\n  [!] TAFNet-Full not found in results - skipping statistical comparison.")
        return

    print("\n  Statistical Comparison (TAFNet-Full vs others):")
    tafnet_aucs = np.asarray(results["TAFNet-Full"]["AUC_per_fold"], dtype=float)
    for method, res in results.items():
        if method == "TAFNet-Full":
            continue
        other_aucs = np.asarray(res["AUC_per_fold"], dtype=float)
        try:
            _, p_wilcox = wilcoxon(tafnet_aucs, other_aucs, alternative="greater")
        except Exception:    # noqa: BLE001
            p_wilcox = 1.0
        try:
            t_stat, p_two = ttest_rel(tafnet_aucs, other_aucs)
            p_ttest = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        except Exception:    # noqa: BLE001
            p_ttest = 1.0
        delta = float(tafnet_aucs.mean() - other_aucs.mean())
        sig = ("***" if p_wilcox < 0.001
               else "**" if p_wilcox < 0.01
               else "*"  if p_wilcox < 0.05 else "")
        print(f"    vs {method:20s}: delta={delta:+.4f}, "
              f"p_wilcoxon={p_wilcox:.4f}, p_ttest={p_ttest:.4f} {sig}")


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args)

    if not args.no_drive_check:
        verify_drive_mount(output_dir)

    results_path = os.path.join(output_dir, "comprehensive_results_v4.json")
    preds_path   = os.path.join(output_dir, "predictions_v4.json")

    if not os.path.exists(results_path):
        raise SystemExit(
            f"[X] Could not find {results_path}.\n"
            f"    Run scripts/02_train.py first, or pass --output-dir pointing at a completed run."
        )

    print(f"[OK] Loading results from {results_path}")
    with open(results_path) as f:
        results = json.load(f)

    print_summary_table(results)
    print_statistical_comparison(results)

    print("\n[*] Regenerating box plot...")
    plot_boxplots(results, output_dir)

    if os.path.exists(preds_path):
        print(f"[*] Regenerating ROC curves from {preds_path}...")
        with open(preds_path) as f:
            predictions = json.load(f)
        plot_roc_curves(predictions, output_dir)
    else:
        print(f"[!] {preds_path} not found - skipping ROC curves.")
        print("    (predictions_v4.json is only written by training runs from this codebase;")
        print("     older runs may not have it. Re-run training to regenerate.)")

    print("\n[OK] Evaluation replay complete. Artifacts written to:")
    print(f"     {output_dir}")


if __name__ == "__main__":
    main()

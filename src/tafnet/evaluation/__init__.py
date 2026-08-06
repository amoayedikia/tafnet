"""Evaluation: metrics, aggregation, and reporting."""
from .metrics import aggregate_fold_metrics, compute_all_metrics
from .reporting import generate_results_summary, plot_boxplots, plot_roc_curves

__all__ = [
    "aggregate_fold_metrics",
    "compute_all_metrics",
    "generate_results_summary",
    "plot_boxplots",
    "plot_roc_curves",
]

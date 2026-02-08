"""
We compute:
- ROC-AUC (threshold-independent)
- PR-AUC / Average Precision (good for imbalance)
- Accuracy, Precision, Recall, F1 (threshold-dependent)
- Confusion matrix

We also add:
- Threshold tuning: sweep thresholds and pick the best one for a chosen metric (default: F1)
"""

from typing import Dict, Any, Tuple, List

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_binary_metrics(y_true, y_proba, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute common binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels (0/1)
    y_proba : array-like
        Predicted probabilities for class 1 (churn)
    threshold : float
        Probability threshold to convert probabilities into 0/1 predictions.

    Returns
    -------
    dict
        Metrics dictionary (JSON serializable).
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    # Convert probabilities -> predicted labels using threshold
    y_pred = (y_proba >= threshold).astype(int)

    # ROC-AUC can fail if only one class present in y_true
    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc_auc = float("nan")

    pr_auc = float(average_precision_score(y_true, y_proba))

    metrics: Dict[str, Any] = {
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def sweep_thresholds(
    y_true,
    y_proba,
    thresholds: List[float] | None = None,
) -> List[Dict[str, Any]]:
    """
    Compute metrics for many thresholds.

    Returns a list of metric dicts (one per threshold).
    """
    if thresholds is None:
        # 0.05, 0.10, ... 0.95
        thresholds = [round(t, 2) for t in np.linspace(0.05, 0.95, 19)]

    results: List[Dict[str, Any]] = []
    for t in thresholds:
        results.append(compute_binary_metrics(y_true, y_proba, threshold=float(t)))
    return results


def find_best_threshold(
    y_true,
    y_proba,
    metric: str = "f1",
    thresholds: List[float] | None = None,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Find the threshold that maximizes a chosen metric.

    metric: "f1" (default), "precision", or "recall"
    returns: (best_threshold, best_metrics_dict, full_sweep_results)
    """
    metric = metric.lower()
    if metric not in {"f1", "precision", "recall"}:
        raise ValueError("metric must be one of: 'f1', 'precision', 'recall'")

    sweep = sweep_thresholds(y_true, y_proba, thresholds=thresholds)

    best = None
    for row in sweep:
        if best is None or row[metric] > best[metric]:
            best = row

    assert best is not None
    return float(best["threshold"]), best, sweep

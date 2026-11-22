"""
Classification evaluation metrics for Nāḍī-Q.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """
    Compute a comprehensive set of medical classification metrics.
    
    Handles edge cases like one-class validation sets gracefully.
    """
    metrics: dict[str, Any] = {}
    
    # Basic counts
    metrics["n_samples"] = len(y_true)
    metrics["n_positives"] = int(np.sum(y_true == 1))
    metrics["n_negatives"] = int(np.sum(y_true == 0))
    
    # Check if we have only one class
    if len(np.unique(y_true)) < 2:
        warnings.warn("Only one class present in y_true. ROC AUC and PRC AUC are not defined.")
        metrics["roc_auc"] = np.nan
        metrics["prc_auc"] = np.nan
    else:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            metrics["prc_auc"] = float(average_precision_score(y_true, y_prob[:, 1]))
        except ValueError:
            metrics["roc_auc"] = np.nan
            metrics["prc_auc"] = np.nan
            
    # Threshold-based metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    
    # Confusion matrix elements
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # If only one class is present, sklearn returns 1x1 matrix. We manually build it.
        tn = fp = fn = tp = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1: tp += 1
            elif yt == 1 and yp == 0: fn += 1
            elif yt == 0 and yp == 1: fp += 1
            elif yt == 0 and yp == 0: tn += 1
            
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    
    # Derived metrics with safe division
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan  # same as recall
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    metrics["ppv"] = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan  # same as precision
    metrics["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan
    
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    
    return metrics

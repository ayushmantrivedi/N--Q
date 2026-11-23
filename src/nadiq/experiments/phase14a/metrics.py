"""
NĀḌĪ-Q Phase 14A: Evaluation Metrics and Paired Bootstrap Statistics.
Computes AUROC, AUPRC, Brier score, ECE, Sensitivity, Specificity,
and paired bootstrap 95% confidence intervals for statistical superiority testing.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    confusion_matrix,
)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) across uniform confidence bins.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        b_lo, b_hi = bins[i], bins[i + 1]
        mask = (y_prob >= b_lo) & (y_prob < b_hi) if i < n_bins - 1 else (y_prob >= b_lo) & (y_prob <= b_hi)
        if np.sum(mask) > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute full classification diagnostic suite.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)

    # AUROC & AUPRC
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = float(np.mean(y_true))

    # Brier & ECE
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece(y_true, y_prob)

    # Optimal cutoff via Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    # Clip extreme threshold
    best_thresh = float(np.clip(best_thresh, 0.05, 0.95))

    y_pred = (y_prob >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    accuracy = float((tp + tn) / len(y_true))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "ece": ece,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "optimal_threshold": best_thresh,
    }


def compute_paired_bootstrap_ci(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute paired bootstrap difference Delta = Metric(A) - Metric(B) with 95% CI.
    A positive difference indicates Model A is superior to Model B.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    probs_a = np.asarray(probs_a, dtype=float)
    probs_b = np.asarray(probs_b, dtype=float)
    n = len(y_true)

    base_a = float(metric_fn(y_true, probs_a))
    base_b = float(metric_fn(y_true, probs_b))
    base_diff = base_a - base_b

    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        # Ensure both classes are present in sample
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            m_a = float(metric_fn(y_true[idx], probs_a[idx]))
            m_b = float(metric_fn(y_true[idx], probs_b[idx]))
            diffs.append(m_a - m_b)
        except Exception:
            continue

    if len(diffs) < 100:
        return {
            "diff_mean": base_diff,
            "ci_lower": base_diff,
            "ci_upper": base_diff,
            "p_superior": 0.5,
        }

    diffs_arr = np.array(diffs)
    ci_lower = float(np.percentile(diffs_arr, 2.5))
    ci_upper = float(np.percentile(diffs_arr, 97.5))
    p_superior = float(np.mean(diffs_arr > 0.0))

    return {
        "diff_mean": float(np.mean(diffs_arr)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_superior": p_superior,
    }

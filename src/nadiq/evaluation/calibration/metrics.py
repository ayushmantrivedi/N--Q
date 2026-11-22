"""
Calibration metrics for Nāḍī-Q.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """
    Compute calibration metrics: Brier score and Expected Calibration Error (ECE).
    """
    metrics: dict[str, Any] = {}
    
    # Handle single class edge case
    if len(np.unique(y_true)) < 2:
        metrics["brier_score"] = np.nan
        metrics["ece"] = np.nan
        return metrics

    # Brier Score
    prob_pos = y_prob[:, 1]
    metrics["brier_score"] = float(brier_score_loss(y_true, prob_pos))
    
    # Expected Calibration Error (ECE)
    prob_true, prob_pred = calibration_curve(y_true, prob_pos, n_bins=n_bins, strategy='uniform')
    
    # Calculate bin counts to weight the ECE
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(prob_pos, bin_edges, right=True)
    # digitize returns 1-indexed, shift to 0-indexed and handle edge cases
    bin_indices = np.clip(bin_indices - 1, 0, n_bins - 1)
    
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    
    # Filter out empty bins for ECE calculation
    non_empty_mask = bin_counts > 0
    weights = bin_counts[non_empty_mask] / len(y_true)
    
    # ECE is the weighted average of absolute differences between predicted and true probabilities in each bin
    ece = np.sum(weights * np.abs(prob_true - prob_pred))
    metrics["ece"] = float(ece)
    
    # Also return raw curve data for plotting if needed
    metrics["calibration_curve_true"] = prob_true.tolist()
    metrics["calibration_curve_pred"] = prob_pred.tolist()
    
    return metrics

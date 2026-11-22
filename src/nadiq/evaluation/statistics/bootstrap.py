"""
Bootstrap confidence intervals for Nāḍī-Q.
"""

from __future__ import annotations

from typing import Callable, Any

import numpy as np


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_or_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Compute bootstrap confidence intervals for a given metric.
    
    Args:
        y_true: Ground truth labels.
        y_pred_or_prob: Predictions or probabilities.
        metric_fn: A function that takes (y_true, y_pred_or_prob) and returns a float.
        n_bootstrap: Number of bootstrap iterations.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary containing estimate, lower, upper, and n_bootstrap.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    
    scores = []
    
    # Calculate base estimate
    try:
        base_estimate = float(metric_fn(y_true, y_pred_or_prob))
    except Exception:
        base_estimate = np.nan
        
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = rng.integers(0, n_samples, n_samples)
        
        # We need to ensure we don't accidentally sample only one class for metrics that require both
        y_true_boot = y_true[indices]
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        try:
            score = metric_fn(y_true_boot, y_pred_or_prob[indices])
            scores.append(score)
        except Exception:
            continue
            
    if not scores:
        return {
            "estimate": base_estimate,
            "lower": np.nan,
            "upper": np.nan,
            "n_bootstrap": n_bootstrap,
            "successful_bootstraps": 0,
        }
        
    scores_array = np.array(scores)
    alpha = (1.0 - confidence_level) / 2.0
    lower = np.percentile(scores_array, alpha * 100)
    upper = np.percentile(scores_array, (1.0 - alpha) * 100)
    
    return {
        "estimate": base_estimate,
        "lower": float(lower),
        "upper": float(upper),
        "n_bootstrap": n_bootstrap,
        "successful_bootstraps": len(scores),
    }

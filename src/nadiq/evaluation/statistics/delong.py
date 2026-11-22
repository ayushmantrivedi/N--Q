"""
DeLong Test for ROC AUC Comparison.

Implements the fast DeLong algorithm by Sun and Xu (2014):
"Fast Implementation of DeLong's Algorithm for Comparing the Areas
Under Correlated Receiver Operating Characteristic Curves"
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.stats


def compute_midrank(x: np.ndarray) -> np.ndarray:
    """
    Computes midranks.
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fast_delong(preds1: np.ndarray, preds2: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Computes the p-value for the difference between two ROC AUCs using the fast DeLong algorithm.
    
    Args:
        preds1: Probabilities from model 1
        preds2: Probabilities from model 2
        labels: Ground truth labels (0 or 1)
        
    Returns:
        Z-statistic, p-value (two-tailed)
    """
    m = np.sum(labels == 1)
    n = np.sum(labels == 0)
    
    if m == 0 or n == 0:
        raise ValueError("Labels must contain at least one positive and one negative example.")
        
    x_A = preds1[labels == 1]
    y_A = preds1[labels == 0]
    
    x_B = preds2[labels == 1]
    y_B = preds2[labels == 0]
    
    tx_A = compute_midrank(x_A)
    ty_A = compute_midrank(y_A)
    tx_B = compute_midrank(x_B)
    ty_B = compute_midrank(y_B)
    
    tz_A = compute_midrank(preds1)
    tz_B = compute_midrank(preds2)
    
    v0101_A = (tz_A[labels == 1] - tx_A) / n
    v0110_A = 1.0 - (tz_A[labels == 0] - ty_A) / m
    
    v0101_B = (tz_B[labels == 1] - tx_B) / n
    v0110_B = 1.0 - (tz_B[labels == 0] - ty_B) / m
    
    auc_A = np.sum(v0101_A) / m
    auc_B = np.sum(v0101_B) / m
    
    v1010_A = v0101_A - auc_A
    v0110_A_centered = v0110_A - auc_A
    v1010_B = v0101_B - auc_B
    v0110_B_centered = v0110_B - auc_B
    
    # Covariance matrix calculation
    s10 = np.cov(v1010_A, v1010_B, rowvar=True, bias=True)
    s01 = np.cov(v0110_A_centered, v0110_B_centered, rowvar=True, bias=True)
    
    # s10 and s01 are 2x2 covariance matrices
    S = s10 / m + s01 / n
    
    diff = auc_A - auc_B
    variance = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    
    if variance <= 0:
        if np.isclose(diff, 0.0):
            return 0.0, 1.0
        return np.inf * np.sign(diff), 0.0
        
    z = diff / np.sqrt(variance)
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z)))
    
    return float(z), float(p_value)

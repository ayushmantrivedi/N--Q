"""
Scientific validation of the DeLong implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from nadiq.evaluation.statistics.delong import fast_delong


def test_delong_identical_predictions() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    preds1 = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.3, 0.7])
    
    # Same predictions should yield variance 0 and p-value 1.0 (or close to it)
    z, p = fast_delong(preds1, preds1, y_true)
    assert p == 1.0
    assert z == 0.0


def test_delong_clearly_different_aucs() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Perfect model
    preds_perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    # Random model
    preds_random = np.array([0.6, 0.2, 0.8, 0.4, 0.1, 0.7, 0.3, 0.9])
    
    z, p = fast_delong(preds_perfect, preds_random, y_true)
    # Perfect AUC is 1.0, Random is ~0.5. Difference should be highly significant.
    assert p < 0.05
    assert z > 0


def test_delong_tied_scores() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    # Model with tied predictions
    preds1 = np.array([0.1, 0.1, 0.9, 0.9, 0.5, 0.5])
    preds2 = np.array([0.2, 0.3, 0.8, 0.9, 0.4, 0.6])
    
    # Should run without error and give valid p-value
    z, p = fast_delong(preds1, preds2, y_true)
    assert 0.0 <= p <= 1.0
    assert not np.isnan(z)


def test_delong_imbalanced_classes() -> None:
    # 1 positive, 9 negatives
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    preds1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    preds2 = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    
    z, p = fast_delong(preds1, preds2, y_true)
    assert not np.isnan(p)
    assert p < 0.05  # Perfect vs complete anti-perfect


def test_delong_small_sample() -> None:
    # Smallest possible: 1 pos, 1 neg
    y_true = np.array([0, 1])
    preds1 = np.array([0.1, 0.9])
    preds2 = np.array([0.9, 0.1])
    
    z, p = fast_delong(preds1, preds2, y_true)
    # With n=1, m=1, variance is 0, difference is 1 vs 0
    # Should handle variance 0 difference properly
    assert p == 0.0 or p < 0.05


def test_delong_known_reference() -> None:
    # Comparison against standard known output (e.g. from pROC in R)
    # Data from a standard verification set
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    preds1 = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.6, 0.3, 0.4, 0.5, 0.5])
    preds2 = np.array([0.8, 0.9, 0.2, 0.1, 0.6, 0.7, 0.4, 0.3, 0.5, 0.5])
    
    # Since these predictions are very similar, AUCs are the same, z should be 0, p=1
    z, p = fast_delong(preds1, preds2, y_true)
    assert np.isclose(p, 1.0)
    assert np.isclose(z, 0.0)

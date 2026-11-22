"""
Tests for evaluation metrics and bootstrapping.
"""

from __future__ import annotations

import numpy as np
import pytest

from nadiq.evaluation.calibration.metrics import compute_calibration_metrics
from nadiq.evaluation.classification.metrics import compute_classification_metrics
from nadiq.evaluation.statistics.bootstrap import compute_bootstrap_ci


def test_classification_metrics() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.4, 0.6], [0.2, 0.8], [0.8, 0.2], [0.7, 0.3]])
    
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    
    assert metrics["n_samples"] == 6
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 4 / 6
    assert "roc_auc" in metrics
    assert "prc_auc" in metrics


def test_classification_metrics_single_class() -> None:
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 0])
    y_prob = np.array([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3]])
    
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["prc_auc"])
    assert metrics["tp"] == 2
    assert metrics["fn"] == 1
    assert metrics["tn"] == 0
    assert metrics["fp"] == 0


def test_calibration_metrics() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.4, 0.6], [0.2, 0.8], [0.8, 0.2], [0.7, 0.3]])
    
    metrics = compute_calibration_metrics(y_true, y_prob)
    
    assert "brier_score" in metrics
    assert "ece" in metrics
    assert "calibration_curve_true" in metrics


def test_bootstrap_ci() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6], [0.55, 0.45], [0.45, 0.55]])
    
    from sklearn.metrics import roc_auc_score
    def auc_fn(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(roc_auc_score(yt, yp[:, 1]))
        
    ci = compute_bootstrap_ci(y_true, y_prob, auc_fn, n_bootstrap=50, seed=42)
    
    assert ci["n_bootstrap"] == 50
    assert not np.isnan(ci["estimate"])
    assert not np.isnan(ci["lower"])
    assert not np.isnan(ci["upper"])
    assert ci["lower"] <= ci["estimate"] <= ci["upper"]

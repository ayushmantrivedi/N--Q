"""
Tests for classical baseline models.
"""

from __future__ import annotations

import numpy as np
import pytest

from nadiq.models.classical.gradient_boosting import GradientBoostedTreeModel
from nadiq.models.classical.logistic import LogisticRegressionModel
from nadiq.models.classical.mlp import MLPModel


@pytest.fixture
def dummy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 5))
    y = rng.integers(0, 2, size=100)
    return X, y


def test_logistic_regression(dummy_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = dummy_data
    model = LogisticRegressionModel()
    
    assert not model._is_fitted
    
    report_train = model.fit(X, y)
    assert "training_time" in report_train
    assert model._is_fitted
    
    out = model.predict(X)
    assert out.probabilities is not None
    assert out.probabilities.shape == (100, 2)
    assert out.logits is not None
    
    report = model.resource_report()
    assert report["n_parameters"] > 0
    assert report["is_fitted"]


def test_gradient_boosting(dummy_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = dummy_data
    model = GradientBoostedTreeModel()
    
    model.fit(X, y)
    out = model.predict(X)
    
    assert out.probabilities is not None
    assert out.probabilities.shape == (100, 2)
    
    report = model.resource_report()
    assert report["n_parameters"] == -1  # Placeholder for tree models


def test_mlp(dummy_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = dummy_data
    model = MLPModel(max_iter=10)
    
    model.fit(X, y)
    out = model.predict(X)
    
    assert out.probabilities is not None
    assert out.probabilities.shape == (100, 2)
    
    report = model.resource_report()
    assert report["n_parameters"] > 0

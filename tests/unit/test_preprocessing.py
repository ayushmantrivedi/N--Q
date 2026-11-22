"""
Tests for preprocessing pipeline and transformers.
"""

from __future__ import annotations

import numpy as np
import pytest

from nadiq.preprocessing.filtering import VarianceFilter
from nadiq.preprocessing.imputation import MedianImputer
from nadiq.preprocessing.normalization import ZScoreNormalizer
from nadiq.preprocessing.pipeline import PreprocessingPipeline


def test_zscore_normalizer() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    norm = ZScoreNormalizer()
    norm.fit(X)
    
    Xt = norm.transform(X)
    assert np.allclose(np.mean(Xt, axis=0), 0.0)
    assert np.allclose(np.std(Xt, axis=0), 1.0)
    
    X_inv = norm.inverse_transform(Xt)
    assert np.allclose(X, X_inv)


def test_median_imputer() -> None:
    X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
    imputer = MedianImputer()
    imputer.fit(X)
    
    Xt = imputer.transform(X)
    assert not np.isnan(Xt).any()
    assert Xt[0, 1] == 5.0
    assert Xt[2, 0] == 2.0


def test_variance_filter() -> None:
    X = np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 3.0], [1.0, 6.0, 3.0]])
    filt = VarianceFilter(threshold=0.1)
    filt.fit(X)
    
    Xt = filt.transform(X)
    # Features 0 and 2 have 0 variance and should be dropped
    assert Xt.shape == (3, 1)
    assert np.allclose(Xt[:, 0], X[:, 1])
    
    X_inv = filt.inverse_transform(Xt)
    assert X_inv.shape == (3, 3)
    assert np.allclose(X_inv[:, 1], X[:, 1])
    assert np.allclose(X_inv[:, 0], 0.0)
    assert np.allclose(X_inv[:, 2], 0.0)


def test_preprocessing_pipeline() -> None:
    X = np.array([[1.0, np.nan, 3.0], [1.0, 4.0, 3.0], [5.0, 6.0, 3.0]])
    pipe = PreprocessingPipeline([
        ("imputer", MedianImputer()),
        ("filter", VarianceFilter(threshold=0.1)),
        ("norm", ZScoreNormalizer())
    ])
    
    pipe.fit(X)
    assert pipe.is_fitted_
    
    fp1 = pipe.fingerprint()
    assert isinstance(fp1, str)
    
    Xt = pipe.transform(X)
    assert Xt.shape == (3, 2)  # Col 2 is constant, dropped
    assert not np.isnan(Xt).any()

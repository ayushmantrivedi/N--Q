"""
Unit tests for the synthetic tabular loader.
"""

from __future__ import annotations

import numpy as np
import pytest

from nadiq.data.contracts import DataSplit, FeatureDtype, ModalityType
from nadiq.data.loaders.synthetic import SyntheticTabularLoader


def test_synthetic_loader_deterministic() -> None:
    loader1 = SyntheticTabularLoader(dataset_id="test1", seed=42)
    loader2 = SyntheticTabularLoader(dataset_id="test2", seed=42)
    
    samples1 = list(loader1.load_split(DataSplit.TRAIN))
    samples2 = list(loader2.load_split(DataSplit.TRAIN))
    
    assert len(samples1) == len(samples2)
    
    for s1, s2 in zip(samples1, samples2):
        assert np.allclose(s1.features["tabular"].values, s2.features["tabular"].values)
        assert s1.label.value == s2.label.value


def test_synthetic_loader_different_seeds() -> None:
    loader1 = SyntheticTabularLoader(dataset_id="test1", seed=42)
    loader2 = SyntheticTabularLoader(dataset_id="test2", seed=100)
    
    samples1 = list(loader1.load_split(DataSplit.TRAIN))
    samples2 = list(loader2.load_split(DataSplit.TRAIN))
    
    # Very likely to differ
    diffs = 0
    for s1, s2 in zip(samples1, samples2):
        if not np.allclose(s1.features["tabular"].values, s2.features["tabular"].values):
            diffs += 1
    assert diffs > 0


def test_synthetic_loader_contracts() -> None:
    loader = SyntheticTabularLoader(dataset_id="test", n_samples=100, n_features=10)
    
    schema = loader.get_schema()
    assert schema.dataset_id == "test"
    assert "tabular" in schema.modalities
    
    stats = loader.get_stats()
    assert sum(stats.n_samples.values()) == 100
    
    samples = list(loader.load_split(DataSplit.TEST))
    assert len(samples) > 0
    s = samples[0]
    assert s.metadata.dataset_id == "test"
    assert s.metadata.split == DataSplit.TEST
    assert "tabular" in s.features
    assert s.features["tabular"].modality == ModalityType.TABULAR
    assert s.features["tabular"].dtype == FeatureDtype.CONTINUOUS
    assert len(s.features["tabular"].values) == 10

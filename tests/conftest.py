"""
Pytest configuration and shared fixtures for Nāḍī-Q.

Fixtures defined here are available to all test files.
"""

from __future__ import annotations

import numpy as np
import pytest

from nadiq.data.contracts import (
    ClinicalLabel,
    ClinicalSample,
    DataSplit,
    FeatureBlock,
    FeatureDtype,
    LabelType,
    ModalityType,
    SampleMetadata,
)


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """A seeded random number generator for deterministic test data."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def simple_tabular_block(rng: np.random.Generator) -> FeatureBlock:
    """A minimal valid tabular FeatureBlock for testing."""
    n_features = 16
    values = rng.standard_normal(n_features).astype(np.float32)
    names = [f"feature_{i:02d}" for i in range(n_features)]
    return FeatureBlock(
        modality=ModalityType.TABULAR,
        values=values,
        feature_names=names,
        dtype=FeatureDtype.CONTINUOUS,
    )


@pytest.fixture
def simple_clinical_sample(simple_tabular_block: FeatureBlock) -> ClinicalSample:
    """A minimal valid ClinicalSample for testing."""
    return ClinicalSample(
        sample_id="fixture_sample_001",
        features={"tabular": simple_tabular_block},
        label=ClinicalLabel(
            task="binary_risk",
            value=0,
            label_type=LabelType.BINARY,
            source="synthetic_fixture",
        ),
        metadata=SampleMetadata(
            dataset_id="synthetic_tabular_v1",
            split=DataSplit.TRAIN,
            preprocessing_version="unprocessed",
        ),
    )


@pytest.fixture(autouse=True)
def reset_random_state() -> None:
    """
    Ensure each test starts with a known random seed.

    This autouse fixture prevents random state leakage between tests.
    It seeds numpy, Python random, and (if available) torch before each test.
    """
    from nadiq.utils.seeding import seed_all

    seed_all(0)

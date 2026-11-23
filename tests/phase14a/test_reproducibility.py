"""
Tests for Deterministic Reproducibility.
"""

import numpy as np
import pytest
from nadiq.models.quantum.brqk.config import BRQKConfig
from nadiq.models.quantum.brqk.quantum_kernel import BiologicalRelationalQuantumKernel


def test_reproducibility_identical_seeds():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(8, 6))

    k1 = BiologicalRelationalQuantumKernel(BRQKConfig(seed=42))
    k1.fit(X)
    K1 = k1.compute_gram_matrix(X)

    k2 = BiologicalRelationalQuantumKernel(BRQKConfig(seed=42))
    k2.fit(X)
    K2 = k2.compute_gram_matrix(X)

    assert np.allclose(K1, K2, atol=1e-12)

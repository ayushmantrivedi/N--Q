"""
Tests for Positive Semi-Definiteness (PSD) and Kernel Validation.
"""

import numpy as np
import pytest
from nadiq.models.quantum.brqk.quantum_kernel import BiologicalRelationalQuantumKernel
from nadiq.models.quantum.brqk.kernel_validation import validate_kernel_matrix, regularize_kernel_matrix


def test_kernel_psd_eigenvalues():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 6))

    kernel = BiologicalRelationalQuantumKernel()
    kernel.fit(X)
    K = kernel.compute_gram_matrix(X)

    val = validate_kernel_matrix(K)
    assert val["is_square"] is True
    assert val["is_symmetric"] is True
    assert val["has_unit_diagonal"] is True
    assert val["is_bounded"] is True
    assert val["is_psd"] is True
    assert val["min_eigenval"] >= -1e-6


def test_regularize_kernel_matrix():
    K = np.array([[1.0, 0.9999], [0.9999, 1.0]])
    K_reg = regularize_kernel_matrix(K, ridge=1e-3)
    assert np.isclose(K_reg[0, 0], 1.001)
    assert np.isclose(K_reg[1, 1], 1.001)

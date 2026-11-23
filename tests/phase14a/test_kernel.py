"""
Tests for Biological Relational Quantum Kernel (BRQK) Gram matrix properties.
"""

import numpy as np
import pytest
from nadiq.models.quantum.brqk.config import BRQKConfig
from nadiq.models.quantum.brqk.quantum_kernel import BiologicalRelationalQuantumKernel


def test_kernel_gram_matrix_properties():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(12, 6))

    kernel = BiologicalRelationalQuantumKernel()
    kernel.fit(X)

    K = kernel.compute_gram_matrix(X)

    # 1. Shape check
    assert K.shape == (12, 12)
    # 2. Exact unit diagonal
    assert np.allclose(np.diag(K), 1.0, atol=1e-12)
    # 3. Exact symmetry
    assert np.allclose(K, K.T, atol=1e-12)
    # 4. Values in [0, 1]
    assert np.all(K >= 0.0)
    assert np.all(K <= 1.0 + 1e-12)


def test_kernel_statevector_matches_overlap_qnode():
    """Statevector batch matrix multiplication must match individual overlap QNode to 1e-10."""
    rng = np.random.default_rng(456)
    X = rng.normal(size=(4, 6))

    kernel = BiologicalRelationalQuantumKernel()
    kernel.fit(X)
    K_batch = kernel.compute_gram_matrix(X)

    thetas = kernel.transform_to_angles(X)
    for i in range(4):
        for j in range(4):
            qnode_val = kernel.compute_overlap_qnode(thetas[i], thetas[j])
            assert np.isclose(K_batch[i, j], qnode_val, atol=1e-10)


def test_kernel_evaluation_matrix_shape():
    rng = np.random.default_rng(789)
    X_tr = rng.normal(size=(15, 6))
    X_te = rng.normal(size=(7, 6))

    kernel = BiologicalRelationalQuantumKernel()
    kernel.fit(X_tr)

    K_eval = kernel.compute_gram_matrix(X_tr, X_eval=X_te)
    assert K_eval.shape == (7, 15)
    assert np.all(K_eval >= 0.0)
    assert np.all(K_eval <= 1.0 + 1e-12)

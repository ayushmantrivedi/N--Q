"""
Tests for Biological Relational Quantum Feature Map.
"""

import numpy as np
import pytest
from nadiq.models.quantum.brqk.feature_map import (
    FeatureStandardizer,
    compute_angles,
    compute_interaction_terms,
    biological_feature_map,
    adjoint_biological_feature_map,
)
import pennylane as qml


def test_standardizer_train_isolation():
    rng = np.random.default_rng(42)
    X_tr = rng.normal(10.0, 2.0, size=(50, 6))
    X_te = rng.normal(20.0, 5.0, size=(20, 6))

    scaler = FeatureStandardizer()
    scaler.fit(X_tr)

    assert np.allclose(scaler.mean_, np.mean(X_tr, axis=0))
    assert np.allclose(scaler.scale_, np.std(X_tr, axis=0))

    X_te_scaled = scaler.transform(X_te)
    # Check that transform did not alter scaler's mean_ or scale_
    assert np.allclose(scaler.mean_, np.mean(X_tr, axis=0))
    # Transform on train should have mean ~0 and std ~1
    X_tr_scaled = scaler.transform(X_tr)
    assert np.allclose(np.mean(X_tr_scaled, axis=0), 0.0, atol=1e-5)
    assert np.allclose(np.std(X_tr_scaled, axis=0), 1.0, atol=1e-3)


def test_compute_angles_bounds():
    X = np.array([[-1000.0, -10.0, 0.0, 10.0, 1000.0, 0.5]])
    thetas = compute_angles(X, angle_scale=np.pi)
    assert thetas.shape == (1, 6)
    assert np.all(thetas >= -np.pi)
    assert np.all(thetas <= np.pi)
    assert np.isclose(thetas[0, 2], 0.0)


def test_compute_interaction_terms():
    theta = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    pairs = [(0, 1), (2, 3), (0, 5)]
    r = compute_interaction_terms(theta, pairs)
    assert len(r) == 3
    assert np.isclose(r[0], 2.0)
    assert np.isclose(r[1], 12.0)
    assert np.isclose(r[2], 6.0)

    # 2D batch test
    thetas_2d = np.stack([theta, theta * 2])
    r_2d = compute_interaction_terms(thetas_2d, pairs)
    assert r_2d.shape == (2, 3)
    assert np.isclose(r_2d[1, 0], 8.0)


def test_feature_map_unitary_adjoint_cancellation():
    """Applying U followed by U^dagger should return exact |000000> state (probability 1.0)."""
    theta = np.array([0.5, -0.8, 1.2, -0.3, 0.9, -1.1])
    pairs = [(0, 2), (1, 2), (0, 3), (1, 3), (2, 4), (3, 4), (3, 5), (0, 4)]
    dev = qml.device("default.qubit", wires=6)

    @qml.qnode(dev)
    def identity_test():
        biological_feature_map(theta, pairs, entanglement_type="biological")
        adjoint_biological_feature_map(theta, pairs, entanglement_type="biological")
        return qml.probs(wires=range(6))

    probs = identity_test()
    assert np.isclose(probs[0], 1.0, atol=1e-10)
    assert np.allclose(probs[1:], 0.0, atol=1e-10)

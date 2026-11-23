"""
Smoke test for Phase 14A Benchmark Runner.
"""

import numpy as np
import pytest
from nadiq.experiments.phase14a.benchmark import Phase14ABenchmarkRunner


def test_benchmark_runner_smoke():
    rng = np.random.default_rng(42)
    N_tr = 16
    N_te = 6
    X_tr = rng.normal(size=(N_tr, 6))
    y_tr = rng.binomial(1, 0.5, size=N_tr)
    # Ensure both classes present
    y_tr[0] = 0
    y_tr[1] = 1

    X_te = rng.normal(size=(N_te, 6))
    y_te = rng.binomial(1, 0.5, size=N_te)

    ctclm_inputs = {
        "seq": rng.normal(size=(N_tr + N_te, 2, 1)),
        "times": np.tile(np.array([0.0, 1.0]), (N_tr + N_te, 1)),
        "mask": np.ones((N_tr + N_te, 2)),
    }
    tr_idx = np.arange(N_tr)
    te_idx = np.arange(N_tr, N_tr + N_te)

    runner = Phase14ABenchmarkRunner()
    results = runner.run_fold(
        X_tr=X_tr,
        y_tr=y_tr,
        X_te=X_te,
        y_te=y_te,
        ctclm_inputs=ctclm_inputs,
        train_idx=tr_idx,
        test_idx=te_idx,
        seed=42,
    )

    expected_models = [
        "M0_Logistic_Regression",
        "M1_RBF_SVM",
        "M2_Poly_SVM",
        "M3_Classical_Relational_Kernel",
        "M4_Classical_Relational_Classifier",
        "M5_Quantum_Relational_Kernel",
        "M6_CTCLM_Baseline",
        "M7_CTCLM_Classical_Relational_Hybrid",
        "M8_CTCLM_Quantum_Relational_Hybrid",
        "M9_Quantum_Shuffled_Topology",
        "M5_Ablation_No_Entanglement",
    ]

    for m in expected_models:
        assert m in results, f"Missing model {m} in results"
        probs = results[m]["prob"]
        assert len(probs) == N_te
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

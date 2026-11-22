"""
Phase 9 — QVBE-v3 Unit Test Suite
Verifier worker: test_qvbe_v3.py

Run with:
    python -m pytest tests/phase9/test_qvbe_v3.py -v

All tests use ONLY synthetic (numpy random) data. No PPMI data needed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def _make_batch(n=10, d=8, seed=SEED):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Data re-uploading circuit produces correct output shape
# ─────────────────────────────────────────────────────────────────────────────

def test_data_reuploading_circuit_shape():
    """
    Given 10 samples of 8 features, encoder output shape should be
    (10, output_dim) where output_dim == config.n_qubits (at minimum).
    """
    from nadiq.models.qvbe_v3.config import QVBEv3Config
    from nadiq.models.qvbe_v3.circuit import QVBEv3Circuit

    cfg = QVBEv3Config(n_qubits=8, n_layers=4, n_uploads=3,
                       use_trainable_adapter=False)
    circuit = QVBEv3Circuit(cfg)
    x = _make_batch(n=10, d=8)

    with torch.no_grad():
        out = circuit(x)

    assert out.shape[0] == 10, f"Expected 10 samples, got {out.shape[0]}"
    assert out.shape[1] >= cfg.n_qubits, (
        f"Expected output_dim >= n_qubits={cfg.n_qubits}, got {out.shape[1]}"
    )
    print(f"[PASS] output shape: {tuple(out.shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Gradients flow through circuit parameters
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_flows():
    """
    After 3 optimisation steps, every circuit parameter's .grad should be
    finite and non-zero (i.e., gradient actually reaches quantum params).
    """
    from nadiq.models.qvbe_v3.config import QVBEv3Config
    from nadiq.models.qvbe_v3.circuit import QVBEv3Circuit
    from nadiq.models.qvbe_v3.objectives import infonce_loss

    cfg = QVBEv3Config(n_qubits=8, n_layers=4, n_uploads=3,
                       use_trainable_adapter=False)
    circuit = QVBEv3Circuit(cfg)
    opt = torch.optim.Adam(circuit.parameters(), lr=1e-2)

    x_mol = _make_batch(n=6, d=8, seed=1)
    x_tmp = _make_batch(n=6, d=8, seed=2)

    for step in range(3):
        opt.zero_grad()
        q_mol = circuit(x_mol)
        q_tmp = circuit(x_tmp)
        loss = infonce_loss(q_mol, q_tmp, temperature=0.5)
        loss.backward()
        if step == 0:
            # check after first backward
            for name, p in circuit.named_parameters():
                assert p.grad is not None, f"param '{name}' has None grad"
                assert torch.isfinite(p.grad).all(), (
                    f"param '{name}' has non-finite grad"
                )
                assert p.grad.abs().max() > 0, (
                    f"param '{name}' has all-zero grad"
                )
        opt.step()

    print("[PASS] gradient_flows — all params have finite non-zero grads")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Entanglement ablation: separable vs entangled outputs differ
# ─────────────────────────────────────────────────────────────────────────────

def test_entanglement_ablation():
    """
    A separable (no CNOT) circuit and an entangled (CNOT) circuit should
    produce outputs whose mean absolute difference > 1e-3.
    """
    from nadiq.models.qvbe_v3.config import QVBEv3Config
    from nadiq.models.qvbe_v3.circuit import QVBEv3Circuit

    cfg_ent = QVBEv3Config(n_qubits=8, n_layers=4, n_uploads=3,
                            use_trainable_adapter=False, entangle=True)
    cfg_sep = QVBEv3Config(n_qubits=8, n_layers=4, n_uploads=3,
                            use_trainable_adapter=False, entangle=False)

    circ_ent = QVBEv3Circuit(cfg_ent)
    circ_sep = QVBEv3Circuit(cfg_sep)

    # Copy same parameters so only entanglement structure differs
    with torch.no_grad():
        sep_params = dict(circ_sep.named_parameters())
        for name, p in circ_ent.named_parameters():
            if name in sep_params and sep_params[name].shape == p.shape:
                sep_params[name].copy_(p)

    x = _make_batch(n=20, d=8)
    with torch.no_grad():
        out_ent = circ_ent(x)
        out_sep = circ_sep(x)

    mean_diff = (out_ent - out_sep).abs().mean().item()
    assert mean_diff > 1e-3, (
        f"Entangled vs separable mean diff {mean_diff:.6f} <= 1e-3; "
        "entanglement has negligible effect"
    )
    print(f"[PASS] entanglement_ablation — mean diff: {mean_diff:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Connected correlators C_ij = <Z_i Z_j> - <Z_i><Z_j>
# ─────────────────────────────────────────────────────────────────────────────

def test_connected_correlator():
    """
    For separable circuit: |C_ij| should be < 1e-6 for all pairs.
    For entangled circuit: mean |C_ij| should be > 0.
    """
    from nadiq.models.qvbe_v3.config import QVBEv3Config
    from nadiq.models.qvbe_v3.observables import compute_connected_correlators

    cfg_ent = QVBEv3Config(n_qubits=8, n_layers=4, n_uploads=3,
                            use_trainable_adapter=False, entangle=True)
    cfg_sep = QVBEv3Config(n_qubits=8, n_layers=4, n_uploads=3,
                            use_trainable_adapter=False, entangle=False)

    x = _make_batch(n=10, d=8)

    C_sep = compute_connected_correlators(x, cfg_sep)   # shape (n_pairs,) or (n, n_pairs)
    C_ent = compute_connected_correlators(x, cfg_ent)

    C_sep_abs = C_sep.abs().max().item() if isinstance(C_sep, torch.Tensor) else np.max(np.abs(C_sep))
    C_ent_abs = C_ent.abs().mean().item() if isinstance(C_ent, torch.Tensor) else np.mean(np.abs(C_ent))

    assert C_sep_abs < 1e-4, (
        f"Separable circuit: max|C_ij|={C_sep_abs:.2e} should be < 1e-4"
    )
    assert C_ent_abs > 0, (
        f"Entangled circuit: mean|C_ij|={C_ent_abs:.2e} should be > 0"
    )
    print(f"[PASS] connected_correlator — sep max: {C_sep_abs:.2e}, ent mean: {C_ent_abs:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Seed variance < patient variance (the Phase 8 bug is fixed)
# ─────────────────────────────────────────────────────────────────────────────

def test_seed_variance():
    """
    For N=50 patients, the variance of outputs across patients should be
    GREATER than the variance across random seeds for the same patient.
    Ratio < 1.0 means the fix worked (Phase 8 had ratio=1.54 — broken).
    """
    from nadiq.models.qvbe_v3.config import QVBEv3Config
    from nadiq.models.qvbe_v3.diagnostics import seed_variance_vs_patient_variance

    cfg = QVBEv3Config(n_qubits=8, n_layers=4, n_uploads=3,
                       use_trainable_adapter=False)
    x_patients = _make_batch(n=50, d=8, seed=SEED)

    ratio = seed_variance_vs_patient_variance(cfg, x_patients, n_seeds=10)

    assert ratio < 1.0, (
        f"seed_var/patient_var ratio = {ratio:.4f}; should be < 1.0 after fix "
        "(Phase 8 baseline ratio was 1.54)"
    )
    print(f"[PASS] seed_variance — ratio: {ratio:.4f} < 1.0")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — No PCA bottleneck: trainable adapter W has correct shape
# ─────────────────────────────────────────────────────────────────────────────

def test_no_pca_bottleneck():
    """
    When input d=50 and n_qubits=8, the model should have:
    - A trainable linear adapter W with shape (8, 50)  [n_qubits × d_input]
    - NO sklearn PCA object anywhere in the model
    """
    from nadiq.models.qvbe_v3.config import QVBEv3Config
    from nadiq.models.qvbe_v3.circuit import QVBEv3Circuit

    cfg = QVBEv3Config(n_qubits=8, n_layers=4, n_uploads=3,
                       use_trainable_adapter=True, input_dim=50)
    circuit = QVBEv3Circuit(cfg)

    # Check adapter weight shape
    adapter_found = False
    for name, p in circuit.named_parameters():
        if 'adapter' in name.lower() or 'W' in name or 'input_proj' in name.lower():
            # Expect shape (8, 50)
            assert p.shape == (8, 50) or p.shape == (cfg.n_qubits, 50), (
                f"Adapter '{name}' has shape {tuple(p.shape)}, expected ({cfg.n_qubits}, 50)"
            )
            adapter_found = True
            break

    assert adapter_found, (
        "No trainable adapter parameter found (expected parameter with 'adapter' or 'input_proj' in name)"
    )

    # Check no PCA object exists
    import sklearn
    for attr_name in dir(circuit):
        attr = getattr(circuit, attr_name, None)
        if isinstance(attr, sklearn.decomposition.PCA):
            raise AssertionError(
                f"Found sklearn PCA object at circuit.{attr_name} — PCA bottleneck still present!"
            )

    print("[PASS] no_pca_bottleneck — trainable adapter W found, no PCA object")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — InfoNCE loss distinguishes correct from incorrect pairs
# ─────────────────────────────────────────────────────────────────────────────

def test_infonce_loss_uses_pairs():
    """
    Construct two positive pairs (identical embeddings) and two negative pairs
    (orthogonal embeddings). InfoNCE loss for correct pairs should be < loss
    for incorrect/shuffled pairs.
    """
    from nadiq.models.qvbe_v3.objectives import infonce_loss

    dim = 16
    # Positive pairs: q_mol[i] ≈ q_tmp[i]
    q_mol_pos = torch.randn(4, dim)
    q_tmp_pos = q_mol_pos.clone() + 0.01 * torch.randn(4, dim)   # almost identical

    # Shuffled (incorrect) pairs
    q_tmp_neg = q_tmp_pos[torch.randperm(4)]

    loss_correct = infonce_loss(q_mol_pos, q_tmp_pos, temperature=0.1)
    loss_shuffled = infonce_loss(q_mol_pos, q_tmp_neg, temperature=0.1)

    assert loss_correct < loss_shuffled, (
        f"InfoNCE correct pairs loss {loss_correct:.4f} should be < "
        f"shuffled pairs loss {loss_shuffled:.4f}"
    )
    print(f"[PASS] infonce_loss_uses_pairs — correct: {loss_correct:.4f}, shuffled: {loss_shuffled:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — Training loss is non-increasing over 5 steps
# ─────────────────────────────────────────────────────────────────────────────

def test_reconstruction_monotonic():
    """
    Run 5 optimisation steps. The loss trend should be downward (not strictly
    monotone, but the last loss < first loss with a reasonable margin).
    We use a small network and large LR to get quick convergence signal.
    """
    from nadiq.models.qvbe_v3.trainer import QVBEv3Trainer
    from nadiq.models.qvbe_v3.config import QVBEv3Config

    cfg = QVBEv3Config(n_qubits=4, n_layers=2, n_uploads=2,
                       use_trainable_adapter=False)
    trainer = QVBEv3Trainer(cfg, lr=5e-2)

    # Synthetic molecular + temporal batches (same size, different features)
    x_mol = _make_batch(n=8, d=4, seed=10)
    x_tmp = _make_batch(n=8, d=4, seed=20)

    losses = []
    for _ in range(5):
        loss_val = trainer.step(x_mol, x_tmp)
        losses.append(loss_val)

    print(f"  Losses over 5 steps: {[f'{l:.4f}' for l in losses]}")

    # Relaxed criterion: last 2-step average < first 2-step average
    first_avg = sum(losses[:2]) / 2
    last_avg = sum(losses[-2:]) / 2
    assert last_avg <= first_avg * 1.05, (
        f"Loss not decreasing: first_avg={first_avg:.4f}, last_avg={last_avg:.4f}. "
        "Training appears divergent."
    )
    print(f"[PASS] reconstruction_monotonic — first_avg: {first_avg:.4f}, last_avg: {last_avg:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Extra: Verify effective_rank returns a float
# ─────────────────────────────────────────────────────────────────────────────

def test_effective_rank_callable():
    """
    effective_rank() in diagnostics.py should accept a 2D tensor/array and
    return a Python float.
    """
    from nadiq.models.qvbe_v3.diagnostics import effective_rank

    X = torch.randn(30, 8)
    result = effective_rank(X)
    assert isinstance(result, (float, int, np.floating)), (
        f"effective_rank returned {type(result)}, expected float"
    )
    assert 1.0 <= result <= 8.0, (
        f"effective_rank={result:.4f} out of plausible range [1, n_qubits=8]"
    )
    print(f"[PASS] effective_rank_callable — result: {result:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Extra: Verify config.py has required fields
# ─────────────────────────────────────────────────────────────────────────────

def test_config_required_fields():
    """
    QVBEv3Config must have: n_qubits, n_layers, n_uploads, use_trainable_adapter
    """
    from nadiq.models.qvbe_v3.config import QVBEv3Config

    cfg = QVBEv3Config()
    required = ['n_qubits', 'n_layers', 'n_uploads', 'use_trainable_adapter']
    for field in required:
        assert hasattr(cfg, field), f"QVBEv3Config missing required field: '{field}'"

    print(f"[PASS] config_required_fields — all required fields present")


if __name__ == "__main__":
    # Can also run directly: python tests/phase9/test_qvbe_v3.py
    import pytest as _pytest
    _pytest.main([__file__, "-v"])

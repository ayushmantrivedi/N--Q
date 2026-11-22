"""
NĀḌĪ-Q QVBE-v3 Physical Observables and Connected Correlation Metrics.
"""

from __future__ import annotations

import torch
from nadiq.models.qvbe_v3.config import QVBEv3Config
from nadiq.models.qvbe_v3.circuit import QVBEv3Circuit


def compute_connected_correlators(x: torch.Tensor, config: QVBEv3Config) -> torch.Tensor:
    """
    Compute connected 2-qubit correlators:
        C_{i, i+1} = <Z_i Z_{i+1}> - <Z_i><Z_{i+1}>
    For separable circuits, C_{i, j} is strictly zero within floating point precision.
    For entangled circuits, C_{i, j} captures true genuine quantum correlations.
    
    x: (B, d) input tensor
    returns: (B, n-1) tensor of connected correlators
    """
    n = config.n_qubits
    # Force full basis readout so we have both single-qubit <Z> and 2-qubit <ZZ>
    cfg_eval = QVBEv3Config(
        n_qubits=config.n_qubits,
        n_layers=config.n_layers,
        n_uploads=config.n_uploads,
        use_trainable_adapter=config.use_trainable_adapter,
        input_dim=config.input_dim,
        entangle=config.entangle,
        measurement_basis="full",
        shots=config.shots,
        backend=config.backend,
        seed=config.seed
    )
    circuit = QVBEv3Circuit(cfg_eval)

    with torch.no_grad():
        out = circuit(x)  # shape (B, 3n - 1)

    # Indices:
    # 0 .. n-1: <Z_i>
    # n .. 2n-1: <X_i>
    # 2n .. 3n-2: <Z_i Z_{i+1}> for i=0 .. n-2
    z_singles = out[:, :n]            # (B, n)
    zz_pairs = out[:, 2 * n: 3 * n - 1] # (B, n-1)

    # Compute disconnected product <Z_i><Z_{i+1}>
    z_prod = z_singles[:, :n - 1] * z_singles[:, 1:n]  # (B, n-1)

    # Connected correlator C_{i, i+1}
    c_connected = zz_pairs - z_prod
    return c_connected

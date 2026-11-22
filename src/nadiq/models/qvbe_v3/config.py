"""
NĀḌĪ-Q QVBE-v3 Configuration Specification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class QVBEv3Config:
    """
    Configuration specification for QVBE v3 (Remediated Quantum Biomarker Encoder).
    """
    n_qubits: int = 8
    n_layers: int = 4
    n_uploads: int = 3
    use_trainable_adapter: bool = False
    input_dim: int = 8
    entangle: bool = True
    measurement_basis: str = "full"  # "z_only", "dual_basis", "full", "full_plus_connected"
    shots: Optional[int] = None      # None = exact analytic statevector simulation
    backend: str = "default.qubit"
    seed: int = 42
    learning_rate: float = 0.01
    temperature: float = 0.5
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValueError(f"n_qubits must be >= 2, got {self.n_qubits}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.n_uploads < 1:
            raise ValueError(f"n_uploads must be >= 1, got {self.n_uploads}")

    @property
    def output_dim(self) -> int:
        n = self.n_qubits
        if self.measurement_basis == "z_only":
            return n
        elif self.measurement_basis == "dual_basis":
            return 2 * n
        elif self.measurement_basis == "full":
            return 3 * n - 1  # n (Z) + n (X) + (n-1) (ZZ)
        elif self.measurement_basis == "full_plus_connected":
            return 3 * n - 1 + (n - 1)  # Includes connected correlators
        return n

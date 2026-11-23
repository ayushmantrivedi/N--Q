"""
NĀḌĪ-Q Phase 14A: Biological Relational Quantum Kernel (BRQK) Configuration.
Defines hyperparameters for quantum feature encoding, pairwise biological interaction topology,
simulation mode, and finite-shot estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BRQKConfig:
    """
    Configuration for Biological Relational Quantum Kernel (BRQK).

    Attributes:
        n_qubits: Number of qubits (default: 6, matching molecular, structural, and motor channels).
        feature_dim: Dimension of input feature vector (default: 6).
        interaction_pairs: List of (qubit_i, qubit_j) tuples representing biological relationships.
        entanglement_type: 'biological' (specified pairs), 'all_to_all', 'linear', or 'none'.
        angle_scale: Scaling factor for tanh bounding (default: pi).
        simulation_mode: 'statevector' (exact) or 'finite_shot' (measurement sampling).
        n_shots: Number of measurement shots for finite-shot estimation (e.g. 1024 or 4096).
        epsilon: Small numerical stability term for standardization.
        seed: Random seed for deterministic reproducibility.
    """
    n_qubits: int = 6
    feature_dim: int = 6
    # Default biological interaction pairs:
    # 0: asyn, 1: ptau, 2: caudate, 3: putamen, 4: motor_total, 5: motor_subscore
    # (0, 2): asyn <-> caudate (molecular-striatal)
    # (1, 2): ptau <-> caudate (molecular-striatal)
    # (0, 3): asyn <-> putamen (molecular-striatal)
    # (1, 3): ptau <-> putamen (molecular-striatal)
    # (2, 4): caudate <-> motor_total (structural-motor)
    # (3, 4): putamen <-> motor_total (structural-motor)
    # (3, 5): putamen <-> motor_subscore (structural-motor)
    # (0, 4): asyn <-> motor_total (molecular-motor shortcut)
    interaction_pairs: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (0, 2), (1, 2), (0, 3), (1, 3),
            (2, 4), (3, 4), (3, 5), (0, 4),
        ]
    )
    entanglement_type: str = "biological"  # "biological", "shuffled", "all_to_all", "none"
    angle_scale: float = 3.141592653589793  # pi
    simulation_mode: str = "statevector"  # "statevector" or "finite_shot"
    n_shots: int = 2048
    epsilon: float = 1e-6
    seed: int = 42

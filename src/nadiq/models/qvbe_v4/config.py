"""
NĀḌĪ-Q QVBE-v4 Configuration Specification.
Defines hyperparameters for the Graph-Conditioned Temporal Quantum Relational Engine.
Supports pairwise relational learning, learned temporal decays, uncertainty modulation,
and multi-objective optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class QVBEv4Config:
    """
    Configuration specification for QVBE-v4.
    """
    # Register Architecture
    n_nodes: int = 3
    qubits_per_node: int = 2  # Qubit 0: State, Qubit 1: Temporal Change
    node_names: List[str] = field(
        default_factory=lambda: ["molecular", "structural", "functional"]
    )
    node_input_dims: Dict[str, int] = field(
        default_factory=lambda: {
            "molecular": 15,
            "structural": 36,
            "functional": 35,
        }
    )

    # Circuit Layout & Hamiltonian Evolution
    n_layers: int = 1
    trotter_steps: int = 1
    use_data_reuploading: bool = True
    enable_pairwise_coupling: bool = True
    enable_change_channel_coupling: bool = True
    enable_direction_coupling: bool = True
    enable_higher_order_coupling: bool = False  # Disabled by default in Phase 11C
    enable_time_conditioning: bool = True
    decay_lambda: float = 0.10

    # Advanced Relational & Temporal Parameterization (Phase 11C)
    enable_uncertainty_weighting: bool = False
    enable_learned_temporal_decay: bool = False
    temporal_decay_mode: str = "exp"  # "exp" or "rational" (1 / (1 + lambda * dt))
    edge_coupling_mode: str = "direct"  # "direct", "learned_scale", "confidence_weighted"

    # Interaction Angle Bounds & Transverse Mixing Field
    max_coupling_angle: float = 1.5708  # pi / 2
    init_coupling_std: float = 0.50
    transverse_mixing_angle: float = 0.7854  # pi / 4
    near_identity_init_std: float = 0.05

    # Quantum Device
    device_name: str = "default.qubit"
    shots: Optional[int] = None  # None = exact analytic differentiation

    # Classical Projection & Embeddings
    motif_embed_dim: int = 16
    hidden_dim: int = 16
    dropout: float = 0.10

    # Multi-Objective Loss Weights
    lambda_edge: float = 0.10
    lambda_temporal: float = 0.10
    lambda_masked: float = 0.0
    lambda_task: float = 1.00

    # Optimization
    learning_rate: float = 0.02
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    seed: int = 42

    @property
    def total_qubits(self) -> int:
        return self.n_nodes * self.qubits_per_node

    def get_node_qubits(self, node_idx: int) -> Tuple[int, int]:
        """Returns (state_qubit, change_qubit) for a given node index."""
        state_q = node_idx * self.qubits_per_node
        change_q = state_q + 1
        return state_q, change_q

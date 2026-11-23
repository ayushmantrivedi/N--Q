"""
NĀḌĪ-Q Integrated Hybrid Intelligence System Configuration.
Defines parameters for unified MeTTa + QVBE-v4 + CTCLM + QCDF + Q-XAI architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nadiq.models.qvbe_v4.config import QVBEv4Config
from nadiq.models.quantum.ctclm import CTCLMConfig


@dataclass
class NadiQIntegratedConfig:
    """
    Unified configuration for the full NĀḌĪ-Q Hybrid Intelligence System.
    """
    # 1. Component Configurations
    qvbe_config: QVBEv4Config = field(default_factory=lambda: QVBEv4Config(
        n_nodes=3,
        qubits_per_node=2,
        node_names=["molecular", "structural", "functional"],
        node_input_dims={"molecular": 1, "structural": 1, "functional": 1},
        n_layers=1,
        motif_embed_dim=16,
        hidden_dim=16,
        learning_rate=0.01,
        enable_higher_order_coupling=False,
    ))

    ctclm_config: CTCLMConfig = field(default_factory=lambda: CTCLMConfig(
        input_dim=1,           # Unidimensional functional score per timestep (e.g. UPDRS3)
        latent_dim=16,         # Hidden dimension of temporal GRU
        time_embedding_dim=4,  # Temporal encoding dimension
        num_layers=1,
        dropout=0.0,
    ))

    # 2. Classical Relational Model Settings
    enable_classical_relational: bool = True
    classical_relational_dim: int = 12

    # 3. Adaptive QCDF Fusion Settings
    fusion_type: str = "adaptive"  # "adaptive" or "residual_adaptive"
    fusion_projection_dim: int = 16
    fusion_hidden_dim: int = 32
    fusion_dropout: float = 0.10
    num_classes: int = 1  # Binary progression classification

    # 4. Dynamic Quantum Contribution Gating
    initial_quantum_gate_bias: float = -0.50  # Start moderately conservative (sigmoid(-0.5) ~ 0.38)
    initial_temporal_gate_bias: float = 0.50   # Start active for temporal progression
    initial_node_gate_bias: float = 0.50       # Start active for static features
    force_quantum_suppression: bool = False   # Hard suppression toggle for M6 ablation

    # 5. Branch Activation Toggles
    enable_quantum_branch: bool = True
    enable_ctclm_branch: bool = True
    enable_node_branch: bool = True

    # 6. Optimization
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    seed: int = 42

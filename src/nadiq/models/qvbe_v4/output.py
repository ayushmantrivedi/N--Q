"""
NĀḌĪ-Q QVBE-v4 Output Data Structures.
Defines structured quantum relational representations, correlation matrices,
observables, diagnostics, and downstream integration contracts for CTCLM, QCDF, and Q-XAI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import torch


@dataclass
class QVBEv4Output:
    """
    Structured output from QVBE-v4 Graph-Conditioned Temporal Relational Engine.
    Exposes full quantum representations, edge observables, diagnostics, and
    downstream integration interfaces.
    """
    # 1. Primary Required Relational Representations (No Defaults)
    motif_embedding: torch.Tensor                          # (B, motif_embed_dim)
    local_expectations_z: torch.Tensor                      # (B, n_qubits)
    local_expectations_x: torch.Tensor                      # (B, n_qubits)
    local_expectations_y: torch.Tensor                      # (B, n_qubits)
    pairwise_correlations_zz: torch.Tensor                  # (B, n_edges)
    pairwise_correlations_xx: torch.Tensor                  # (B, n_edges)
    connected_correlators: torch.Tensor                     # (B, n_edges) -> <Z_i Z_j> - <Z_i><Z_j>

    # 2. Downstream Integration & Convenience Aliases (With Defaults)
    quantum_relational_embeddings: Optional[torch.Tensor] = None # Alias for motif_embedding
    edge_observables: Optional[Dict[str, torch.Tensor]] = None   # {"zz": (B, n_edges), "xx": ..., "connected": ...}
    three_body_expectation: Optional[torch.Tensor] = None        # (B, 1) -> <Z_i Z_j Z_k>
    three_body_cumulant: Optional[torch.Tensor] = None           # (B, 1) -> Connected 3-body cumulant
    higher_order_observables: Optional[Dict[str, torch.Tensor]] = None
    node_embeddings: Optional[Dict[str, torch.Tensor]] = None     # {node_name: (B, 2) projected (state, change)}
    availability_masks: Optional[torch.Tensor] = None           # (B, n_nodes) binary mask
    temporal_relational_features: Optional[torch.Tensor] = None  # (B, n_edges) effective temporal factors
    subsystem_entropies: Optional[torch.Tensor] = None          # (B, n_nodes) -> von Neumann entropy S(rho_i)
    state_vector: Optional[torch.Tensor] = None                 # (B, 2^n_qubits) in simulator mode
    quantum_diagnostics: Dict[str, Any] = field(default_factory=dict)
    downstream_logits: Optional[torch.Tensor] = None            # (B, 1) or (B,)
    reconstructed_edge_weights: Optional[torch.Tensor] = None   # (B, n_edges)
    predicted_next_delta: Optional[torch.Tensor] = None         # (B, n_nodes)
    reconstructed_node_features: Optional[Dict[str, torch.Tensor]] = None
    classical_graph_metadata: Dict[str, Any] = field(default_factory=dict)

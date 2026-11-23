"""
NĀḌĪ-Q QVBE-v4 Graph-Conditioned Temporal Relational Encoder.
Combines trainable input adapters, graph-conditioned quantum Hamiltonian circuit,
relational motif projection, and multi-objective prediction heads.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nadiq.models.qvbe_v4.config import QVBEv4Config
from nadiq.models.qvbe_v4.circuit import QVBEv4Circuit
from nadiq.models.qvbe_v4.graph_compiler import NumericalGraphCompiler
from nadiq.models.qvbe_v4.output import QVBEv4Output


class NodeInputAdapter(nn.Module):
    """
    Trainable input adapter for a specific biological node modality.
    Projects variable raw feature dimensions into 1D state and change channels.
    """

    def __init__(self, raw_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.raw_dim = raw_dim
        self.state_proj = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.change_proj = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Explicit learned missing representation
        self.missing_state = nn.Parameter(torch.zeros(1))
        self.missing_change = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x_raw: torch.Tensor,
        dx_raw: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns (B, 2) tensor containing (state, change).
        If mask is 0, outputs are strictly replaced by learned missing parameters.
        """
        batch_size = x_raw.shape[0]

        # Ensure 2D tensor
        if x_raw.ndim == 1:
            x_raw = x_raw.unsqueeze(-1)
        if dx_raw.ndim == 1:
            dx_raw = dx_raw.unsqueeze(-1)

        val_s = self.state_proj(x_raw).squeeze(-1)
        val_c = self.change_proj(dx_raw).squeeze(-1)

        # Apply missingness mask
        if mask is not None:
            if mask.ndim == 2:
                m = mask.squeeze(-1)
            else:
                m = mask
            # Zero out arbitrary placeholder values and add missing embedding
            val_s = val_s * m + self.missing_state.expand(batch_size) * (1.0 - m)
            val_c = val_c * m + self.missing_change.expand(batch_size) * (1.0 - m)

        return torch.stack([val_s, val_c], dim=-1)


class QVBEv4Encoder(nn.Module):
    """
    Full QVBE-v4 Graph-Conditioned Temporal Relational Encoder.
    """

    def __init__(self, config: QVBEv4Config):
        super().__init__()
        self.config = config
        self.compiler = NumericalGraphCompiler(
            node_names=config.node_names,
            decay_lambda=config.decay_lambda,
            max_coupling_angle=config.max_coupling_angle,
        )

        # 1. Trainable Node Input Adapters
        self.adapters = nn.ModuleDict()
        for name in config.node_names:
            in_dim = config.node_input_dims.get(name, 1)
            self.adapters[name] = NodeInputAdapter(raw_dim=in_dim, hidden_dim=config.hidden_dim)

        # 2. Quantum Relational Circuit
        self.circuit = QVBEv4Circuit(config)

        # 3. Output Dimensions
        # Local expectations: 6 Z + 6 X + 6 Y = 18
        # Two-body correlations: 3 ZZ + 3 XX = 6
        # Connected correlators: 3
        # Higher-order cumulant: 1
        # Total quantum features = 18 + 6 + 3 + 1 = 28
        self.n_quantum_features = (
            self.config.total_qubits * 3  # Z, X, Y
            + self.circuit.n_pairs * 2     # ZZ, XX
            + self.circuit.n_pairs         # connected ZZ
            + 1                            # cumulant 3
        )

        # 4. Relational Motif Embedding Projection
        self.motif_projector = nn.Sequential(
            nn.Linear(self.n_quantum_features, config.motif_embed_dim),
            nn.LayerNorm(config.motif_embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # 5. Multi-Objective Prediction Heads
        # Downstream classification head
        self.downstream_head = nn.Linear(config.motif_embed_dim, 1)

        # Edge reconstruction head (predicts n_pairs edge weights)
        self.edge_recon_head = nn.Sequential(
            nn.Linear(config.motif_embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, self.circuit.n_pairs),
        )

        # Temporal prediction head (predicts next-visit state changes for all nodes)
        self.temporal_pred_head = nn.Sequential(
            nn.Linear(config.motif_embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.n_nodes),
        )

        # Node feature reconstruction heads (for masked node self-supervision)
        self.node_recon_heads = nn.ModuleDict()
        for name in config.node_names:
            out_dim = config.node_input_dims.get(name, 1)
            self.node_recon_heads[name] = nn.Sequential(
                nn.Linear(config.motif_embed_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, out_dim),
            )

    def forward(
        self,
        node_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        graph_edges: List[Dict[str, Any]],
        node_masks: Optional[torch.Tensor] = None,
        elapsed_times: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> QVBEv4Output:
        """
        Executes end-to-end forward pass:
        - node_inputs: {node_name: (x_raw, dx_raw)}
        - graph_edges: list of edge dicts from MeTTa
        - node_masks: (B, n_nodes) binary availability mask
        """
        first_node = self.config.node_names[0]
        batch_size = node_inputs[first_node][0].shape[0]

        # 1. Project node representations through trainable adapters
        projected_nodes = []
        for i, name in enumerate(self.config.node_names):
            x_raw, dx_raw = node_inputs[name]
            mask_i = node_masks[:, i] if node_masks is not None else None
            adapter_out = self.adapters[name](x_raw, dx_raw, mask=mask_i)
            projected_nodes.append(adapter_out)

        # Stack into (B, n_nodes, 2)
        node_states = torch.stack(projected_nodes, dim=1)

        # 2. Compile numerical graph tensors
        graph_tensors = self.compiler.compile_graph_motif(
            edges=graph_edges,
            node_masks=node_masks,
            elapsed_times=elapsed_times,
            batch_size=batch_size,
        )

        # Move graph tensors to node_states device
        device = node_states.device
        for k, v in graph_tensors.items():
            if isinstance(v, torch.Tensor):
                graph_tensors[k] = v.to(device)

        # 3. Execute Quantum Relational Evolution
        q_out = self.circuit(node_states, graph_tensors)

        # 4. Form Feature Vector
        q_features = torch.cat([
            q_out["expval_z"],
            q_out["expval_x"],
            q_out["expval_y"],
            q_out["corrs_zz"],
            q_out["corrs_xx"],
            q_out["connected_corrs"],
            q_out["cumulant_3"],
        ], dim=-1)

        # 5. Project to Motif Embedding
        motif_emb = self.motif_projector(q_features)

        # 6. Multi-Objective Heads
        downstream_logits = self.downstream_head(motif_emb).squeeze(-1)
        recon_edges = self.edge_recon_head(motif_emb)
        pred_delta = self.temporal_pred_head(motif_emb)

        recon_nodes = {}
        for name in self.config.node_names:
            recon_nodes[name] = self.node_recon_heads[name](motif_emb)

        # 7. Optional Diagnostics
        subsystem_ent = None
        diagnostics: Dict[str, Any] = {}
        if return_diagnostics:
            try:
                subsystem_ent = self.circuit.compute_subsystem_entropy(node_states, graph_tensors)
                diagnostics["subsystem_entropies"] = subsystem_ent
            except Exception:
                pass

        # Build clean structured edge observables
        edge_obs = {
            "zz": q_out["corrs_zz"],
            "xx": q_out["corrs_xx"],
            "connected": q_out["connected_corrs"],
        }

        # Build higher-order observables (only when enabled)
        higher_order_obs = None
        if self.config.enable_higher_order_coupling:
            higher_order_obs = {
                "three_body_expectation": q_out["triad_expval"],
                "three_body_cumulant": q_out["cumulant_3"],
            }

        # Node projected embeddings dictionary
        node_embs = {
            name: projected_nodes[i] for i, name in enumerate(self.config.node_names)
        }

        return QVBEv4Output(
            motif_embedding=motif_emb,
            quantum_relational_embeddings=motif_emb,
            local_expectations_z=q_out["expval_z"],
            local_expectations_x=q_out["expval_x"],
            local_expectations_y=q_out["expval_y"],
            pairwise_correlations_zz=q_out["corrs_zz"],
            pairwise_correlations_xx=q_out["corrs_xx"],
            connected_correlators=q_out["connected_corrs"],
            edge_observables=edge_obs,
            three_body_expectation=q_out["triad_expval"] if self.config.enable_higher_order_coupling else None,
            three_body_cumulant=q_out["cumulant_3"] if self.config.enable_higher_order_coupling else None,
            higher_order_observables=higher_order_obs,
            node_embeddings=node_embs,
            availability_masks=node_masks,
            temporal_relational_features=graph_tensors.get("edge_lags"),
            subsystem_entropies=subsystem_ent,
            quantum_diagnostics=diagnostics,
            downstream_logits=downstream_logits,
            reconstructed_edge_weights=recon_edges,
            predicted_next_delta=pred_delta,
            reconstructed_node_features=recon_nodes,
            classical_graph_metadata={
                "n_edges": len(graph_edges),
                "pairs": graph_tensors["pairs"],
                "active_triad": float(graph_tensors["triad_active"][0, 0].item()),
            },
        )

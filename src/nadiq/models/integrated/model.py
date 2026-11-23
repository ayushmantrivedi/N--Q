"""
NĀḌĪ-Q Unified Hybrid Intelligence System Model Architecture.
Integrates MeTTa-governed biological knowledge graph, QVBE-v4 quantum relational encoder,
CTCLM temporal latent model, and Adaptive QCDF fusion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from nadiq.models.integrated.config import NadiQIntegratedConfig
from nadiq.models.integrated.output import IntegratedOutput
from nadiq.models.integrated.fusion import AdaptiveQCDF, ResidualAdaptiveQCDF
from nadiq.models.qvbe_v4.encoder import QVBEv4Encoder
from nadiq.models.quantum.ctclm import ContinualTemporalClinicalLatentModel


class ClassicalRelationalBranch(nn.Module):
    """
    Parameter-matched classical relational branch for head-to-head comparison.
    Computes edge-conditioned relational messages across active knowledge graph edges.
    """
    def __init__(self, out_dim: int = 12, hidden_dim: int = 16):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        self.proj = nn.Linear(3 * 4, out_dim)
        self.pairs = [(0, 1), (0, 2), (1, 2)]

    def forward(
        self,
        node_states_3d: torch.Tensor,      # (B, 3, 2)
        edge_weights: torch.Tensor,        # (B, 3)
        edge_lags: torch.Tensor,           # (B, 3)
        edge_se: torch.Tensor,             # (B, 3)
        direction_signs: torch.Tensor,     # (B, 3)
    ) -> torch.Tensor:
        edge_messages = []
        for p_idx, (u, v) in enumerate(self.pairs):
            x_u = node_states_3d[:, u]
            x_v = node_states_3d[:, v]
            w = edge_weights[:, p_idx : p_idx + 1]
            dt = edge_lags[:, p_idx : p_idx + 1]
            se = edge_se[:, p_idx : p_idx + 1]
            d = direction_signs[:, p_idx : p_idx + 1]
            edge_in = torch.cat([x_u, x_v, w, dt, se, d], dim=-1)
            msg = self.edge_mlp(edge_in) * torch.exp(-0.10 * dt)
            edge_messages.append(msg)

        all_msgs = torch.cat(edge_messages, dim=-1)
        return self.proj(all_msgs)


class NadiQIntegratedModel(nn.Module):
    """
    Unified NĀḌĪ-Q Hybrid Intelligence System.
    Seamlessly orchestrates:
        Input Data -> MeTTa Graph -> QVBE-v4 -> CTCLM -> QCDF -> Prediction -> Q-XAI
    """

    def __init__(self, config: Optional[NadiQIntegratedConfig] = None):
        super().__init__()
        self.config = config or NadiQIntegratedConfig()

        # 1. Quantum Relational Branch: QVBE-v4
        self.qvbe = QVBEv4Encoder(self.config.qvbe_config)

        # 2. Temporal Intelligence Branch: CTCLM
        self.ctclm = ContinualTemporalClinicalLatentModel(self.config.ctclm_config)

        # 3. Classical Relational Branch
        self.classical_relational = ClassicalRelationalBranch(
            out_dim=self.config.classical_relational_dim,
            hidden_dim=self.config.fusion_projection_dim,
        )

        # 4. Adaptive Quantum-Classical Dynamic Fusion (QCDF)
        if getattr(self.config, "fusion_type", "adaptive") == "residual_adaptive":
            self.qcdf = ResidualAdaptiveQCDF(self.config)
        else:
            self.qcdf = AdaptiveQCDF(self.config)

    def forward(
        self,
        node_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        graph_edges: List[Dict[str, Any]],
        ctclm_seq: Optional[torch.Tensor] = None,         # (B, T, D)
        ctclm_times: Optional[torch.Tensor] = None,       # (B, T)
        ctclm_mask: Optional[torch.Tensor] = None,        # (B, T) bool
        node_masks: Optional[torch.Tensor] = None,        # (B, 3)
        elapsed_times: Optional[torch.Tensor] = None,     # (B, 1)
        mode: str = "full",                               # "full", "ctclm_only", "qvbe_only", "classical_relational_only", "zero_quantum"
        force_quantum_suppression: Optional[bool] = None,
        force_ctclm_suppression: Optional[bool] = None,
        force_classical_rel_suppression: Optional[bool] = None,
        return_diagnostics: bool = False,
    ) -> IntegratedOutput:
        """
        Executes end-to-end multimodal forward pass.
        """
        first_node = self.config.qvbe_config.node_names[0]
        batch_size = node_inputs[first_node][0].shape[0]
        device = node_inputs[first_node][0].device

        # Default masks
        if node_masks is None:
            node_masks = torch.ones(batch_size, self.config.qvbe_config.n_nodes, device=device)

        # 1. Compile Graph Tensors via QVBE Compiler
        graph_tensors = self.qvbe.compiler.compile_graph_motif(
            edges=graph_edges,
            node_masks=node_masks,
            elapsed_times=elapsed_times,
            batch_size=batch_size,
        )
        for k, v in graph_tensors.items():
            if isinstance(v, torch.Tensor):
                graph_tensors[k] = v.to(device)

        # 2. Extract Flat Node States (B, 6) from Inputs with explicit missingness masking
        # Channels: [x_mol, dx_mol, x_struct, dx_struct, x_func, dx_func]
        node_states_list = []
        for idx, name in enumerate(self.config.qvbe_config.node_names):
            x_raw, dx_raw = node_inputs[name]
            m = node_masks[:, idx : idx + 1]
            node_states_list.append(x_raw[:, :1] * m)
            node_states_list.append(dx_raw[:, :1] * m)
        node_states_flat = torch.cat(node_states_list, dim=-1) # (B, 6)
        node_states_3d = node_states_flat.reshape(batch_size, 3, 2)

        # 3. Branch Execution according to Mode & Ablations
        qvbe_out = None
        qvbe_emb = None
        suppress_q = force_quantum_suppression
        if mode in ["zero_quantum", "ctclm_only", "classical_relational_only"]:
            suppress_q = True

        if self.config.enable_quantum_branch and not suppress_q:
            qvbe_out = self.qvbe(
                node_inputs=node_inputs,
                graph_edges=graph_edges,
                node_masks=node_masks,
                elapsed_times=elapsed_times,
                return_diagnostics=return_diagnostics,
            )
            qvbe_emb = qvbe_out.motif_embedding

        # CTCLM Temporal Branch
        ctclm_out = None
        ctclm_emb = None
        suppress_t = force_ctclm_suppression
        if mode in ["qvbe_only", "classical_relational_only"]:
            suppress_t = True

        if self.config.enable_ctclm_branch and not suppress_t:
            if ctclm_seq is not None and ctclm_times is not None and ctclm_mask is not None:
                ctclm_out = self.ctclm(ctclm_seq, ctclm_times, ctclm_mask)
                ctclm_emb = ctclm_out.h_final
            else:
                # Fallback: synthesize 2-timestep motor sequence from node states
                # Visit 0: baseline functional, Visit 1: follow-up (baseline + delta)
                x_func_bl = node_inputs["functional"][0][:, :1].unsqueeze(1)
                x_func_fu = (node_inputs["functional"][0][:, :1] + node_inputs["functional"][1][:, :1]).unsqueeze(1)
                syn_seq = torch.cat([x_func_bl, x_func_fu], dim=1) # (B, 2, 1)
                syn_times = torch.zeros(batch_size, 2, device=device)
                if elapsed_times is not None:
                    syn_times[:, 1] = elapsed_times.squeeze(-1)
                else:
                    syn_times[:, 1] = 1.0
                syn_mask = torch.ones(batch_size, 2, dtype=torch.bool, device=device)
                ctclm_out = self.ctclm(syn_seq, syn_times, syn_mask)
                ctclm_emb = ctclm_out.h_final

        # Classical Relational Branch
        cr_emb = None
        suppress_cr = force_classical_rel_suppression
        if mode in ["qvbe_only", "ctclm_only"]:
            suppress_cr = True

        if self.config.enable_classical_relational and not suppress_cr:
            cr_emb = self.classical_relational(
                node_states_3d=node_states_3d,
                edge_weights=graph_tensors["edge_weights"],
                edge_lags=graph_tensors["edge_lags"],
                edge_se=graph_tensors["edge_uncertainties"],
                direction_signs=graph_tensors["direction_signs"],
            )

        # 4. Adaptive QCDF Multimodal Fusion
        edge_uncertainties = graph_tensors.get("edge_uncertainties")
        logits, probs, fused_rep, gate_q, gate_t, gate_n = self.qcdf(
            node_states_flat=node_states_flat,
            qvbe_emb=qvbe_emb,
            ctclm_emb=ctclm_emb,
            classical_rel_emb=cr_emb,
            node_masks=node_masks,
            edge_uncertainties=edge_uncertainties,
            force_quantum_suppression=suppress_q,
            force_ctclm_suppression=suppress_t,
            force_classical_rel_suppression=suppress_cr,
        )

        # 5. Modality Contributions & Diagnostics
        modality_contribs = {
            "molecular": node_masks[:, 0:1] * gate_n,
            "structural": node_masks[:, 1:2] * gate_n,
            "functional": node_masks[:, 2:3] * gate_n + gate_t,
            "quantum_relational": gate_q,
        }

        classical_gate_weight = (gate_t + gate_n) / 2.0

        diag = {
            "mode": mode,
            "gate_quantum_mean": float(gate_q.mean().item()),
            "gate_temporal_mean": float(gate_t.mean().item()),
            "gate_node_mean": float(gate_n.mean().item()),
            "suppressed_quantum": bool(suppress_q),
        }

        return IntegratedOutput(
            logits=logits,
            probabilities=probs,
            fused_representation=fused_rep,
            quantum_gate_weight=gate_q,
            temporal_gate_weight=gate_t,
            node_gate_weight=gate_n,
            classical_gate_weight=classical_gate_weight,
            qvbe_output=qvbe_out,
            ctclm_output=ctclm_out,
            classical_relational_output=cr_emb,
            modality_contributions=modality_contribs,
            availability_masks=node_masks,
            diagnostics=diag,
        )

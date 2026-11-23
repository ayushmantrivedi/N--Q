"""
Adaptive Quantum-Classical Dynamic Fusion (QCDF) for Nāḍī-Q Phase 12.
Provides adaptive, modality-aware fusion with an independently learnable quantum
contribution gate capable of assigning zero or near-zero weight to quantum representations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from nadiq.models.integrated.config import NadiQIntegratedConfig


class AdaptiveQCDF(nn.Module):
    """
    Adaptive Quantum-Classical Dynamic Fusion Layer.
    Fuses QVBE quantum relational representations, CTCLM temporal latent trajectories,
    classical graph relationships, and static node observations.
    """

    def __init__(self, config: NadiQIntegratedConfig):
        super().__init__()
        self.config = config
        proj_dim = config.fusion_projection_dim
        hidden_dim = config.fusion_hidden_dim
        n_nodes = config.qvbe_config.n_nodes
        n_pairs = 3

        # 1. Component Projection Layers into Common Normalized Subspace
        qvbe_dim = config.qvbe_config.motif_embed_dim
        ctclm_dim = config.ctclm_config.latent_dim
        node_dim = n_nodes * 2  # 3 nodes x 2 channels (state, change)
        classical_rel_dim = config.classical_relational_dim

        self.proj_qvbe = nn.Sequential(
            nn.Linear(qvbe_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.proj_ctclm = nn.Sequential(
            nn.Linear(ctclm_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.proj_node = nn.Sequential(
            nn.Linear(node_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.proj_classical_rel = nn.Sequential(
            nn.Linear(classical_rel_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # 2. Context Encoder for Dynamic Gating
        # Context takes node observations + masks + edge uncertainties
        context_dim = node_dim + n_nodes + n_pairs
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
        )

        # 3. Independently Learnable Contribution Gates
        # Quantum gate: can be completely suppressed to 0
        self.quantum_gate_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.quantum_gate_head.bias, config.initial_quantum_gate_bias)

        # Temporal gate
        self.temporal_gate_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.temporal_gate_head.bias, config.initial_temporal_gate_bias)

        # Node/static gate
        self.node_gate_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.node_gate_head.bias, config.initial_node_gate_bias)

        # Classical relational gate
        self.classical_rel_gate_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.classical_rel_gate_head.bias, 0.50)

        # 4. Multimodal Fusion MLP
        # Takes [gated_qvbe, gated_ctclm, gated_node, gated_cr, masks]
        # Total dims = 4 * proj_dim + n_nodes
        fused_in_dim = (4 * proj_dim) + n_nodes
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 5. Downstream Prediction Head
        self.classifier = nn.Linear(hidden_dim, config.num_classes)

    def forward(
        self,
        node_states_flat: torch.Tensor,                          # (B, 6)
        qvbe_emb: Optional[torch.Tensor] = None,                 # (B, qvbe_dim)
        ctclm_emb: Optional[torch.Tensor] = None,                # (B, ctclm_dim)
        classical_rel_emb: Optional[torch.Tensor] = None,        # (B, cr_dim)
        node_masks: Optional[torch.Tensor] = None,               # (B, 3)
        edge_uncertainties: Optional[torch.Tensor] = None,       # (B, 3)
        force_quantum_suppression: Optional[bool] = None,
        force_ctclm_suppression: Optional[bool] = None,
        force_classical_rel_suppression: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes adaptive multimodal dynamic fusion.
        Returns:
            logits, probabilities, fused_vector, gate_q, gate_t, gate_n
        """
        batch_size = node_states_flat.shape[0]
        device = node_states_flat.device
        proj_dim = self.config.fusion_projection_dim

        # Defaults for masks & uncertainties
        if node_masks is None:
            node_masks = torch.ones(batch_size, self.config.qvbe_config.n_nodes, device=device)
        if edge_uncertainties is None:
            edge_uncertainties = torch.zeros(batch_size, 3, device=device)

        # 1. Compute Shared Gating Context
        context_in = torch.cat([node_states_flat, node_masks, edge_uncertainties], dim=-1)
        context = self.context_net(context_in)

        # 2. Compute Individual Contribution Gates
        # Quantum Contribution Gate
        suppress_q = force_quantum_suppression if force_quantum_suppression is not None else self.config.force_quantum_suppression
        if not self.config.enable_quantum_branch or qvbe_emb is None or suppress_q:
            gate_q = torch.zeros(batch_size, 1, device=device)
        else:
            gate_q = torch.sigmoid(self.quantum_gate_head(context))

        # Temporal CTCLM Contribution Gate
        suppress_t = force_ctclm_suppression if force_ctclm_suppression is not None else False
        if not self.config.enable_ctclm_branch or ctclm_emb is None or suppress_t:
            gate_t = torch.zeros(batch_size, 1, device=device)
        else:
            gate_t = torch.sigmoid(self.temporal_gate_head(context))

        # Node/Static Feature Gate
        gate_n = torch.sigmoid(self.node_gate_head(context))

        # Classical Relational Gate
        suppress_cr = force_classical_rel_suppression if force_classical_rel_suppression is not None else False
        if not self.config.enable_classical_relational or classical_rel_emb is None or suppress_cr:
            gate_cr = torch.zeros(batch_size, 1, device=device)
        else:
            gate_cr = torch.sigmoid(self.classical_rel_gate_head(context))

        # 3. Project Representations
        # Quantum branch
        if qvbe_emb is not None and not suppress_q:
            feat_q = self.proj_qvbe(qvbe_emb) * gate_q
        else:
            feat_q = torch.zeros(batch_size, proj_dim, device=device)

        # CTCLM branch
        if ctclm_emb is not None and not suppress_t:
            feat_t = self.proj_ctclm(ctclm_emb) * gate_t
        else:
            feat_t = torch.zeros(batch_size, proj_dim, device=device)

        # Node branch
        feat_n = self.proj_node(node_states_flat) * gate_n

        # Classical relational branch
        if classical_rel_emb is not None and not suppress_cr:
            feat_cr = self.proj_classical_rel(classical_rel_emb) * gate_cr
        else:
            feat_cr = torch.zeros(batch_size, proj_dim, device=device)

        # 4. Fused Representation
        fused_input = torch.cat([feat_q, feat_t, feat_n, feat_cr, node_masks], dim=-1)
        fused_rep = self.fusion_mlp(fused_input)

        # 5. Classification Output
        logits = self.classifier(fused_rep).squeeze(-1)
        probs = torch.sigmoid(logits)

        return logits, probs, fused_rep, gate_q, gate_t, gate_n


class ResidualAdaptiveQCDF(nn.Module):
    """
    Residual Adaptive Quantum-Classical Dynamic Fusion (Phase 12.2).
    Computes:
        z_classical = ClassicalMLP([feat_t, feat_n, feat_cr, masks])
        z_fused = z_classical + alpha * P(z_quantum)
    where alpha is a learnable, bounded parameter in [0, 1] and P is a trainable projection.
    When alpha -> 0 or force_quantum_suppression=True, z_fused strictly equals z_classical.
    """

    def __init__(self, config: NadiQIntegratedConfig):
        super().__init__()
        self.config = config
        proj_dim = config.fusion_projection_dim
        hidden_dim = config.fusion_hidden_dim
        n_nodes = config.qvbe_config.n_nodes
        n_pairs = 3

        qvbe_dim = config.qvbe_config.motif_embed_dim
        ctclm_dim = config.ctclm_config.latent_dim
        node_dim = n_nodes * 2
        classical_rel_dim = config.classical_relational_dim

        # 1. Projections into normalized subspace
        self.proj_qvbe = nn.Sequential(
            nn.Linear(qvbe_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.proj_ctclm = nn.Sequential(
            nn.Linear(ctclm_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.proj_node = nn.Sequential(
            nn.Linear(node_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.proj_classical_rel = nn.Sequential(
            nn.Linear(classical_rel_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # 2. Classical Trunk MLP (aggregates temporal, static, classical relational, and masks)
        classical_in_dim = (3 * proj_dim) + n_nodes
        self.classical_mlp = nn.Sequential(
            nn.Linear(classical_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 3. Residual Quantum Projection into Classical Hidden Subspace
        self.residual_proj_q = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 4. Context Net for Residual Weight Alpha
        context_dim = node_dim + n_nodes + n_pairs
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
        )
        self.alpha_head = nn.Linear(hidden_dim, 1)
        # Conservative initialization: sigmoid(-1.0) ~ 0.27
        nn.init.constant_(self.alpha_head.bias, -1.0)

        # 5. Classifier
        self.classifier = nn.Linear(hidden_dim, config.num_classes)

    def forward(
        self,
        node_states_flat: torch.Tensor,
        qvbe_emb: Optional[torch.Tensor] = None,
        ctclm_emb: Optional[torch.Tensor] = None,
        classical_rel_emb: Optional[torch.Tensor] = None,
        node_masks: Optional[torch.Tensor] = None,
        edge_uncertainties: Optional[torch.Tensor] = None,
        force_quantum_suppression: Optional[bool] = None,
        force_ctclm_suppression: Optional[bool] = None,
        force_classical_rel_suppression: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = node_states_flat.shape[0]
        device = node_states_flat.device
        proj_dim = self.config.fusion_projection_dim

        if node_masks is None:
            node_masks = torch.ones(batch_size, self.config.qvbe_config.n_nodes, device=device)
        if edge_uncertainties is None:
            edge_uncertainties = torch.zeros(batch_size, 3, device=device)

        # 1. Classical Components
        suppress_t = force_ctclm_suppression if force_ctclm_suppression is not None else False
        if ctclm_emb is not None and not suppress_t:
            feat_t = self.proj_ctclm(ctclm_emb)
        else:
            feat_t = torch.zeros(batch_size, proj_dim, device=device)

        feat_n = self.proj_node(node_states_flat)

        suppress_cr = force_classical_rel_suppression if force_classical_rel_suppression is not None else False
        if classical_rel_emb is not None and not suppress_cr:
            feat_cr = self.proj_classical_rel(classical_rel_emb)
        else:
            feat_cr = torch.zeros(batch_size, proj_dim, device=device)

        # 2. Classical Representation Trunk
        classical_in = torch.cat([feat_t, feat_n, feat_cr, node_masks], dim=-1)
        z_classical = self.classical_mlp(classical_in)

        # 3. Dynamic Residual Contribution Weight Alpha
        context_in = torch.cat([node_states_flat, node_masks, edge_uncertainties], dim=-1)
        context = self.context_net(context_in)

        suppress_q = force_quantum_suppression if force_quantum_suppression is not None else self.config.force_quantum_suppression
        if not self.config.enable_quantum_branch or qvbe_emb is None or suppress_q:
            alpha = torch.zeros(batch_size, 1, device=device)
            P_q = torch.zeros_like(z_classical)
        else:
            alpha = torch.sigmoid(self.alpha_head(context))
            feat_q = self.proj_qvbe(qvbe_emb)
            P_q = self.residual_proj_q(feat_q)

        # 4. Residual Fusion
        z_fused = z_classical + alpha * P_q

        # 5. Prediction
        logits = self.classifier(z_fused).squeeze(-1)
        probs = torch.sigmoid(logits)

        # Diagnostic gate weights (alpha as quantum contribution, 1.0 for temporal/node trunk)
        gate_t = torch.ones(batch_size, 1, device=device) * (0.0 if suppress_t else 1.0)
        gate_n = torch.ones(batch_size, 1, device=device)

        return logits, probs, z_fused, alpha, gate_t, gate_n


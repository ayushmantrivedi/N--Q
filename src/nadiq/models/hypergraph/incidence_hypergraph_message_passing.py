"""
NĀḌĪ-Q Phase 10B: Incidence-Aware Temporal Hypergraph Message Passing.
Enforces strict node-to-hyperedge incidence relations (v in e), aggregates messages,
updates node states with residual normalization, and integrates quantum relational drift D_Q.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class IncidenceHypergraphMessagePassingLayer(nn.Module):
    """
    Incidence-Aware Hypergraph Message Passing Layer.
    Only allows message passing between hyperedges and their legitimate incident nodes.
    
    Node set V = {Mol, Struct, Func}
    Hyperedge set E = {Intra_Mol, Intra_Struct, Intra_Func, Cross_Modal}
    Incidence Matrix H in {0, 1}^(3 x 4):
      Mol    : [1, 0, 0, 1]
      Struct : [0, 1, 0, 1]
      Func   : [0, 0, 1, 1]
    """

    def __init__(
        self,
        dim_mol: int = 15,
        dim_struct: int = 36,
        dim_func: int = 35,
        dim_cross_edge: int = 24,
        dim_hidden: int = 32
    ):
        super().__init__()
        self.dim_hidden = dim_hidden

        # Node projection heads: raw features -> latent space
        self.proj_in_mol = nn.Linear(dim_mol, dim_hidden)
        self.proj_in_struct = nn.Linear(dim_struct, dim_hidden)
        self.proj_in_func = nn.Linear(dim_func, dim_hidden)

        # Intra-hyperedge encoders: node -> intra_edge (dim_hidden)
        self.intra_mol_head = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.GELU())
        self.intra_struct_head = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.GELU())
        self.intra_func_head = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.GELU())

        # Hyperedge-to-Node message projections
        # Cross-modal hyperedge (dim_cross_edge -> dim_hidden)
        self.msg_cross2mol = nn.Sequential(nn.Linear(dim_cross_edge, dim_hidden), nn.GELU())
        self.msg_cross2struct = nn.Sequential(nn.Linear(dim_cross_edge, dim_hidden), nn.GELU())
        self.msg_cross2func = nn.Sequential(nn.Linear(dim_cross_edge, dim_hidden), nn.GELU())

        # Intra-edge to Node message projections
        self.msg_intra2mol = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.GELU())
        self.msg_intra2struct = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.GELU())
        self.msg_intra2func = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.GELU())

        # Layer norms for node residual updates
        self.norm_mol = nn.LayerNorm(dim_hidden)
        self.norm_struct = nn.LayerNorm(dim_hidden)
        self.norm_func = nn.LayerNorm(dim_hidden)

        # Graph-level pooling readout: includes updated nodes, cross-modal edge, and quantum drift D_Q
        # Input dim: 3 * dim_hidden (nodes) + dim_cross_edge (cross edge) + 1 (D_Q)
        readout_in_dim = 3 * dim_hidden + dim_cross_edge + 1
        self.graph_readout = nn.Sequential(
            nn.Linear(readout_in_dim, dim_hidden),
            nn.GELU(),
            nn.LayerNorm(dim_hidden)
        )

    def forward(
        self,
        h_mol_raw: torch.Tensor,
        h_struct_raw: torch.Tensor,
        h_func_raw: torch.Tensor,
        e_cross_rep: torch.Tensor,
        drift_dq: Optional[torch.Tensor] = None,
        avail_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Executes incidence-aware message passing.
        Returns:
            updated_nodes_dict: updated latent vectors for mol, struct, func
            g_t: pooled visit graph embedding
            pre_msg_edges: intra-edges prior to message passing (for perturbation tests)
        """
        is_1d = (h_mol_raw.ndim == 1)
        if is_1d:
            h_mol_raw = h_mol_raw.unsqueeze(0)
            h_struct_raw = h_struct_raw.unsqueeze(0)
            h_func_raw = h_func_raw.unsqueeze(0)
            e_cross_rep = e_cross_rep.unsqueeze(0)

        B = h_mol_raw.shape[0]
        device = h_mol_raw.device

        if drift_dq is None:
            drift_dq = torch.zeros(B, 1, device=device)
        elif is_1d and drift_dq.ndim == 0:
            drift_dq = drift_dq.reshape(1, 1)

        # 1. Project nodes into latent space
        v_mol = self.proj_in_mol(h_mol_raw)
        v_struct = self.proj_in_struct(h_struct_raw)
        v_func = self.proj_in_func(h_func_raw)

        # 2. Form intra-modality hyperedges prior to message passing
        e_intra_mol = self.intra_mol_head(v_mol)
        e_intra_struct = self.intra_struct_head(v_struct)
        e_intra_func = self.intra_func_head(v_func)

        pre_msg_edges = {
            "intra_mol": e_intra_mol.squeeze(0) if is_1d else e_intra_mol,
            "intra_struct": e_intra_struct.squeeze(0) if is_1d else e_intra_struct,
            "intra_func": e_intra_func.squeeze(0) if is_1d else e_intra_func,
        }

        # 3. Compute hyperedge-to-node messages based on incidence structure
        # Mol node receives from e_intra_mol AND e_cross
        msg_mol = self.msg_intra2mol(e_intra_mol) + self.msg_cross2mol(e_cross_rep)
        # Struct node receives from e_intra_struct AND e_cross
        msg_struct = self.msg_intra2struct(e_intra_struct) + self.msg_cross2struct(e_cross_rep)
        # Func node receives from e_intra_func AND e_cross
        msg_func = self.msg_intra2func(e_intra_func) + self.msg_cross2func(e_cross_rep)

        # 4. Apply modality availability masking
        if avail_mask is not None:
            msg_mol = msg_mol * avail_mask[:, 0:1]
            msg_struct = msg_struct * avail_mask[:, 1:2]
            msg_func = msg_func * avail_mask[:, 2:3]

        # 5. Residual Node Update + LayerNorm
        v_mol_up = self.norm_mol(v_mol + msg_mol)
        v_struct_up = self.norm_struct(v_struct + msg_struct)
        v_func_up = self.norm_func(v_func + msg_func)

        # 6. Visit Graph Pooling
        concat_visit = torch.cat([v_mol_up, v_struct_up, v_func_up, e_cross_rep, drift_dq], dim=-1)
        g_t = self.graph_readout(concat_visit)

        nodes_out = {
            "molecular": v_mol_up.squeeze(0) if is_1d else v_mol_up,
            "structural": v_struct_up.squeeze(0) if is_1d else v_struct_up,
            "functional": v_func_up.squeeze(0) if is_1d else v_func_up,
            "msg_mol": msg_mol.squeeze(0) if is_1d else msg_mol,
            "msg_struct": msg_struct.squeeze(0) if is_1d else msg_struct,
            "msg_func": msg_func.squeeze(0) if is_1d else msg_func,
        }

        if is_1d:
            return nodes_out, g_t.squeeze(0), pre_msg_edges
        return nodes_out, g_t, pre_msg_edges

"""
NĀḌĪ-Q Phase 10A: Hypergraph Message Passing Network.
Propagates information between Nodes and Hyperedges, updating constituent representations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphMessagePassingLayer(nn.Module):
    """
    Executes bidirectional message passing between Nodes and Hyperedges:
    1. Incident hyperedges send updated messages to member nodes.
    2. Nodes update their latent states via residual normalization.
    3. Aggregates a visit-level graph embedding.
    """

    def __init__(
        self,
        dim_mol: int = 15,
        dim_struct: int = 36,
        dim_func: int = 35,
        dim_edge: int = 23,
        dim_hidden: int = 32
    ):
        super().__init__()
        self.dim_hidden = dim_hidden

        # Node projection heads
        self.proj_in_mol = nn.Linear(dim_mol, dim_hidden)
        self.proj_in_struct = nn.Linear(dim_struct, dim_hidden)
        self.proj_in_func = nn.Linear(dim_func, dim_hidden)

        # Hyperedge-to-Node message projections
        self.msg_e2mol = nn.Sequential(nn.Linear(dim_edge, dim_hidden), nn.GELU())
        self.msg_e2struct = nn.Sequential(nn.Linear(dim_edge, dim_hidden), nn.GELU())
        self.msg_e2func = nn.Sequential(nn.Linear(dim_edge, dim_hidden), nn.GELU())

        # Layer norms for node updates
        self.norm_mol = nn.LayerNorm(dim_hidden)
        self.norm_struct = nn.LayerNorm(dim_hidden)
        self.norm_func = nn.LayerNorm(dim_hidden)

        # Graph-level pooling readout
        self.graph_readout = nn.Sequential(
            nn.Linear(3 * dim_hidden + dim_edge, dim_hidden),
            nn.GELU(),
            nn.LayerNorm(dim_hidden)
        )

    def forward(
        self,
        h_mol_raw: torch.Tensor,
        h_struct_raw: torch.Tensor,
        h_func_raw: torch.Tensor,
        e_cross_rep: torch.Tensor,
        avail_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Executes hyperedge-to-node message passing.
        returns: (updated_nodes_dict, visit_graph_embedding g_t)
        """
        is_1d = (h_mol_raw.ndim == 1)
        if is_1d:
            h_mol_raw = h_mol_raw.unsqueeze(0)
            h_struct_raw = h_struct_raw.unsqueeze(0)
            h_func_raw = h_func_raw.unsqueeze(0)
            e_cross_rep = e_cross_rep.unsqueeze(0)

        # Initial projected node states
        v_mol = self.proj_in_mol(h_mol_raw)
        v_struct = self.proj_in_struct(h_struct_raw)
        v_func = self.proj_in_func(h_func_raw)

        # Compute hyperedge-to-node messages
        msg_mol = self.msg_e2mol(e_cross_rep)
        msg_struct = self.msg_e2struct(e_cross_rep)
        msg_func = self.msg_e2func(e_cross_rep)

        # Apply availability masking if provided
        if avail_mask is not None:
            # avail_mask: (B, 3) [mol, struct, func]
            msg_mol = msg_mol * avail_mask[:, 0:1]
            msg_struct = msg_struct * avail_mask[:, 1:2]
            msg_func = msg_func * avail_mask[:, 2:3]

        # Node state updates (Residual addition + LayerNorm)
        v_mol_up = self.norm_mol(v_mol + msg_mol)
        v_struct_up = self.norm_struct(v_struct + msg_struct)
        v_func_up = self.norm_func(v_func + msg_func)

        # Graph pooling: aggregate updated constituent nodes and hyperedge representation
        concat_visit = torch.cat([v_mol_up, v_struct_up, v_func_up, e_cross_rep], dim=-1)
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
            return nodes_out, g_t.squeeze(0)
        return nodes_out, g_t

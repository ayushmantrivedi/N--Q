"""
NĀḌĪ-Q Numerical Graph-to-Quantum Compiler.
Translates symbolic biological relationship motifs and evidence metadata
into numerical interaction matrices and Hamiltonian coupling tensors.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class NumericalGraphCompiler:
    """
    Consumes structured relationship graphs and motifs from MeTTa / statistical detectors,
    validates missingness, and compiles numerical interaction tensors.
    """

    STATUS_SCALING = {
        "Candidate": 0.20,
        "Provisional": 0.60,
        "Validated": 1.00,
    }

    def __init__(
        self,
        node_names: List[str],
        decay_lambda: float = 0.10,
        max_coupling_angle: float = 1.5708,
    ):
        self.node_names = list(node_names)
        self.node_to_idx = {name: i for i, name in enumerate(self.node_names)}
        self.n_nodes = len(self.node_names)
        self.decay_lambda = decay_lambda
        self.max_coupling_angle = max_coupling_angle

    def compile_graph_motif(
        self,
        edges: List[Dict[str, Any]],
        node_masks: Optional[torch.Tensor] = None,
        elapsed_times: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compiles edge list into batch-compatible interaction tensors:
        - edge_weights: (B, n_pairs)
        - edge_lags: (B, n_pairs)
        - edge_masks: (B, n_pairs)
        - direction_signs: (B, n_pairs)
        - triad_active: (B, 1)
        - triad_weight: (B, 1)
        - triad_lag: (B, 1)
        """
        # Node pairs (canonical ordering: (0,1), (1,2), (0,2) for 3 nodes)
        pairs = []
        pair_to_idx = {}
        idx = 0
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                pairs.append((i, j))
                pair_to_idx[(i, j)] = idx
                idx += 1
        n_pairs = len(pairs)

        adj_matrix = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        weights_arr = np.zeros(n_pairs, dtype=np.float32)
        lags_arr = np.zeros(n_pairs, dtype=np.float32)
        masks_arr = np.zeros(n_pairs, dtype=np.float32)
        dir_arr = np.zeros(n_pairs, dtype=np.float32)
        se_arr = np.zeros(n_pairs, dtype=np.float32)
        conf_arr = np.zeros(n_pairs, dtype=np.float32)

        # Process each edge
        for edge in edges:
            src = edge.get("src")
            dst = edge.get("dst")
            if src not in self.node_to_idx or dst not in self.node_to_idx:
                continue

            u = self.node_to_idx[src]
            v = self.node_to_idx[dst]
            pair_key = (min(u, v), max(u, v))
            if pair_key not in pair_to_idx:
                continue

            p_idx = pair_to_idx[pair_key]

            # Extract edge attributes
            r_eff = float(edge.get("r", 0.0))
            status = edge.get("status", "Candidate")
            status_scale = self.STATUS_SCALING.get(status, 0.20)
            lag = float(edge.get("lag", 1.0))
            se = float(edge.get("se", 0.10))
            p_val = float(edge.get("p", 0.05))

            # Direction: +1 if u < v matches src -> dst, -1 otherwise
            direction = 1.0 if u < v else -1.0

            # Confidence factor: higher when p-value is small and SE is low
            conf = float(np.clip(1.0 - p_val, 0.10, 1.0) / (1.0 + 2.0 * max(0.0, se)))

            weights_arr[p_idx] = r_eff * status_scale
            lags_arr[p_idx] = max(0.0, lag)
            masks_arr[p_idx] = 1.0
            dir_arr[p_idx] = direction
            se_arr[p_idx] = se
            conf_arr[p_idx] = conf

            adj_matrix[u, v] = 1.0
            adj_matrix[v, u] = 1.0

        # Check for 3-node motif (triad) between nodes 0, 1, and 2
        triad_active = 0.0
        triad_w = 0.0
        triad_l = 0.0
        if self.n_nodes >= 3:
            p_01 = pair_to_idx.get((0, 1), 0)
            p_12 = pair_to_idx.get((1, 2), 2)
            has_chain = (masks_arr[p_01] > 0.5) and (masks_arr[p_12] > 0.5)
            if has_chain:
                triad_active = 1.0
                triad_w = float(np.sqrt(abs(weights_arr[p_01] * weights_arr[p_12]) + 1e-6))
                triad_l = float(lags_arr[p_01] + lags_arr[p_12])

        # Convert to batch tensors
        w_t = torch.tensor(weights_arr, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        l_t = torch.tensor(lags_arr, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        m_t = torch.tensor(masks_arr, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        d_t = torch.tensor(dir_arr, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        se_t = torch.tensor(se_arr, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        conf_t = torch.tensor(conf_arr, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        triad_act_t = torch.tensor([[triad_active]], dtype=torch.float32).repeat(batch_size, 1)
        triad_w_t = torch.tensor([[triad_w]], dtype=torch.float32).repeat(batch_size, 1)
        triad_l_t = torch.tensor([[triad_l]], dtype=torch.float32).repeat(batch_size, 1)

        # Apply elapsed time overrides if per-sample elapsed times provided
        if elapsed_times is not None:
            if elapsed_times.ndim == 1:
                elapsed_times = elapsed_times.unsqueeze(-1)
            l_t = elapsed_times.repeat(1, n_pairs)

        # Apply missingness masking:
        # If node i is unavailable (node_masks[:, i] == 0), zero all incident edge weights and triad
        if node_masks is not None:
            for p_idx, (u, v) in enumerate(pairs):
                edge_avail = (node_masks[:, u] * node_masks[:, v]).unsqueeze(-1)
                w_t[:, p_idx : p_idx + 1] = w_t[:, p_idx : p_idx + 1] * edge_avail
                m_t[:, p_idx : p_idx + 1] = m_t[:, p_idx : p_idx + 1] * edge_avail
                conf_t[:, p_idx : p_idx + 1] = conf_t[:, p_idx : p_idx + 1] * edge_avail

            if self.n_nodes >= 3:
                triad_avail = (node_masks[:, 0] * node_masks[:, 1] * node_masks[:, 2]).unsqueeze(-1)
                triad_act_t = triad_act_t * triad_avail
                triad_w_t = triad_w_t * triad_avail

        return {
            "edge_weights": w_t,
            "edge_lags": l_t,
            "edge_masks": m_t,
            "direction_signs": d_t,
            "edge_uncertainties": se_t,
            "edge_confidences": conf_t,
            "triad_active": triad_act_t,
            "triad_weight": triad_w_t,
            "triad_lag": triad_l_t,
            "pairs": pairs,
        }

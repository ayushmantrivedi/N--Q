"""
NĀḌĪ-Q QVBE-v4 Quantum Circuit Architecture.
Implements Graph-Conditioned Hamiltonian evolution, pairwise and change-channel couplings,
triad higher-order interactions, and subsystem entanglement diagnostics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

from nadiq.models.qvbe_v4.config import QVBEv4Config


class QVBEv4Circuit(nn.Module):
    """
    Differentiable Graph-Conditioned Temporal Quantum Relational Circuit.
    """

    def __init__(self, config: QVBEv4Config):
        super().__init__()
        self.config = config
        self.n_qubits = config.total_qubits
        self.n_nodes = config.n_nodes
        self.dev = qml.device(config.device_name, wires=self.n_qubits)

        # Number of node pairs: for 3 nodes, n_pairs = 3 -> (0,1), (1,2), (0,2)
        self.pairs = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                self.pairs.append((i, j))
        self.n_pairs = len(self.pairs)

        # Trainable Variational Parameters
        # 1. Node local scaling and bias: (n_nodes, 2, 2) -> (x, dx) x (scale, bias)
        self.node_scale = nn.Parameter(torch.ones(self.n_nodes, 2))
        self.node_bias = nn.Parameter(torch.zeros(self.n_nodes, 2))

        # 2. Pairwise state coupling weights: (n_layers, n_pairs)
        self.theta_pair = nn.Parameter(
            torch.randn(self.config.n_layers, self.n_pairs) * self.config.init_coupling_std
        )

        # 3. Pairwise change coupling weights: (n_layers, n_pairs)
        self.phi_pair = nn.Parameter(
            torch.randn(self.config.n_layers, self.n_pairs) * self.config.init_coupling_std
        )

        # 4. Directional coupling weights: (n_layers, n_pairs)
        self.dir_weights = nn.Parameter(
            torch.randn(self.config.n_layers, self.n_pairs) * self.config.init_coupling_std
        )

        # 5. Higher-order triad coupling weights: (n_layers, 1)
        self.omega_triad = nn.Parameter(
            torch.randn(self.config.n_layers, 1) * self.config.init_coupling_std
        )

        # 6. Transverse mixing field: (n_layers, n_qubits)
        self.w_transverse = nn.Parameter(
            torch.full((self.config.n_layers, self.n_qubits), self.config.transverse_mixing_angle)
        )

        # 7. Learned Temporal Decay & Edge Scale/Bias (Phase 11C)
        self.log_lambda_decay = nn.Parameter(
            torch.full((self.n_pairs,), math.log(max(1e-4, self.config.decay_lambda)))
        )
        self.edge_scale = nn.Parameter(torch.ones(self.n_pairs))
        self.edge_bias = nn.Parameter(torch.zeros(self.n_pairs))

        # 8. Variational local rotations: (n_layers, n_qubits, 3)
        self.var_rotations = nn.Parameter(
            torch.randn(self.config.n_layers, self.n_qubits, 3) * self.config.near_identity_init_std
        )

        # Build vectorized PennyLane QNode
        self._qnode = self._create_qnode()

    def _create_qnode(self):
        """Constructs the batch-vectorized PennyLane QNode."""
        config = self.config
        n_qubits = self.n_qubits
        n_nodes = self.n_nodes
        pairs = self.pairs

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit_fn(
            node_states,        # (B, n_nodes, 2) -> (x_i, dx_i)
            edge_weights,       # (B, n_pairs)
            edge_lags,          # (B, n_pairs)
            edge_masks,         # (B, n_pairs)
            direction_signs,    # (B, n_pairs)
            edge_confidences,   # (B, n_pairs)
            triad_active,       # (B, 1)
            triad_weight,       # (B, 1)
            triad_lag,          # (B, 1)
            node_scale,         # (n_nodes, 2)
            node_bias,          # (n_nodes, 2)
            theta_pair,         # (n_layers, n_pairs)
            phi_pair,           # (n_layers, n_pairs)
            dir_weights,        # (n_layers, n_pairs)
            omega_triad,        # (n_layers, 1)
            w_transverse,       # (n_layers, n_qubits)
            log_lambda_decay,   # (n_pairs,)
            edge_scale,         # (n_pairs,)
            edge_bias,          # (n_pairs,)
            var_rotations,      # (n_layers, n_qubits, 3)
        ):
            # Trotterized Layers
            for l in range(config.n_layers):
                # -------------------------------------------------------------
                # A. Node State Encoding (RY and RZ)
                # -------------------------------------------------------------
                for k in range(n_nodes):
                    q_state, q_change = config.get_node_qubits(k)

                    # State feature x_k
                    val_x = node_states[:, k, 0] * node_scale[k, 0] + node_bias[k, 0]
                    qml.RY(val_x, wires=q_state)
                    qml.RZ(val_x, wires=q_state)

                    # Temporal change feature dx_k
                    val_dx = node_states[:, k, 1] * node_scale[k, 1] + node_bias[k, 1]
                    qml.RY(val_dx, wires=q_change)
                    qml.RZ(val_dx, wires=q_change)

                # -------------------------------------------------------------
                # B. Transverse Mixing Field (RX)
                # -------------------------------------------------------------
                for q in range(n_qubits):
                    qml.RX(w_transverse[l, q], wires=q)

                # -------------------------------------------------------------
                # C. Pairwise Graph Relationship Encoding
                # -------------------------------------------------------------
                if config.enable_pairwise_coupling:
                    for p_idx, (u, v) in enumerate(pairs):
                        qu_state, _ = config.get_node_qubits(u)
                        qv_state, _ = config.get_node_qubits(v)

                        # Effective coupling angle: bounded by tanh
                        w_uv = edge_weights[:, p_idx] * edge_masks[:, p_idx]
                        if config.enable_uncertainty_weighting and edge_confidences is not None:
                            w_uv = w_uv * edge_confidences[:, p_idx]

                        if config.edge_coupling_mode == "learned_scale":
                            w_uv = (w_uv * edge_scale[p_idx] + edge_bias[p_idx]) * edge_masks[:, p_idx]

                        if config.enable_time_conditioning:
                            if config.enable_learned_temporal_decay:
                                lam = torch.exp(log_lambda_decay[p_idx])
                            else:
                                lam = config.decay_lambda

                            if config.temporal_decay_mode == "rational":
                                decay = 1.0 / (1.0 + lam * edge_lags[:, p_idx])
                            else:
                                decay = torch.exp(-lam * edge_lags[:, p_idx])
                        else:
                            decay = 1.0

                        angle_state = torch.tanh(theta_pair[l, p_idx] * w_uv * decay) * config.max_coupling_angle
                        qml.IsingZZ(angle_state, wires=[qu_state, qv_state])

                        # Change-channel coupling (IsingXX between change qubits)
                        if config.enable_change_channel_coupling:
                            _, qu_change = config.get_node_qubits(u)
                            _, qv_change = config.get_node_qubits(v)
                            angle_change = torch.tanh(phi_pair[l, p_idx] * w_uv * decay) * config.max_coupling_angle
                            qml.IsingXX(angle_change, wires=[qu_change, qv_change])

                        # Direction-aware interaction
                        if config.enable_direction_coupling:
                            d_sign = direction_signs[:, p_idx]
                            angle_dir = torch.tanh(dir_weights[l, p_idx] * w_uv * d_sign) * (config.max_coupling_angle * 0.5)
                            qml.CRY(angle_dir, wires=[qu_state, qv_state])

                # -------------------------------------------------------------
                # C. Higher-Order 3-Node Motif Encoding
                # -------------------------------------------------------------
                if config.enable_higher_order_coupling and n_nodes >= 3:
                    q0_state, _ = config.get_node_qubits(0)
                    q1_state, _ = config.get_node_qubits(1)
                    q2_state, _ = config.get_node_qubits(2)

                    if config.enable_time_conditioning:
                        decay_triad = torch.exp(-config.decay_lambda * triad_lag[:, 0])
                    else:
                        decay_triad = 1.0

                    angle_triad = (
                        torch.tanh(omega_triad[l, 0] * triad_weight[:, 0] * triad_active[:, 0] * decay_triad)
                        * config.max_coupling_angle
                    )
                    qml.MultiRZ(angle_triad, wires=[q0_state, q1_state, q2_state])

                # -------------------------------------------------------------
                # D. Local Variational Rotations
                # -------------------------------------------------------------
                for q in range(n_qubits):
                    qml.RX(var_rotations[l, q, 0], wires=q)
                    qml.RY(var_rotations[l, q, 1], wires=q)
                    qml.RZ(var_rotations[l, q, 2], wires=q)

            # Measurements
            # 1. Local Pauli-Z, Pauli-X, Pauli-Y
            meas_z = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            meas_x = [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
            meas_y = [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

            # 2. Pairwise state-state ZZ correlations
            corrs_zz = []
            for u, v in pairs:
                qu_state, _ = config.get_node_qubits(u)
                qv_state, _ = config.get_node_qubits(v)
                corrs_zz.append(qml.expval(qml.PauliZ(qu_state) @ qml.PauliZ(qv_state)))

            # 3. Pairwise change-change XX correlations
            corrs_xx = []
            for u, v in pairs:
                _, qu_change = config.get_node_qubits(u)
                _, qv_change = config.get_node_qubits(v)
                corrs_xx.append(qml.expval(qml.PauliX(qu_change) @ qml.PauliX(qv_change)))

            # 4. Triad joint expectation <Z_0 Z_2 Z_4>
            q0_state, _ = config.get_node_qubits(0)
            q1_state, _ = config.get_node_qubits(1)
            q2_state, _ = config.get_node_qubits(2)
            triad_meas = [qml.expval(qml.PauliZ(q0_state) @ qml.PauliZ(q1_state) @ qml.PauliZ(q2_state))]

            return meas_z + meas_x + meas_y + corrs_zz + corrs_xx + triad_meas

        return circuit_fn

    def forward(
        self,
        node_states: torch.Tensor,
        graph_tensors: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Executes quantum evolution and decomposes measurement returns.
        """
        edge_confidences = graph_tensors.get("edge_confidences", torch.ones_like(graph_tensors["edge_weights"]))
        raw_res = self._qnode(
            node_states,
            graph_tensors["edge_weights"],
            graph_tensors["edge_lags"],
            graph_tensors["edge_masks"],
            graph_tensors["direction_signs"],
            edge_confidences,
            graph_tensors["triad_active"],
            graph_tensors["triad_weight"],
            graph_tensors["triad_lag"],
            self.node_scale,
            self.node_bias,
            self.theta_pair,
            self.phi_pair,
            self.dir_weights,
            self.omega_triad,
            self.w_transverse,
            self.log_lambda_decay,
            self.edge_scale,
            self.edge_bias,
            self.var_rotations,
        )

        n_q = self.n_qubits
        n_p = self.n_pairs

        # Unpack measurement list
        idx = 0
        z_vals = torch.stack(raw_res[idx : idx + n_q], dim=-1).float()
        idx += n_q
        x_vals = torch.stack(raw_res[idx : idx + n_q], dim=-1).float()
        idx += n_q
        y_vals = torch.stack(raw_res[idx : idx + n_q], dim=-1).float()
        idx += n_q
        zz_vals = torch.stack(raw_res[idx : idx + n_p], dim=-1).float()
        idx += n_p
        xx_vals = torch.stack(raw_res[idx : idx + n_p], dim=-1).float()
        idx += n_p
        triad_val = torch.stack(raw_res[idx : idx + 1], dim=-1).float()

        # Compute connected correlations: C_ij = <Z_i Z_j> - <Z_i><Z_j>
        connected_list = []
        for p_idx, (u, v) in enumerate(self.pairs):
            qu_state, _ = self.config.get_node_qubits(u)
            qv_state, _ = self.config.get_node_qubits(v)
            c_ij = zz_vals[:, p_idx] - (z_vals[:, qu_state] * z_vals[:, qv_state])
            connected_list.append(c_ij)
        connected_corrs = torch.stack(connected_list, dim=-1)

        # Compute 3-body cumulant
        q0_s, _ = self.config.get_node_qubits(0)
        q1_s, _ = self.config.get_node_qubits(1)
        q2_s, _ = self.config.get_node_qubits(2)
        z0, z1, z2 = z_vals[:, q0_s], z_vals[:, q1_s], z_vals[:, q2_s]
        zz01 = zz_vals[:, 0]
        zz12 = zz_vals[:, 1]
        zz02 = zz_vals[:, 2]
        cumulant_3 = triad_val.squeeze(-1) - (zz01 * z2 + zz12 * z0 + zz02 * z1) + 2 * (z0 * z1 * z2)

        return {
            "expval_z": z_vals,
            "expval_x": x_vals,
            "expval_y": y_vals,
            "corrs_zz": zz_vals,
            "corrs_xx": xx_vals,
            "connected_corrs": connected_corrs,
            "triad_expval": triad_val,
            "cumulant_3": cumulant_3.unsqueeze(-1),
        }

    def compute_subsystem_entropy(self, node_states: torch.Tensor, graph_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Diagnostics only: computes reduced density matrix von Neumann entropy S(rho_node) in simulator mode.
        """
        # For small 6-qubit simulator, estimate subsystem entanglement from purity/entropy
        entropies = []
        for k in range(self.n_nodes):
            q_s, q_c = self.config.get_node_qubits(k)
            # Entropy heuristic from local polarization: S approx 1 - <P_i>^2
            edge_confidences = graph_tensors.get("edge_confidences", torch.ones_like(graph_tensors["edge_weights"]))
            pol_s = self._qnode(
                node_states, graph_tensors["edge_weights"], graph_tensors["edge_lags"],
                graph_tensors["edge_masks"], graph_tensors["direction_signs"],
                edge_confidences,
                graph_tensors["triad_active"], graph_tensors["triad_weight"], graph_tensors["triad_lag"],
                self.node_scale, self.node_bias, self.theta_pair, self.phi_pair,
                self.dir_weights, self.omega_triad, self.w_transverse,
                self.log_lambda_decay, self.edge_scale, self.edge_bias,
                self.var_rotations,
            )
            z_s = pol_s[q_s]
            x_s = pol_s[self.n_qubits + q_s]
            y_s = pol_s[2 * self.n_qubits + q_s]
            purity = torch.clamp((z_s**2 + x_s**2 + y_s**2), 0.0, 1.0)
            ent = 1.0 - purity
            entropies.append(ent)
        return torch.stack(entropies, dim=-1)

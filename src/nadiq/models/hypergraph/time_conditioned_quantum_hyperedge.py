"""
NĀḌĪ-Q Phase 10B: Time-Conditioned Quantum Relational Hyperedge Encoder.
Implements time-conditioned 2-body RZZ and 3-body RZZZ cross-modal quantum interactions,
readout of 2-body connected correlations and 3-body connected cumulants, and state-vector extraction
for quantum relational drift computation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeConditionedQuantumHyperedgeEncoder(nn.Module):
    """
    Time-Conditioned Quantum Relational Hyperedge Encoder.
    Encodes current node representations, observation change delta_x, elapsed time delta_t,
    and modality availability into an 8-qubit register.
    """

    def __init__(
        self,
        dim_mol: int = 15,
        dim_struct: int = 36,
        dim_func: int = 35,
        n_qubits: int = 8,
        n_layers: int = 2,
        entangle: bool = True,
        enable_time_conditioning: bool = True,
        enable_three_body: bool = True,
        is_classical: bool = False,
        seed: int = 42
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangle = entangle
        self.enable_time_conditioning = enable_time_conditioning
        self.enable_three_body = enable_three_body
        self.is_classical = is_classical

        # Modality qubit allocation:
        # Qubits 0, 1, 2: Molecular (3 qubits)
        # Qubits 3, 4: Structural MRI (2 qubits)
        # Qubits 5, 6, 7: Functional Motor (3 qubits)
        self.q_mol = [0, 1, 2]
        self.q_struct = [3, 4]
        self.q_func = [5, 6, 7]

        # Trainable input adapters: [current_feature ; delta_feature] -> qubit register
        self.proj_mol = nn.Linear(dim_mol * 2, len(self.q_mol))
        self.proj_struct = nn.Linear(dim_struct * 2, len(self.q_struct))
        self.proj_func = nn.Linear(dim_func * 2, len(self.q_func))

        if is_classical:
            # Classical baseline MLP: processes [x, delta_x, dt, avail] -> 24-dim representation
            total_in = (dim_mol + dim_struct + dim_func) * 2 + 1 + 3
            self.classical_mlp = nn.Sequential(
                nn.Linear(total_in, 48),
                nn.GELU(),
                nn.Linear(48, 24)
            )
            self._circuit_fn = None
            self._state_fn = None
            self.u3_weights = None
        else:
            self.classical_mlp = None
            torch.manual_seed(seed)
            # Parameterized rotation angles per layer per qubit
            self.u3_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.05)

            # Trainable 2-body coupling parameters: theta_0 and theta_t
            # Pairs: (2, 3) [Mol-Struct], (4, 5) [Struct-Func], (7, 0) [Func-Mol]
            self.theta_2body_0 = nn.Parameter(torch.tensor([0.4, 0.4, 0.4], dtype=torch.float32))
            self.theta_2body_t = nn.Parameter(torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32))

            # Trainable 3-body coupling parameters: phi_0 and phi_t (qubits 2, 4, 5)
            self.phi_3body_0 = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
            self.phi_3body_t = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

            self.dev = qml.device("default.qubit", wires=n_qubits)
            self._circuit_fn, self._state_fn = self._build_qnodes()

    def _build_qnodes(self):
        n = self.n_qubits
        n_layers = self.n_layers
        entangle = self.entangle
        enable_tc = self.enable_time_conditioning
        enable_3b = self.enable_three_body

        def _apply_circuit(inputs, u3_w, dt_val, th0, th_t, p0, p_t):
            # inputs shape: (B, 8) or (8,)
            is_batched = (inputs.ndim == 2)
            # 1. Superposition initialization
            for i in range(n):
                qml.Hadamard(wires=i)

            # 2. Angle encoding
            for i in range(n):
                in_val = inputs[:, i] if is_batched else inputs[i]
                angle = torch.sigmoid(in_val) * math.pi
                qml.RY(angle, wires=i)
                qml.RZ(angle * 0.5, wires=i)

            # 3. Variational layers with time-conditioned entangling
            for l in range(n_layers):
                for i in range(n):
                    qml.Rot(u3_w[l, i, 0], u3_w[l, i, 1], u3_w[l, i, 2], wires=i)

                if entangle:
                    # Intra-modality entanglement
                    qml.CNOT(wires=[0, 1])
                    qml.CNOT(wires=[1, 2])
                    qml.CNOT(wires=[3, 4])
                    qml.CNOT(wires=[5, 6])
                    qml.CNOT(wires=[6, 7])

                    # Time-conditioned 2-body RZZ gates: exp(-i theta Z_i Z_j / 2)
                    dt_scalar = dt_val if enable_tc else (torch.zeros_like(dt_val) if is_batched else 0.0)
                    th_ms = th0[0] + th_t[0] * dt_scalar
                    th_sf = th0[1] + th_t[1] * dt_scalar
                    th_fm = th0[2] + th_t[2] * dt_scalar

                    # RZZ(2, 3) [Mol -> Struct]
                    qml.CNOT(wires=[2, 3])
                    qml.RZ(th_ms, wires=3)
                    qml.CNOT(wires=[2, 3])

                    # RZZ(4, 5) [Struct -> Func]
                    qml.CNOT(wires=[4, 5])
                    qml.RZ(th_sf, wires=5)
                    qml.CNOT(wires=[4, 5])

                    # RZZ(7, 0) [Func -> Mol]
                    qml.CNOT(wires=[7, 0])
                    qml.RZ(th_fm, wires=0)
                    qml.CNOT(wires=[7, 0])

                    # 3-body RZZZ interaction on (2, 4, 5) if enabled
                    if enable_3b:
                        phi_val = p0 + p_t * dt_scalar
                        qml.CNOT(wires=[2, 4])
                        qml.CNOT(wires=[4, 5])
                        qml.RZ(phi_val, wires=5)
                        qml.CNOT(wires=[4, 5])
                        qml.CNOT(wires=[2, 4])

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, u3_w, dt_val, th0, th_t, p0, p_t):
            _apply_circuit(inputs, u3_w, dt_val, th0, th_t, p0, p_t)

            # Measurements:
            obs = []
            # Local <Z_i> (8)
            for i in range(n):
                obs.append(qml.expval(qml.PauliZ(i)))
            # Local <X_i> (8)
            for i in range(n):
                obs.append(qml.expval(qml.PauliX(i)))
            # Cross-modal 2-body: <Z_2 Z_3>, <Z_4 Z_5>, <Z_7 Z_0> (3)
            obs.append(qml.expval(qml.PauliZ(2) @ qml.PauliZ(3)))
            obs.append(qml.expval(qml.PauliZ(4) @ qml.PauliZ(5)))
            obs.append(qml.expval(qml.PauliZ(7) @ qml.PauliZ(0)))
            # Cross-modal 3-body: <Z_2 Z_4 Z_5> (1)
            obs.append(qml.expval(qml.PauliZ(2) @ qml.PauliZ(4) @ qml.PauliZ(5)))
            # Intra 2-body: (4)
            obs.append(qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)))
            obs.append(qml.expval(qml.PauliZ(1) @ qml.PauliZ(2)))
            obs.append(qml.expval(qml.PauliZ(5) @ qml.PauliZ(6)))
            obs.append(qml.expval(qml.PauliZ(6) @ qml.PauliZ(7)))
            # Pair correlations for 3-body cumulant: <Z_2 Z_4>, <Z_2 Z_5> (2)
            obs.append(qml.expval(qml.PauliZ(2) @ qml.PauliZ(4)))
            obs.append(qml.expval(qml.PauliZ(2) @ qml.PauliZ(5)))

            return obs

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def state_circuit(inputs, u3_w, dt_val, th0, th_t, p0, p_t):
            _apply_circuit(inputs, u3_w, dt_val, th0, th_t, p0, p_t)
            return qml.state()

        return circuit, state_circuit

    def get_state_vector(
        self,
        h_mol: torch.Tensor,
        h_struct: torch.Tensor,
        h_func: torch.Tensor,
        delta_mol: torch.Tensor,
        delta_struct: torch.Tensor,
        delta_func: torch.Tensor,
        dt_val: float
    ) -> torch.Tensor:
        """
        Extracts 256-dimensional complex state vector for an individual sample.
        """
        if self.is_classical:
            # Synthetic normalized state representation for classical baseline
            concat = torch.cat([h_mol, delta_mol, h_struct, delta_struct, h_func, delta_func], dim=-1)
            v = F.normalize(concat, p=2, dim=-1)
            return v

        in_m = torch.cat([h_mol, delta_mol], dim=-1)
        in_s = torch.cat([h_struct, delta_struct], dim=-1)
        in_f = torch.cat([h_func, delta_func], dim=-1)

        p_m = self.proj_mol(in_m.unsqueeze(0) if in_m.ndim == 1 else in_m)
        p_s = self.proj_struct(in_s.unsqueeze(0) if in_s.ndim == 1 else in_s)
        p_f = self.proj_func(in_f.unsqueeze(0) if in_f.ndim == 1 else in_f)
        q_in = torch.cat([p_m, p_s, p_f], dim=-1).squeeze(0)

        dt_t = torch.tensor(float(dt_val), dtype=torch.float32, device=h_mol.device)
        state_vec = self._state_fn(
            q_in, self.u3_weights, dt_t,
            self.theta_2body_0, self.theta_2body_t,
            self.phi_3body_0, self.phi_3body_t
        )
        return state_vec

    def forward(
        self,
        h_mol: torch.Tensor,
        h_struct: torch.Tensor,
        h_func: torch.Tensor,
        delta_mol: Optional[torch.Tensor] = None,
        delta_struct: Optional[torch.Tensor] = None,
        delta_func: Optional[torch.Tensor] = None,
        dt_years: Optional[torch.Tensor] = None,
        avail_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes time-conditioned hyperedge representation and physical observables.
        Returns:
            edge_rep: (B, 24) representation vector directly containing expectations,
                      connected 2-body correlations, and 3-body cumulants.
            metrics_dict: diagnostic observables.
        """
        is_1d = (h_mol.ndim == 1)
        if is_1d:
            h_mol = h_mol.unsqueeze(0)
            h_struct = h_struct.unsqueeze(0)
            h_func = h_func.unsqueeze(0)

        B = h_mol.shape[0]
        device = h_mol.device

        if delta_mol is None:
            delta_mol = torch.zeros_like(h_mol)
        elif is_1d and delta_mol.ndim == 1:
            delta_mol = delta_mol.unsqueeze(0)

        if delta_struct is None:
            delta_struct = torch.zeros_like(h_struct)
        elif is_1d and delta_struct.ndim == 1:
            delta_struct = delta_struct.unsqueeze(0)

        if delta_func is None:
            delta_func = torch.zeros_like(h_func)
        elif is_1d and delta_func.ndim == 1:
            delta_func = delta_func.unsqueeze(0)

        if dt_years is None:
            dt_years = torch.zeros(B, 1, device=device)
        elif is_1d and dt_years.ndim == 0:
            dt_years = dt_years.reshape(1, 1)

        if avail_mask is None:
            avail_mask = torch.ones(B, 3, device=device)
        elif is_1d and avail_mask.ndim == 1:
            avail_mask = avail_mask.unsqueeze(0)

        if self.is_classical:
            concat_in = torch.cat([
                h_mol, delta_mol,
                h_struct, delta_struct,
                h_func, delta_func,
                dt_years, avail_mask
            ], dim=-1)
            out_rep = self.classical_mlp(concat_in)
            metrics = {
                "C_mol_struct": torch.zeros(B, device=device),
                "C_struct_func": torch.zeros(B, device=device),
                "C_func_mol": torch.zeros(B, device=device),
                "Cumulant_3body": torch.zeros(B, device=device),
            }
            if is_1d:
                return out_rep.squeeze(0), {k: v.squeeze(0) for k, v in metrics.items()}
            return out_rep, metrics

        # Project current + rate of change into 8-qubit register
        in_m = torch.cat([h_mol, delta_mol], dim=-1)
        in_s = torch.cat([h_struct, delta_struct], dim=-1)
        in_f = torch.cat([h_func, delta_func], dim=-1)

        p_m = self.proj_mol(in_m)        # (B, 3)
        p_s = self.proj_struct(in_s)     # (B, 2)
        p_f = self.proj_func(in_f)       # (B, 3)

        q_in = torch.cat([p_m, p_s, p_f], dim=-1)  # (B, 8)

        # Execute QNode natively with batching
        dt_in = dt_years.squeeze(-1) if dt_years.ndim > 1 else dt_years
        raw_res = self._circuit_fn(
            q_in, self.u3_weights, dt_in,
            self.theta_2body_0, self.theta_2body_t,
            self.phi_3body_0, self.phi_3body_t
        )
        if is_1d:
            raw_obs = torch.stack(raw_res).float().unsqueeze(0)  # (1, 26)
        else:
            raw_obs = torch.stack(raw_res, dim=1).float()        # (B, 26)

        # Indices in raw_obs:
        # 0..7: <Z_0> .. <Z_7>
        # 8..15: <X_0> .. <X_7>
        # 16: <Z_2 Z_3> (Mol-Struct)
        # 17: <Z_4 Z_5> (Struct-Func)
        # 18: <Z_7 Z_0> (Func-Mol)
        # 19: <Z_2 Z_4 Z_5> (3-body)
        # 20..23: local pairs <Z_0 Z_1>, <Z_1 Z_2>, <Z_5 Z_6>, <Z_6 Z_7>
        # 24: <Z_2 Z_4>
        # 25: <Z_2 Z_5>

        z0, z2, z3, z4, z5, z7 = (
            raw_obs[:, 0], raw_obs[:, 2], raw_obs[:, 3],
            raw_obs[:, 4], raw_obs[:, 5], raw_obs[:, 7]
        )
        z2z3 = raw_obs[:, 16]
        z4z5 = raw_obs[:, 17]
        z7z0 = raw_obs[:, 18]
        z2z4z5 = raw_obs[:, 19]
        z2z4 = raw_obs[:, 24]
        z2z5 = raw_obs[:, 25]

        # 1. Connected 2-body correlations: C_ij = <Z_i Z_j> - <Z_i><Z_j>
        c_23 = z2z3 - (z2 * z3)
        c_45 = z4z5 - (z4 * z5)
        c_70 = z7z0 - (z7 * z0)

        # 2. Connected 3-body cumulant:
        # C_ijk = <Z_i Z_j Z_k> - <Z_i Z_j><Z_k> - <Z_j Z_k><Z_i> - <Z_k Z_i><Z_j> + 2 <Z_i><Z_j><Z_k>
        cumulant_3b = z2z4z5 - (z2z4 * z5) - (z4z5 * z2) - (z2z5 * z4) + 2.0 * (z2 * z4 * z5)

        # 3. Assemble 24-dimensional physical hyperedge representation:
        # [8 Pauli-Z, 8 Pauli-X, 3 Connected 2-body, 1 Connected 3-body cumulant, 4 Local 2-body]
        edge_rep = torch.cat([
            raw_obs[:, 0:16],                  # 16 local expectations
            c_23.unsqueeze(1),                 # Connected Mol-Struct
            c_45.unsqueeze(1),                 # Connected Struct-Func
            c_70.unsqueeze(1),                 # Connected Func-Mol
            cumulant_3b.unsqueeze(1),          # Connected 3-body Cumulant
            raw_obs[:, 20:24]                  # 4 local correlators
        ], dim=-1)  # (B, 24)

        metrics = {
            "C_mol_struct": c_23,
            "C_struct_func": c_45,
            "C_func_mol": c_70,
            "Cumulant_3body": cumulant_3b,
            "mean_connected_2body": (c_23.abs() + c_45.abs() + c_70.abs()) / 3.0
        }

        if is_1d:
            return edge_rep.squeeze(0), {k: v.squeeze(0) for k, v in metrics.items()}
        return edge_rep, metrics

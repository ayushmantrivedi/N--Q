"""
NĀḌĪ-Q Phase 10A: Quantum Hyperedge Encoder.
Implements Cross-Node Quantum Entanglement and Physical Readout for Multimodal Hyperedges.
Reuses PennyLane-PyTorch infrastructure from QVBE-v3.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

from nadiq.models.qvbe_v3.config import QVBEv3Config


class QuantumHyperedgeEncoder(nn.Module):
    """
    Encodes cross-modal hyperedge nodes (Molecular, Structural, Functional) into an 8-qubit register,
    executing cross-node entangling unitaries and reading out physical observables & connected correlators.
    """

    def __init__(
        self,
        dim_mol: int = 15,
        dim_struct: int = 36,
        dim_func: int = 35,
        n_qubits: int = 8,
        n_layers: int = 2,
        entangle: bool = True,
        is_classical: bool = False,
        seed: int = 42
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangle = entangle
        self.is_classical = is_classical

        # Modality qubit allocation:
        # Qubits 0, 1, 2: Molecular (3 qubits)
        # Qubits 3, 4: Structural MRI (2 qubits)
        # Qubits 5, 6, 7: Functional Motor (3 qubits)
        self.q_mol = [0, 1, 2]
        self.q_struct = [3, 4]
        self.q_func = [5, 6, 7]

        # Trainable input projection adapters per modality
        self.proj_mol = nn.Linear(dim_mol, len(self.q_mol))
        self.proj_struct = nn.Linear(dim_struct, len(self.q_struct))
        self.proj_func = nn.Linear(dim_func, len(self.q_func))

        if is_classical:
            # Classical MLP hyperedge encoder baseline
            total_in = dim_mol + dim_struct + dim_func
            self.classical_mlp = nn.Sequential(
                nn.Linear(total_in, 32),
                nn.GELU(),
                nn.Linear(32, 23)
            )
            self._circuit_fn = None
            self.u3_weights = None
        else:
            self.classical_mlp = None
            torch.manual_seed(seed)
            # Parameterized rotation angles per layer per qubit
            self.u3_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.05)
            self.dev = qml.device("default.qubit", wires=n_qubits)
            self._circuit_fn = self._build_qnode()

    def _build_qnode(self):
        n = self.n_qubits
        n_layers = self.n_layers
        entangle = self.entangle

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, u3_w):
            # inputs shape: (B, 8)
            # 1. Superposition initialization
            for i in range(n):
                qml.Hadamard(wires=i)

            # 2. Angle encoding of partitioned modality representations
            for i in range(n):
                angle = torch.sigmoid(inputs[:, i]) * math.pi
                qml.RY(angle, wires=i)
                qml.RZ(angle * 0.5, wires=i)

            # 3. Variational and Entangling Layers
            for l in range(n_layers):
                for i in range(n):
                    qml.Rot(u3_w[l, i, 0], u3_w[l, i, 1], u3_w[l, i, 2], wires=i)

                if entangle:
                    # Intra-node local entanglement
                    qml.CNOT(wires=[0, 1])
                    qml.CNOT(wires=[1, 2])
                    qml.CNOT(wires=[3, 4])
                    qml.CNOT(wires=[5, 6])
                    qml.CNOT(wires=[6, 7])

                    # CROSS-NODE ENTANGLEMENT:
                    # Bridge Mol -> Struct
                    qml.CNOT(wires=[2, 3])
                    # Bridge Struct -> Func
                    qml.CNOT(wires=[4, 5])
                    # Bridge Func -> Mol (Cyclic)
                    qml.CNOT(wires=[7, 0])

            # 4. Multi-basis expectation readouts
            obs = []
            # Single-qubit Pauli-Z expectations: <Z_i> (8 observables)
            for i in range(n):
                obs.append(qml.expval(qml.PauliZ(i)))

            # Single-qubit Pauli-X expectations: <X_i> (8 observables)
            for i in range(n):
                obs.append(qml.expval(qml.PauliX(i)))

            # Cross-node 2-qubit correlators:
            # Mol - Struct: <Z_2 Z_3>
            obs.append(qml.expval(qml.PauliZ(2) @ qml.PauliZ(3)))
            # Struct - Func: <Z_4 Z_5>
            obs.append(qml.expval(qml.PauliZ(4) @ qml.PauliZ(5)))
            # Func - Mol: <Z_7 Z_0>
            obs.append(qml.expval(qml.PauliZ(7) @ qml.PauliZ(0)))

            # Local adjacent correlators: <Z_0 Z_1>, <Z_1 Z_2>, <Z_5 Z_6>, <Z_6 Z_7> (4 observables)
            obs.append(qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)))
            obs.append(qml.expval(qml.PauliZ(1) @ qml.PauliZ(2)))
            obs.append(qml.expval(qml.PauliZ(5) @ qml.PauliZ(6)))
            obs.append(qml.expval(qml.PauliZ(6) @ qml.PauliZ(7)))

            return obs

        return circuit

    def forward(
        self,
        h_mol: torch.Tensor,
        h_struct: torch.Tensor,
        h_func: torch.Tensor,
        avail_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass computing hyperedge embedding and connected correlators.
        h_mol: (B, dim_mol)
        h_struct: (B, dim_struct)
        h_func: (B, dim_func)
        returns: (hyperedge_embedding: (B, 23), metrics_dict)
        """
        is_1d = (h_mol.ndim == 1)
        if is_1d:
            h_mol = h_mol.unsqueeze(0)
            h_struct = h_struct.unsqueeze(0)
            h_func = h_func.unsqueeze(0)

        batch_size = h_mol.shape[0]

        # Classical baseline branch
        if self.is_classical:
            concat_in = torch.cat([h_mol, h_struct, h_func], dim=-1)
            out = self.classical_mlp(concat_in)
            metrics = {
                "C_mol_struct": torch.zeros(batch_size, device=h_mol.device),
                "C_struct_func": torch.zeros(batch_size, device=h_mol.device),
                "C_func_mol": torch.zeros(batch_size, device=h_mol.device),
            }
            if is_1d:
                return out.squeeze(0), {k: v.squeeze(0) for k, v in metrics.items()}
            return out, metrics

        # Quantum branch: Project modality features into register
        p_m = self.proj_mol(h_mol)        # (B, 3)
        p_s = self.proj_struct(h_struct)  # (B, 2)
        p_f = self.proj_func(h_func)      # (B, 3)

        # Assemble partitioned 8-qubit input vector
        q_in = torch.cat([p_m, p_s, p_f], dim=-1)  # (B, 8)

        # Execute QNode
        res = self._circuit_fn(q_in, self.u3_weights)
        raw_obs = torch.stack(res, dim=1).float()  # (B, 23)

        # Extract connected cross-node correlators: C_ij = <Z_i Z_j> - <Z_i><Z_j>
        # Indices:
        # 0..7: <Z_0> .. <Z_7>
        # 8..15: <X_0> .. <X_7>
        # 16: <Z_2 Z_3> (Mol-Struct)
        # 17: <Z_4 Z_5> (Struct-Func)
        # 18: <Z_7 Z_0> (Func-Mol)
        z2 = raw_obs[:, 2]
        z3 = raw_obs[:, 3]
        z4 = raw_obs[:, 4]
        z5 = raw_obs[:, 5]
        z7 = raw_obs[:, 7]
        z0 = raw_obs[:, 0]

        c_mol_struct = raw_obs[:, 16] - (z2 * z3)
        c_struct_func = raw_obs[:, 17] - (z4 * z5)
        c_func_mol = raw_obs[:, 18] - (z7 * z0)

        metrics = {
            "C_mol_struct": c_mol_struct,
            "C_struct_func": c_struct_func,
            "C_func_mol": c_func_mol,
            "mean_connected_cross_node": (c_mol_struct.abs() + c_struct_func.abs() + c_func_mol.abs()) / 3.0
        }

        if is_1d:
            return raw_obs.squeeze(0), {k: v.squeeze(0) for k, v in metrics.items()}
        return raw_obs, metrics

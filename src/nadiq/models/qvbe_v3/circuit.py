"""
NĀḌĪ-Q QVBE-v3 Quantum Circuit Architecture.
Implements Data Re-uploading, Differentiable PyTorch-PennyLane QNode, and Configurable Entanglement.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from nadiq.models.qvbe_v3.config import QVBEv3Config


class QVBEv3Circuit(nn.Module):
    """
    Differentiable Quantum Variational Biomarker Encoder v3.
    Features:
    - Native PyTorch autograd integration via PennyLane backprop.
    - Data Re-uploading across multiple layers to extend the Fourier spectrum.
    - Trainable linear adapter to eliminate classical PCA bottleneck.
    - Separable vs. Entangled modes for rigorous ablation.
    - Multi-basis expectation readouts with connected correlators.
    """

    def __init__(self, config: Optional[QVBEv3Config] = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = QVBEv3Config(**kwargs)
        self.config = config

        torch.manual_seed(config.seed)

        # 1. Trainable Input Adapter (eliminates fixed classical PCA bottleneck)
        if config.use_trainable_adapter:
            # Shape (n_qubits, input_dim)
            scale = 1.0 / math.sqrt(max(1, config.input_dim))
            self.adapter = nn.Parameter(torch.randn(config.n_qubits, config.input_dim) * scale)
        else:
            self.adapter = None

        # 2. Variational Quantum Parameters
        # Controlled near-identity initialization to prevent random seed dominance
        self.u3_weights = nn.Parameter(torch.randn(config.n_layers, config.n_qubits, 3) * 0.05)
        self.upload_scales = nn.Parameter(torch.ones(config.n_uploads, config.n_qubits))

        # 3. Construct PennyLane device and QNode
        self.dev = qml.device(config.backend, wires=config.n_qubits, shots=config.shots)
        self._circuit_fn = self._build_qnode()

    def _build_qnode(self):
        n = self.config.n_qubits
        n_layers = self.config.n_layers
        n_uploads = self.config.n_uploads
        entangle = self.config.entangle
        basis = self.config.measurement_basis

        @qml.qnode(self.dev, interface="torch", diff_method="backprop" if self.config.shots is None else "parameter-shift")
        def circuit(inputs, u3_w, up_scales):
            # inputs shape: (B, n)
            # 1. Superposition initialization
            for i in range(n):
                qml.Hadamard(wires=i)

            # 2. Sequential Data Re-uploading & Variational Layers
            for l in range(n_layers):
                u = l % n_uploads
                # Data re-uploading stage: encode features modulated by upload_scales
                for i in range(n):
                    angle = torch.sigmoid(inputs[:, i] * up_scales[u, i]) * math.pi
                    qml.RY(angle, wires=i)
                    qml.RZ(angle * 0.5, wires=i)

                # Parameterized rotation layer
                for i in range(n):
                    qml.Rot(u3_w[l, i, 0], u3_w[l, i, 1], u3_w[l, i, 2], wires=i)

                # Entanglement layer (cyclic CNOTs)
                if entangle:
                    for i in range(n):
                        qml.CNOT(wires=[i, (i + 1) % n])

            # 3. Multi-basis expectation readouts
            obs = []
            # Pauli-Z expectations: <Z_i>
            for i in range(n):
                obs.append(qml.expval(qml.PauliZ(i)))

            # Pauli-X expectations: <X_i>
            if basis in ["dual_basis", "full", "full_plus_connected"]:
                for i in range(n):
                    obs.append(qml.expval(qml.PauliX(i)))

            # Adjacent 2-qubit correlators: <Z_i Z_{i+1}>
            if basis in ["full", "full_plus_connected"]:
                for i in range(n - 1):
                    obs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)))

            # Non-adjacent correlators for full_plus_connected
            if basis == "full_plus_connected":
                for i in range(n - 1):
                    target = (i + 2) % n
                    obs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(target)))

            return obs

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass mapping input features to physical quantum observables.
        x: (B, d) or (d,)
        returns: (B, output_dim)
        """
        is_1d = (x.ndim == 1)
        if is_1d:
            x = x.unsqueeze(0)

        # Apply trainable linear adapter if enabled
        if self.config.use_trainable_adapter and self.adapter is not None:
            x_proj = torch.matmul(x, self.adapter.t())
        elif x.shape[1] == self.config.n_qubits:
            x_proj = x
        else:
            # If d != n_qubits and no adapter specified, slice or pad smoothly
            if x.shape[1] > self.config.n_qubits:
                x_proj = x[:, :self.config.n_qubits]
            else:
                pad = torch.zeros(x.shape[0], self.config.n_qubits - x.shape[1], device=x.device, dtype=x.dtype)
                x_proj = torch.cat([x, pad], dim=1)

        # Evaluate quantum node
        res = self._circuit_fn(x_proj, self.u3_weights, self.upload_scales)
        out = torch.stack(res, dim=1).float()

        if is_1d:
            return out.squeeze(0)
        return out

"""
Quantum Variational Biomarker Encoder (QVBE) — Nāḍī-Q v1.

QVBE is the first core intelligence component of the Nāḍī-Q engine.
It encodes a generic feature tensor (from any modality, any dataset) into
a quantum embedding via a trainable hybrid classical-quantum architecture.

Architecture (per sample):
  FeatureBlock → [Linear projection] → ℝ^n_qubits
               → [Angle encoding]   → |ψ⟩ ∈ ℂ^{2^n_qubits}
               → [VQC: L layers]    → parameterised |ψ(θ)⟩
               → [Pauli-Z measure]  → ℝ^n_qubits  (embedding)

Design principles:
  - Dataset-agnostic: accepts any (batch, n_features) tensor.
  - No dementia-specific or TADPOLE-specific logic.
  - Differentiable end-to-end via PennyLane + PyTorch autograd.
  - Deterministic parameter initialization via seeded RNG.
  - Batched via vmap-style loop (PennyLane default.qubit).
  - Clean PyTorch nn.Module interface.

Parameter count:
  Classical projection : n_features × n_qubits + n_qubits (bias)
  Quantum VQC         : n_qubits × 3 × n_layers  (Ry/Rz per qubit per layer)
  Total               : n_features × n_qubits + n_qubits + 3 × n_qubits × n_layers

References:
  - Havlíček et al. (2019) — Supervised learning with quantum-enhanced feature spaces
  - Schuld et al. (2020) — Circuit-centric quantum classifiers
  - PennyLane docs — qml.qnn.TorchLayer
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

__all__ = [
    "QVBEConfig",
    "QuantumVariationalBiomarkerEncoder",
    "QVBEOutput",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QVBEConfig:
    """
    Full configuration for a QVBE instance.

    Attributes:
        input_dim     : Dimensionality of the incoming feature vector.
        n_qubits      : Number of qubits used in the quantum circuit.
        n_layers      : Number of variational ansatz repetitions (circuit depth).
        latent_dim    : Intermediate classical projection dim.
                        If None, defaults to n_qubits.
        backend       : PennyLane device string (e.g. "default.qubit").
        seed          : Random seed for deterministic parameter initialization.
        shots         : Number of shots. None → exact statevector (differentiable).
        diff_method   : PennyLane differentiation method.
    """

    input_dim: int
    n_qubits: int = 4
    n_layers: int = 2
    latent_dim: Optional[int] = None  # defaults to n_qubits
    backend: str = "default.qubit"
    seed: int = 42
    shots: Optional[int] = None
    diff_method: str = "best"

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be ≥ 1, got {self.input_dim}")
        if self.n_qubits < 1:
            raise ValueError(f"n_qubits must be ≥ 1, got {self.n_qubits}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1, got {self.n_layers}")
        if self.latent_dim is None:
            self.latent_dim = self.n_qubits
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be ≥ 1, got {self.latent_dim}")

    @property
    def n_vqc_params(self) -> int:
        """Number of trainable variational circuit parameters."""
        # 3 rotation parameters (Ry, Rz, Ry) per qubit per layer
        return 3 * self.n_qubits * self.n_layers

    @property
    def n_projection_params(self) -> int:
        """Number of trainable classical projection parameters (weights + bias)."""
        return self.input_dim * (self.latent_dim or self.n_qubits) + (self.latent_dim or self.n_qubits)

    @property
    def n_total_params(self) -> int:
        return self.n_projection_params + self.n_vqc_params


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class QVBEOutput:
    """
    Output of a QVBE forward pass.

    Attributes:
        embedding     : Quantum embedding tensor, shape (batch, n_qubits).
        pre_quantum   : Classical projection output before encoding, shape (batch, n_qubits).
        vqc_params    : Snapshot of the VQC parameters at inference time.
    """

    embedding: torch.Tensor           # (batch, n_qubits)
    pre_quantum: torch.Tensor         # (batch, latent_dim → n_qubits via angle encoding)
    vqc_params: torch.Tensor          # (n_vqc_params,)


# ---------------------------------------------------------------------------
# Variational Quantum Circuit
# ---------------------------------------------------------------------------


def _build_vqc_circuit(n_qubits: int, n_layers: int) -> Any:
    """
    Build the parameterised quantum circuit as a PennyLane QNode.

    Circuit ansatz (per layer):
        1. Per-qubit Ry(θ) rotation from input angles (data re-uploading style).
        2. Per-qubit Ry(φ₁), Rz(φ₂), Ry(φ₃) trainable rotations.
        3. Linear CNOT entanglement: qubit i → qubit (i+1) % n_qubits.

    After all layers:
        Measure ⟨Z⟩ on each qubit → returns expectation vector ∈ ℝ^n_qubits.

    Args:
        n_qubits : Number of qubits.
        n_layers : Number of ansatz repetitions.

    Returns:
        A PennyLane QNode callable.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[Any]:
        """
        Args:
            inputs  : Angle-encoded feature vector, shape (n_qubits,).
            weights : Variational parameters, shape (n_layers, n_qubits, 3).
        """
        # Initial angle encoding (data re-uploading)
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        # Variational layers
        for layer in range(n_layers):
            # Trainable rotations
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
                qml.RY(weights[layer, q, 2], wires=q)
            # Ring entanglement
            if n_qubits > 1:
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])

        # Measure Pauli-Z expectation on every qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


# ---------------------------------------------------------------------------
# QVBE Module
# ---------------------------------------------------------------------------


class QuantumVariationalBiomarkerEncoder(nn.Module):
    """
    Quantum Variational Biomarker Encoder (QVBE).

    Hybrid classical-quantum encoder. Accepts any real-valued feature vector
    of fixed dimension and produces a fixed-size quantum embedding.

    This module is dataset-agnostic: it receives a generic feature tensor,
    not TADPOLE columns or dementia-specific features.

    Forward pass (batched):
        x ∈ ℝ^{B × input_dim}
        → projection: ℝ^{B × n_qubits}   (trainable linear layer)
        → angle encoding: [0, 2π]^{B × n_qubits}  (tanh + π scaling)
        → VQC: ⟨Z⟩ per qubit → ℝ^{B × n_qubits}
        → QVBEOutput.embedding ∈ [-1, 1]^{B × n_qubits}

    Usage:
        config = QVBEConfig(input_dim=13, n_qubits=4, n_layers=2)
        qvbe = QuantumVariationalBiomarkerEncoder(config)
        out = qvbe(x)          # x: torch.Tensor (batch, 13)
        embedding = out.embedding  # torch.Tensor (batch, 4)
    """

    def __init__(self, config: QVBEConfig) -> None:
        super().__init__()
        self.config = config

        # Seed deterministic parameter init
        torch.manual_seed(config.seed)
        rng = np.random.default_rng(config.seed)

        # ── Classical projection: input_dim → n_qubits ────────────────────
        assert config.latent_dim is not None
        self.projection = nn.Linear(config.input_dim, config.n_qubits)

        # ── Variational quantum circuit ────────────────────────────────────
        circuit_fn = _build_vqc_circuit(config.n_qubits, config.n_layers)

        # VQC weight shape: (n_layers, n_qubits, 3)
        weight_shapes = {"weights": (config.n_layers, config.n_qubits, 3)}
        self.vqc = qml.qnn.TorchLayer(circuit_fn, weight_shapes)

        # Re-seed after TorchLayer potentially shuffles rng state
        with torch.no_grad():
            init_weights = torch.tensor(
                rng.uniform(-math.pi, math.pi,
                            size=(config.n_layers, config.n_qubits, 3)),
                dtype=torch.float32,
            )
            self.vqc.weights.copy_(init_weights)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def embedding_dim(self) -> int:
        return self.config.n_qubits

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> QVBEOutput:
        """
        Encode a batch of feature vectors into quantum embeddings.

        Args:
            x: Feature tensor of shape (batch_size, input_dim).
               May contain NaN values — they are zeroed before encoding.
               The caller is responsible for proper imputation if needed.

        Returns:
            QVBEOutput with:
                .embedding   : (batch_size, n_qubits) ∈ [-1, 1]
                .pre_quantum : (batch_size, n_qubits) after projection + scaling
                .vqc_params  : current VQC parameter snapshot
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Replace NaN with zero (caller should impute; this is a safety guard)
        x = torch.nan_to_num(x, nan=0.0)

        # 1. Classical projection: (B, input_dim) → (B, n_qubits)
        projected = self.projection(x)  # (B, n_qubits)

        # 2. Scale to angle range [−π, π] via tanh
        angles = torch.tanh(projected) * math.pi  # (B, n_qubits) ∈ (−π, π)

        # 3. Per-sample VQC execution (batched loop)
        embeddings = []
        for i in range(angles.shape[0]):
            out = self.vqc(angles[i])  # returns (n_qubits,) tensor
            if isinstance(out, (list, tuple)):
                out = torch.stack(out)
            embeddings.append(out)

        embedding = torch.stack(embeddings, dim=0)  # (B, n_qubits)

        return QVBEOutput(
            embedding=embedding,
            pre_quantum=angles,
            vqc_params=self.vqc.weights.detach().clone(),
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        """Return serialisable configuration dictionary."""
        cfg = self.config
        return {
            "input_dim": cfg.input_dim,
            "n_qubits": cfg.n_qubits,
            "n_layers": cfg.n_layers,
            "latent_dim": cfg.latent_dim,
            "backend": cfg.backend,
            "seed": cfg.seed,
            "shots": cfg.shots,
            "diff_method": cfg.diff_method,
            "n_vqc_params": cfg.n_vqc_params,
            "n_projection_params": cfg.n_projection_params,
            "n_total_params_config": cfg.n_total_params,
            "n_total_params_actual": self.n_parameters,
        }

    def __repr__(self) -> str:
        c = self.config
        return (
            f"QuantumVariationalBiomarkerEncoder("
            f"input_dim={c.input_dim}, n_qubits={c.n_qubits}, "
            f"n_layers={c.n_layers}, params={self.n_parameters})"
        )

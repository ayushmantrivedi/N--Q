"""
NĀḌĪ-Q QVBE-v3 Trainer.
Executes True Variational Cross-Modal Alignment via Parameter-Shift / Autograd Gradients.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim

from nadiq.models.qvbe_v3.config import QVBEv3Config
from nadiq.models.qvbe_v3.circuit import QVBEv3Circuit
from nadiq.models.qvbe_v3.objectives import infonce_loss


class QVBEv3Trainer:
    """
    Supervised/Self-Supervised Trainer for QVBE-v3.
    Replaces the broken Mode B loop with exact cross-modal InfoNCE alignment.
    """
    def __init__(self, config: QVBEv3Config, lr: Optional[float] = None) -> None:
        self.config = config
        self.lr = lr if lr is not None else config.learning_rate
        self.circuit = QVBEv3Circuit(config)

        # Optimizer over circuit parameters
        self.optimizer = optim.Adam(
            self.circuit.parameters(),
            lr=self.lr,
            weight_decay=config.weight_decay
        )

        self.tmp_proj = None

    def step(self, x_mol: torch.Tensor, x_tmp: torch.Tensor) -> float:
        """
        Single optimization step over a paired multimodal batch.
        x_mol: (B, d_mol)
        x_tmp: (B, d_tmp)
        returns: float loss value
        """
        self.circuit.train()
        self.optimizer.zero_grad()

        # Compute molecular quantum embedding
        q_mol = self.circuit(x_mol).float()

        # Compute or project temporal embedding to match representation space
        if x_tmp.shape[1] == x_mol.shape[1] and not self.config.use_trainable_adapter:
            q_tmp = self.circuit(x_tmp).float()
        else:
            if self.tmp_proj is None:
                self.tmp_proj = nn.Linear(x_tmp.shape[1], q_mol.shape[1]).to(x_tmp.device)
                self.optimizer.add_param_group({'params': self.tmp_proj.parameters()})
            q_tmp = self.tmp_proj(x_tmp).float()

        # Symmetric InfoNCE loss
        loss = infonce_loss(q_mol, q_tmp, temperature=self.config.temperature)
        loss.backward()

        self.optimizer.step()
        return float(loss.item())

    def fit(self, x_mol: torch.Tensor, x_tmp: torch.Tensor, epochs: int = 20) -> list[float]:
        """
        Train over multiple epochs on paired representations.
        """
        losses = []
        for ep in range(epochs):
            l = self.step(x_mol, x_tmp)
            losses.append(l)
        return losses

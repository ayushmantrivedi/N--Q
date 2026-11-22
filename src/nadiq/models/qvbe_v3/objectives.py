"""
NĀḌĪ-Q QVBE-v3 Learning Objectives.
Implements Cross-Modal InfoNCE Alignment and Masked Feature Reconstruction without target leakage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def infonce_loss(q_a: torch.Tensor, q_b: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Symmetric InfoNCE Cross-Modal Contrastive Loss.
    Encourages paired representations (e.g. Molecular q_mol[i] and Temporal h_temp[i] for the same patient)
    to have high mutual alignment, while treating all other in-batch participants as negative pairs.
    
    q_a: (B, D) representation from modality A
    q_b: (B, D) representation from modality B
    temperature: scalar scaling factor tau
    """
    if q_a.shape[0] != q_b.shape[0]:
        raise ValueError(f"Batch size mismatch: {q_a.shape[0]} vs {q_b.shape[0]}")
    
    # 1. Cast to float and L2 normalization along feature dimension
    q_a = q_a.float()
    q_b = q_b.float()
    q_a_norm = F.normalize(q_a, p=2, dim=-1)
    q_b_norm = F.normalize(q_b, p=2, dim=-1)

    # 2. Pairwise cosine similarity matrix (B, B)
    sim_matrix = torch.matmul(q_a_norm, q_b_norm.t()) / max(1e-6, temperature)

    # 3. Targets are diagonal identity indices (i, i)
    batch_size = q_a.shape[0]
    labels = torch.arange(batch_size, device=q_a.device)

    # 4. Bidirectional InfoNCE loss
    loss_a2b = F.cross_entropy(sim_matrix, labels)
    loss_b2a = F.cross_entropy(sim_matrix.t(), labels)

    return (loss_a2b + loss_b2a) / 2.0


def masked_reconstruction_loss(x_original: torch.Tensor, x_reconstructed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error loss evaluated exclusively on masked input features.
    """
    diff = (x_original - x_reconstructed) * mask
    num_masked = torch.clamp(mask.sum(), min=1.0)
    return (diff ** 2).sum() / num_masked

"""
NĀḌĪ-Q QVBE-v4 Trainer and Multi-Objective Optimization Engine.
Implements the 4-part relational loss formulation, gradient clipping,
representation diagnostics, and effective rank computation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nadiq.models.qvbe_v4.config import QVBEv4Config
from nadiq.models.qvbe_v4.encoder import QVBEv4Encoder
from nadiq.models.qvbe_v4.output import QVBEv4Output


class QVBEv4Trainer:
    """
    Trainer and evaluation harness for QVBE-v4.
    Optimizes the multi-objective loss:
        L_total = lambda_edge * L_edge_rec
                + lambda_temp * L_temp_pred
                + lambda_mask * L_masked_node
                + lambda_task * L_downstream
    """

    def __init__(
        self,
        model: QVBEv4Encoder,
        config: Optional[QVBEv4Config] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.config = config or model.config
        self.device = next(model.parameters()).device

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def compute_loss(
        self,
        output: QVBEv4Output,
        targets: Dict[str, Any],
        node_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the weighted multi-objective loss.
        targets may contain:
        - "y": (B,) or (B, 1) binary downstream progression label
        - "edge_weights": (B, n_pairs) ground-truth edge weights to reconstruct
        - "next_delta": (B, n_nodes) ground-truth next-visit state change
        - "node_raw": Dict[str, torch.Tensor] raw unmasked node features
        """
        device = output.motif_embedding.device
        loss_dict: Dict[str, float] = {}

        # 1. Downstream Classification Task Loss
        loss_task = torch.tensor(0.0, device=device)
        if "y" in targets and targets["y"] is not None and output.downstream_logits is not None:
            y_true = targets["y"].to(device).float()
            if y_true.ndim == 1:
                y_true = y_true.unsqueeze(-1)
            logits = output.downstream_logits
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)
            loss_task = self.bce_loss(logits, y_true)
        loss_dict["loss_task"] = float(loss_task.item())

        # 2. Relational Edge Reconstruction Loss
        loss_edge = torch.tensor(0.0, device=device)
        if "edge_weights" in targets and targets["edge_weights"] is not None and output.reconstructed_edge_weights is not None:
            w_true = targets["edge_weights"].to(device).float()
            loss_edge = self.mse_loss(output.reconstructed_edge_weights, w_true)
        loss_dict["loss_edge"] = float(loss_edge.item())

        # 3. Next-Visit Temporal Change Prediction Loss
        loss_temporal = torch.tensor(0.0, device=device)
        if "next_delta" in targets and targets["next_delta"] is not None and output.predicted_next_delta is not None:
            delta_true = targets["next_delta"].to(device).float()
            loss_temporal = self.mse_loss(output.predicted_next_delta, delta_true)
        loss_dict["loss_temporal"] = float(loss_temporal.item())

        # 4. Masked Node Feature Reconstruction Loss
        loss_masked = torch.tensor(0.0, device=device)
        if "node_raw" in targets and targets["node_raw"] is not None and output.reconstructed_node_features is not None:
            recon_losses = []
            node_raw = targets["node_raw"]
            for i, name in enumerate(self.config.node_names):
                if name in node_raw and name in output.reconstructed_node_features:
                    x_target = node_raw[name].to(device).float()
                    x_pred = output.reconstructed_node_features[name]
                    # If mask is provided, evaluate reconstruction loss on observed features
                    if node_masks is not None:
                        m_i = node_masks[:, i : i + 1].to(device).float()
                        diff = (x_pred - x_target) ** 2
                        recon_losses.append((diff * m_i).sum() / (m_i.sum() * diff.shape[-1] + 1e-6))
                    else:
                        recon_losses.append(self.mse_loss(x_pred, x_target))
            if recon_losses:
                loss_masked = torch.stack(recon_losses).mean()
        loss_dict["loss_masked"] = float(loss_masked.item())

        # Combine with configured loss weights
        total_loss = (
            self.config.lambda_task * loss_task
            + self.config.lambda_edge * loss_edge
            + self.config.lambda_temporal * loss_temporal
            + self.config.lambda_masked * loss_masked
        )
        loss_dict["loss_total"] = float(total_loss.item())

        return total_loss, loss_dict

    def train_step(
        self,
        node_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        graph_edges: List[Dict[str, Any]],
        targets: Dict[str, Any],
        node_masks: Optional[torch.Tensor] = None,
        elapsed_times: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, float], float]:
        """
        Executes a single optimization step with gradient clipping and returns loss metrics.
        """
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(
            node_inputs=node_inputs,
            graph_edges=graph_edges,
            node_masks=node_masks,
            elapsed_times=elapsed_times,
            return_diagnostics=False,
        )

        # If targets do not contain edge_weights, compile default from graph_edges
        if "edge_weights" not in targets:
            compiled = self.model.compiler.compile_graph_motif(
                edges=graph_edges,
                node_masks=node_masks,
                elapsed_times=elapsed_times,
                batch_size=node_inputs[self.config.node_names[0]][0].shape[0],
            )
            targets = dict(targets)
            targets["edge_weights"] = compiled["edge_weights"]

        total_loss, loss_dict = self.compute_loss(output, targets, node_masks=node_masks)

        total_loss.backward()

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.max_grad_norm,
            ).item()
        )

        self.optimizer.step()

        return loss_dict, grad_norm

    @torch.no_grad()
    def evaluate(
        self,
        node_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        graph_edges: List[Dict[str, Any]],
        targets: Dict[str, Any],
        node_masks: Optional[torch.Tensor] = None,
        elapsed_times: Optional[torch.Tensor] = None,
    ) -> Tuple[QVBEv4Output, Dict[str, float]]:
        """
        Runs evaluation in no_grad mode and returns output and loss metrics.
        """
        self.model.eval()
        output = self.model(
            node_inputs=node_inputs,
            graph_edges=graph_edges,
            node_masks=node_masks,
            elapsed_times=elapsed_times,
            return_diagnostics=True,
        )

        if "edge_weights" not in targets:
            compiled = self.model.compiler.compile_graph_motif(
                edges=graph_edges,
                node_masks=node_masks,
                elapsed_times=elapsed_times,
                batch_size=node_inputs[self.config.node_names[0]][0].shape[0],
            )
            targets = dict(targets)
            targets["edge_weights"] = compiled["edge_weights"]

        _, loss_dict = self.compute_loss(output, targets, node_masks=node_masks)
        return output, loss_dict

    @staticmethod
    def compute_effective_rank(embeddings: torch.Tensor) -> float:
        """
        Computes the effective representation rank via SVD:
            R_eff = exp(-sum_i p_i * ln(p_i))
        where p_i = sigma_i / sum_j sigma_j.
        Measures the effective dimensionality and prevents representation collapse.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings tensor, got shape {embeddings.shape}")

        # Center embeddings
        z = embeddings - embeddings.mean(dim=0, keepdim=True)
        # SVD on centered embeddings
        try:
            _, s, _ = torch.linalg.svd(z, full_matrices=False)
            s_sum = s.sum()
            if s_sum <= 1e-12:
                return 1.0
            p = s / s_sum
            # Non-zero probabilities
            p_pos = p[p > 1e-12]
            entropy = -(p_pos * torch.log(p_pos)).sum().item()
            return float(np.exp(entropy))
        except Exception:
            return 1.0

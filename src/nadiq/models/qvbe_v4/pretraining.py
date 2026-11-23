"""
NĀḌĪ-Q Phase 12.2: Quantum Relational Self-Supervised Pretraining.

Implements:
1. Masked Relational Reconstruction (edge weights, lags, uncertainties, directions).
2. Temporal Relational Reconstruction (predicting observed BL-to-Month-12 delta transitions).
3. Masked Node-Feature Reconstruction (predicting baseline node representations).
4. Matched Classical Relational Pretraining Model (identical objectives, inputs, and budget).
5. Fine-tuning strategy managers (frozen, partial fine-tuning, full fine-tuning).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nadiq.models.qvbe_v4.config import QVBEv4Config
from nadiq.models.qvbe_v4.encoder import QVBEv4Encoder
from nadiq.models.integrated.model import ClassicalRelationalBranch


@dataclass
class RelationalPretrainingConfig:
    """Configuration for self-supervised relational pretraining."""
    weight_edge_recon: float = 1.0        # lambda 1
    weight_temporal_recon: float = 1.0    # lambda 2
    weight_node_recon: float = 0.5        # lambda 3
    mask_edge_prob: float = 0.33          # probability of masking an edge (at least 1 masked per batch)
    learning_rate: float = 0.015
    weight_decay: float = 1e-4
    n_epochs: int = 25
    hidden_dim: int = 32
    freeze_strategy: str = "full"         # "frozen", "partial", "full"


class RelationalReconstructionHead(nn.Module):
    """
    Multitask reconstruction head mapping relational motif embeddings (16 dims)
    to missing graph edge attributes, temporal delta transitions, and node states.
    """
    def __init__(self, embed_dim: int = 16, n_nodes: int = 3, n_pairs: int = 3):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_pairs = n_pairs

        # 1. Edge Attribute Prediction Heads
        self.head_weight = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_pairs),
            nn.Tanh(),  # weights in [-1, 1]
        )
        self.head_lag = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_pairs),
            nn.ReLU(),  # elapsed lag >= 0
        )
        self.head_se = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_pairs),
            nn.Sigmoid(),  # SE in [0, 1]
        )
        self.head_direction = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_pairs),  # logits for direction sign
        )

        # 2. Temporal Transition Prediction Head (observed 1-year delta for each node)
        self.head_temporal_delta = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_nodes),
        )

        # 3. Node State Prediction Head (baseline state for each node)
        self.head_node_state = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_nodes),
        )

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "pred_weight": self.head_weight(embedding),
            "pred_lag": self.head_lag(embedding),
            "pred_se": self.head_se(embedding),
            "pred_direction": self.head_direction(embedding),
            "pred_temporal_delta": self.head_temporal_delta(embedding),
            "pred_node_state": self.head_node_state(embedding),
        }


class QVBEv4PretrainingModel(nn.Module):
    """
    Self-Supervised Relational Pretraining Wrapper for QVBE-v4.
    """
    def __init__(self, qvbe_config: Optional[QVBEv4Config] = None, pretrain_config: Optional[RelationalPretrainingConfig] = None):
        super().__init__()
        if qvbe_config is None:
            self.qvbe_config = QVBEv4Config(
                n_nodes=3,
                qubits_per_node=2,
                node_names=["molecular", "structural", "functional"],
                node_input_dims={"molecular": 1, "structural": 1, "functional": 1},
                motif_embed_dim=16,
            )
        else:
            self.qvbe_config = qvbe_config
        self.pretrain_config = pretrain_config or RelationalPretrainingConfig()

        self.encoder = QVBEv4Encoder(self.qvbe_config)
        self.head = RelationalReconstructionHead(
            embed_dim=self.qvbe_config.motif_embed_dim,
            n_nodes=self.qvbe_config.n_nodes,
            n_pairs=3,
        )

    def forward_pretrain(
        self,
        node_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        graph_edges: List[Dict[str, Any]],
        node_masks: torch.Tensor,
        elapsed_times: Optional[torch.Tensor] = None,
        masked_pair_idx: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """
        Executes self-supervised pretraining forward pass.
        Randomly masks one relational edge, extracts quantum representation,
        and reconstructs the masked attributes.
        """
        first_node = self.qvbe_config.node_names[0]
        batch_size = node_inputs[first_node][0].shape[0]
        device = node_inputs[first_node][0].device

        # Ground-truth targets from complete graph and node inputs
        gt_weights = torch.tensor([e["weight"] for e in graph_edges], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)
        gt_lags = torch.tensor([e["lag_months"] / 12.0 for e in graph_edges], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)
        gt_se = torch.tensor([e["se"] for e in graph_edges], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)
        gt_dirs = torch.tensor([float(e["direction"]) for e in graph_edges], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)

        gt_node_states = torch.cat([node_inputs[name][0][:, :1] for name in self.qvbe_config.node_names], dim=-1)
        gt_temporal_deltas = torch.cat([node_inputs[name][1][:, :1] for name in self.qvbe_config.node_names], dim=-1)

        targets = {
            "weight": gt_weights,
            "lag": gt_lags,
            "se": gt_se,
            "direction": gt_dirs,
            "node_state": gt_node_states,
            "temporal_delta": gt_temporal_deltas,
        }

        # Apply relational masking: select one edge to mask
        if masked_pair_idx is None:
            masked_pair_idx = int(torch.randint(0, len(graph_edges), (1,)).item())

        masked_edges = []
        for idx, e in enumerate(graph_edges):
            if idx == masked_pair_idx:
                e_masked = copy.deepcopy(e)
                e_masked["weight"] = 0.0
                e_masked["lag_months"] = 0.0
                e_masked["se"] = 0.50
                masked_edges.append(e_masked)
            else:
                masked_edges.append(e)

        # Encode via QVBE-v4 under masked topology
        qvbe_out = self.encoder(
            node_inputs=node_inputs,
            graph_edges=masked_edges,
            node_masks=node_masks,
            elapsed_times=elapsed_times,
        )
        z_q = qvbe_out.motif_embedding

        # Multitask reconstruction
        preds = self.head(z_q)

        # Compute losses
        loss_w = F.mse_loss(preds["pred_weight"][:, masked_pair_idx], targets["weight"][:, masked_pair_idx])
        loss_lag = F.smooth_l1_loss(preds["pred_lag"][:, masked_pair_idx], targets["lag"][:, masked_pair_idx])
        loss_se = F.smooth_l1_loss(preds["pred_se"][:, masked_pair_idx], targets["se"][:, masked_pair_idx])
        loss_dir = F.mse_loss(preds["pred_direction"][:, masked_pair_idx], targets["direction"][:, masked_pair_idx])

        loss_edge = loss_w + 0.5 * loss_lag + 0.5 * loss_se + 0.25 * loss_dir
        loss_temp = F.mse_loss(preds["pred_temporal_delta"], targets["temporal_delta"])
        loss_node = F.mse_loss(preds["pred_node_state"], targets["node_state"])

        total_loss = (
            self.pretrain_config.weight_edge_recon * loss_edge
            + self.pretrain_config.weight_temporal_recon * loss_temp
            + self.pretrain_config.weight_node_recon * loss_node
        )

        loss_dict = {
            "total_loss": total_loss,
            "loss_edge": loss_edge,
            "loss_temporal": loss_temp,
            "loss_node": loss_node,
            "masked_edge_idx": torch.tensor(masked_pair_idx),
        }

        return preds, loss_dict, z_q

    def configure_fine_tuning(self, strategy: str = "full") -> None:
        """
        Configures fine-tuning parameter gradients:
        - 'frozen': freeze entire encoder.
        - 'partial': freeze circuit coupling and decay parameters; fine-tune variational rotation layers & input adapters.
        - 'full': unfreeze all encoder parameters.
        """
        if strategy == "frozen":
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif strategy == "partial":
            for p in self.encoder.circuit.parameters():
                p.requires_grad = False
            # Allow variational local rotations and adapters to fine-tune
            self.encoder.circuit.var_rotations.requires_grad = True
            for a in self.encoder.adapters.values():
                for p in a.parameters():
                    p.requires_grad = True
        elif strategy == "full":
            for p in self.encoder.parameters():
                p.requires_grad = True


class ClassicalRelationalPretrainingModel(nn.Module):
    """
    Matched Classical Relational Pretraining Control.
    Uses identical reconstruction heads, masking strategies, and pretraining losses.
    """
    def __init__(self, embed_dim: int = 16, pretrain_config: Optional[RelationalPretrainingConfig] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.pretrain_config = pretrain_config or RelationalPretrainingConfig()

        self.branch = ClassicalRelationalBranch(out_dim=12, hidden_dim=16)
        self.proj_out = nn.Linear(12, embed_dim)

        self.head = RelationalReconstructionHead(
            embed_dim=embed_dim,
            n_nodes=3,
            n_pairs=3,
        )

    def forward_pretrain(
        self,
        node_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        graph_edges: List[Dict[str, Any]],
        graph_tensors: Dict[str, torch.Tensor],
        node_masks: torch.Tensor,
        masked_pair_idx: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        batch_size = node_masks.shape[0]
        device = node_masks.device

        # Ground-truth targets
        gt_weights = torch.tensor([e["weight"] for e in graph_edges], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)
        gt_lags = torch.tensor([e["lag_months"] / 12.0 for e in graph_edges], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)
        gt_se = torch.tensor([e["se"] for e in graph_edges], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)
        gt_dirs = torch.tensor([float(e["direction"]) for e in graph_edges], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)

        gt_node_states = torch.cat([node_inputs[name][0][:, :1] for name in ["molecular", "structural", "functional"]], dim=-1)
        gt_temporal_deltas = torch.cat([node_inputs[name][1][:, :1] for name in ["molecular", "structural", "functional"]], dim=-1)

        targets = {
            "weight": gt_weights,
            "lag": gt_lags,
            "se": gt_se,
            "direction": gt_dirs,
            "node_state": gt_node_states,
            "temporal_delta": gt_temporal_deltas,
        }

        # Masked index
        if masked_pair_idx is None:
            masked_pair_idx = int(torch.randint(0, len(graph_edges), (1,)).item())

        # Mask graph tensors for the classical branch
        w_masked = graph_tensors["edge_weights"].clone()
        w_masked[:, masked_pair_idx] = 0.0

        lags_masked = graph_tensors["edge_lags"].clone()
        lags_masked[:, masked_pair_idx] = 0.0

        se_masked = graph_tensors["edge_uncertainties"].clone()
        se_masked[:, masked_pair_idx] = 0.50

        dirs_masked = graph_tensors["direction_signs"].clone()

        # Build node_states_3d
        node_states_list = []
        for idx, name in enumerate(["molecular", "structural", "functional"]):
            x_raw, dx_raw = node_inputs[name]
            m = node_masks[:, idx : idx + 1]
            node_states_list.append(x_raw[:, :1] * m)
            node_states_list.append(dx_raw[:, :1] * m)
        node_flat = torch.cat(node_states_list, dim=-1)
        node_states_3d = node_flat.reshape(batch_size, 3, 2)

        # Classical relational message passing
        cr_raw = self.branch(
            node_states_3d=node_states_3d,
            edge_weights=w_masked,
            edge_lags=lags_masked,
            edge_se=se_masked,
            direction_signs=dirs_masked,
        )
        z_cr = F.gelu(self.proj_out(cr_raw))

        # Reconstruct
        preds = self.head(z_cr)

        loss_w = F.mse_loss(preds["pred_weight"][:, masked_pair_idx], targets["weight"][:, masked_pair_idx])
        loss_lag = F.smooth_l1_loss(preds["pred_lag"][:, masked_pair_idx], targets["lag"][:, masked_pair_idx])
        loss_se = F.smooth_l1_loss(preds["pred_se"][:, masked_pair_idx], targets["se"][:, masked_pair_idx])
        loss_dir = F.mse_loss(preds["pred_direction"][:, masked_pair_idx], targets["direction"][:, masked_pair_idx])

        loss_edge = loss_w + 0.5 * loss_lag + 0.5 * loss_se + 0.25 * loss_dir
        loss_temp = F.mse_loss(preds["pred_temporal_delta"], targets["temporal_delta"])
        loss_node = F.mse_loss(preds["pred_node_state"], targets["node_state"])

        total_loss = (
            self.pretrain_config.weight_edge_recon * loss_edge
            + self.pretrain_config.weight_temporal_recon * loss_temp
            + self.pretrain_config.weight_node_recon * loss_node
        )

        loss_dict = {
            "total_loss": total_loss,
            "loss_edge": loss_edge,
            "loss_temporal": loss_temp,
            "loss_node": loss_node,
            "masked_edge_idx": torch.tensor(masked_pair_idx),
        }

        return preds, loss_dict, z_cr

    def configure_fine_tuning(self, strategy: str = "full") -> None:
        if strategy == "frozen":
            for p in self.parameters():
                p.requires_grad = False
        elif strategy == "partial":
            for p in self.branch.parameters():
                p.requires_grad = False
            self.proj_out.weight.requires_grad = True
            self.proj_out.bias.requires_grad = True
        elif strategy == "full":
            for p in self.parameters():
                p.requires_grad = True

"""
NĀḌĪ-Q Phase 10A: Temporal State Updater.
Maintains persistent, time-aware patient hidden states across sequential hypergraph visits.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalStateUpdater(nn.Module):
    """
    Recurrent Patient-Level Latent State Updater.
    Updates patient hidden state h_t across visits using hypergraph embedding g_t,
    actual elapsed time delta_t, and modality availability flags.
    """

    def __init__(
        self,
        dim_graph: int = 32,
        dim_hidden: int = 32,
        dim_time_emb: int = 8,
        num_classes: int = 1
    ):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_time_emb = dim_time_emb

        # Time encoding MLP: Delta t (years) -> R^dim_time_emb
        self.time_mlp = nn.Sequential(
            nn.Linear(1, dim_time_emb),
            nn.GELU(),
            nn.Linear(dim_time_emb, dim_time_emb)
        )

        # GRU input: [g_t ; time_emb ; avail_vector (3)]
        gru_in_dim = dim_graph + dim_time_emb + 3
        self.gru_cell = nn.GRUCell(gru_in_dim, dim_hidden)

        # Progression prediction head
        self.classifier = nn.Sequential(
            nn.Linear(dim_hidden, 16),
            nn.GELU(),
            nn.Linear(16, num_classes)
        )

    def init_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.zeros(batch_size, self.dim_hidden, device=device)

    def forward_step(
        self,
        g_t: torch.Tensor,
        dt_years: torch.Tensor,
        avail_t: torch.Tensor,
        h_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single visit recurrence step.
        g_t: (B, dim_graph) hypergraph embedding at visit t
        dt_years: (B, 1) elapsed time from previous visit in years
        avail_t: (B, 3) modality availability flags
        h_prev: (B, dim_hidden) previous patient hidden state
        returns: (h_t, progression_logit)
        """
        time_emb = self.time_mlp(dt_years)
        gru_input = torch.cat([g_t, time_emb, avail_t], dim=-1)
        h_t = self.gru_cell(gru_input, h_prev)
        logit = self.classifier(h_t)
        return h_t, logit

    def forward_sequence(
        self,
        g_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        avail_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Process a multi-visit sequence for a patient or batch of patients.
        g_seq: (B, T, dim_graph)
        dt_seq: (B, T, 1)
        avail_seq: (B, T, 3)
        returns: (h_final, final_logits, list_of_step_logits)
        """
        B, T, _ = g_seq.shape
        device = g_seq.device
        h_curr = self.init_hidden(B, device=device)

        logits_list = []
        for t in range(T):
            g_t = g_seq[:, t, :]
            dt_t = dt_seq[:, t, :]
            av_t = avail_seq[:, t, :]
            h_curr, logit_t = self.forward_step(g_t, dt_t, av_t, h_curr)
            logits_list.append(logit_t)

        final_logits = logits_list[-1]
        return h_curr, final_logits, logits_list

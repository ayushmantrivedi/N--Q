"""
NĀḌĪ-Q Integrated Hybrid Intelligence System Output Data Structures.
Defines unified outputs exposing multimodal predictions, adaptive fusion gates,
component embeddings, and Q-XAI diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import torch

from nadiq.models.qvbe_v4.output import QVBEv4Output
from nadiq.models.quantum.ctclm import CTCLMOutput


@dataclass
class IntegratedOutput:
    """
    Structured output from NĀḌĪ-Q Integrated Hybrid Intelligence System.
    """
    # 1. Primary Downstream Clinical Predictions
    logits: torch.Tensor                                     # (B, 1) or (B,)
    probabilities: torch.Tensor                              # (B, 1) or (B,) in [0, 1]
    fused_representation: torch.Tensor                       # (B, fusion_hidden_dim)

    # 2. Adaptive Contribution Gating & Balance
    quantum_gate_weight: torch.Tensor                        # (B, 1) in [0, 1]
    temporal_gate_weight: torch.Tensor                       # (B, 1) in [0, 1]
    node_gate_weight: torch.Tensor                           # (B, 1) in [0, 1]
    classical_gate_weight: torch.Tensor                      # (B, 1) aggregate classical weight

    # 3. Component Representations (Inspection Interfaces)
    qvbe_output: Optional[QVBEv4Output] = None
    ctclm_output: Optional[CTCLMOutput] = None
    classical_relational_output: Optional[torch.Tensor] = None # (B, classical_rel_dim)

    # 4. Modality Contributions & Availability
    modality_contributions: Dict[str, torch.Tensor] = field(default_factory=dict)
    availability_masks: Optional[torch.Tensor] = None        # (B, n_modalities)

    # 5. Provenance & Diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)

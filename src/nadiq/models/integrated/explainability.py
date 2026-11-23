"""
NĀḌĪ-Q Q-XAI: Faithful Explainable Multimodal Intelligence.
Provides mechanistic attribution, branch ablation, edge relationship importance,
modality contribution tracking, and perturbation invariance verification.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from nadiq.models.integrated.model import NadiQIntegratedModel
from nadiq.models.integrated.output import IntegratedOutput


@dataclass
class ExplanationResult:
    """
    Structured outcome of Q-XAI explanation for a patient or batch of patients.
    """
    baseline_probability: torch.Tensor                     # (B,)
    quantum_attribution: torch.Tensor                     # (B,) Delta P_q = P_base - P_no_q
    temporal_attribution: torch.Tensor                    # (B,) Delta P_t = P_base - P_no_t
    classical_relational_attribution: torch.Tensor        # (B,) Delta P_cr = P_base - P_no_cr
    modality_attributions: Dict[str, torch.Tensor]        # {modality: (B,) Delta P_m}
    edge_attributions: Dict[str, torch.Tensor]            # {edge_pair: (B,) Delta P_e}
    gate_weights: Dict[str, torch.Tensor]                 # {quantum, temporal, node, classical}
    primary_dominance: List[str]                          # List of categorical dominance tags per patient
    faithfulness_diagnostics: Dict[str, Any] = field(default_factory=dict)


class NadiQXAIExplainer:
    """
    Faithful Multimodal Explainability Analyzer for NĀḌĪ-Q.
    Performs rigorous mechanistic branch ablations and perturbation invariance checks.
    """

    def __init__(self, model: NadiQIntegratedModel):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def explain_prediction(
        self,
        node_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        graph_edges: List[Dict[str, Any]],
        ctclm_seq: Optional[torch.Tensor] = None,
        ctclm_times: Optional[torch.Tensor] = None,
        ctclm_mask: Optional[torch.Tensor] = None,
        node_masks: Optional[torch.Tensor] = None,
        elapsed_times: Optional[torch.Tensor] = None,
    ) -> ExplanationResult:
        """
        Computes faithful counterfactual branch and edge attributions.
        """
        first_node = self.model.config.qvbe_config.node_names[0]
        batch_size = node_inputs[first_node][0].shape[0]
        device = node_inputs[first_node][0].device

        if node_masks is None:
            node_masks = torch.ones(batch_size, self.model.config.qvbe_config.n_nodes, device=device)

        # 1. Baseline Prediction
        base_out: IntegratedOutput = self.model(
            node_inputs=node_inputs,
            graph_edges=graph_edges,
            ctclm_seq=ctclm_seq,
            ctclm_times=ctclm_times,
            ctclm_mask=ctclm_mask,
            node_masks=node_masks,
            elapsed_times=elapsed_times,
            mode="full",
        )
        p_base = base_out.probabilities.view(batch_size)

        # 2. Quantum Branch Ablation (Delta P_q)
        no_q_out = self.model(
            node_inputs=node_inputs,
            graph_edges=graph_edges,
            ctclm_seq=ctclm_seq,
            ctclm_times=ctclm_times,
            ctclm_mask=ctclm_mask,
            node_masks=node_masks,
            elapsed_times=elapsed_times,
            force_quantum_suppression=True,
        )
        p_no_q = no_q_out.probabilities.view(batch_size)
        delta_p_q = p_base - p_no_q

        # 3. Temporal CTCLM Branch Ablation (Delta P_t)
        no_t_out = self.model(
            node_inputs=node_inputs,
            graph_edges=graph_edges,
            ctclm_seq=ctclm_seq,
            ctclm_times=ctclm_times,
            ctclm_mask=ctclm_mask,
            node_masks=node_masks,
            elapsed_times=elapsed_times,
            force_ctclm_suppression=True,
        )
        p_no_t = no_t_out.probabilities.view(batch_size)
        delta_p_t = p_base - p_no_t

        # 4. Classical Relational Branch Ablation (Delta P_cr)
        no_cr_out = self.model(
            node_inputs=node_inputs,
            graph_edges=graph_edges,
            ctclm_seq=ctclm_seq,
            ctclm_times=ctclm_times,
            ctclm_mask=ctclm_mask,
            node_masks=node_masks,
            elapsed_times=elapsed_times,
            force_classical_rel_suppression=True,
        )
        p_no_cr = no_cr_out.probabilities.view(batch_size)
        delta_p_cr = p_base - p_no_cr

        # 5. Modality Ablations (Delta P_m)
        modality_attributions = {}
        for m_idx, mod_name in enumerate(self.model.config.qvbe_config.node_names):
            # Mask out this modality
            abl_masks = node_masks.clone()
            abl_masks[:, m_idx] = 0.0

            abl_inputs = {}
            for name, (x, dx) in node_inputs.items():
                if name == mod_name:
                    abl_inputs[name] = (torch.zeros_like(x), torch.zeros_like(dx))
                else:
                    abl_inputs[name] = (x.clone(), dx.clone())

            # If functional is ablated, also suppress temporal CTCLM
            suppress_ctclm = (mod_name == "functional")

            abl_out = self.model(
                node_inputs=abl_inputs,
                graph_edges=graph_edges,
                ctclm_seq=ctclm_seq,
                ctclm_times=ctclm_times,
                ctclm_mask=ctclm_mask,
                node_masks=abl_masks,
                elapsed_times=elapsed_times,
                force_ctclm_suppression=suppress_ctclm,
            )
            p_abl = abl_out.probabilities.view(batch_size)
            modality_attributions[mod_name] = p_base - p_abl

        # 6. Edge Relationship Ablations (Delta P_e)
        edge_attributions = {}
        for edge in graph_edges:
            pair_key = f"{edge['source']}->{edge['target']}"
            abl_edges = []
            for e in graph_edges:
                if e['source'] == edge['source'] and e['target'] == edge['target']:
                    # Zero out edge weight
                    e_zero = copy.deepcopy(e)
                    e_zero['weight'] = 0.0
                    abl_edges.append(e_zero)
                else:
                    abl_edges.append(e)

            e_abl_out = self.model(
                node_inputs=node_inputs,
                graph_edges=abl_edges,
                ctclm_seq=ctclm_seq,
                ctclm_times=ctclm_times,
                ctclm_mask=ctclm_mask,
                node_masks=node_masks,
                elapsed_times=elapsed_times,
            )
            p_e_abl = e_abl_out.probabilities.view(batch_size)
            edge_attributions[pair_key] = p_base - p_e_abl

        # 7. Dominance Assessment per Sample
        dominance_labels = []
        for i in range(batch_size):
            q_impact = abs(delta_p_q[i].item())
            t_impact = abs(delta_p_t[i].item())
            cr_impact = abs(delta_p_cr[i].item())
            m_impacts = {m: abs(modality_attributions[m][i].item()) for m in modality_attributions}
            max_m_impact = max(m_impacts.values()) if m_impacts else 0.0

            if q_impact > max(t_impact, cr_impact, max_m_impact) and q_impact > 0.05:
                dominance_labels.append("quantum_relational")
            elif t_impact > max(q_impact, cr_impact, max_m_impact) and t_impact > 0.05:
                dominance_labels.append("classical_temporal")
            elif max_m_impact > max(q_impact, t_impact, cr_impact) and max_m_impact > 0.05:
                dominance_labels.append("classical_static")
            else:
                dominance_labels.append("balanced_hybrid")

        gate_weights = {
            "quantum": base_out.quantum_gate_weight.view(batch_size),
            "temporal": base_out.temporal_gate_weight.view(batch_size),
            "node": base_out.node_gate_weight.view(batch_size),
            "classical": base_out.classical_gate_weight.view(batch_size),
        }

        return ExplanationResult(
            baseline_probability=p_base,
            quantum_attribution=delta_p_q,
            temporal_attribution=delta_p_t,
            classical_relational_attribution=delta_p_cr,
            modality_attributions=modality_attributions,
            edge_attributions=edge_attributions,
            gate_weights=gate_weights,
            primary_dominance=dominance_labels,
            faithfulness_diagnostics={
                "mean_abs_quantum_attr": float(torch.abs(delta_p_q).mean().item()),
                "mean_abs_temporal_attr": float(torch.abs(delta_p_t).mean().item()),
                "mean_quantum_gate": float(gate_weights["quantum"].mean().item()),
                "mean_temporal_gate": float(gate_weights["temporal"].mean().item()),
            },
        )

    @torch.no_grad()
    def verify_perturbation_invariance(
        self,
        node_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        graph_edges: List[Dict[str, Any]],
        ctclm_seq: Optional[torch.Tensor] = None,
        ctclm_times: Optional[torch.Tensor] = None,
        ctclm_mask: Optional[torch.Tensor] = None,
        node_masks: Optional[torch.Tensor] = None,
        elapsed_times: Optional[torch.Tensor] = None,
        target_modality_to_mask: str = "molecular",
    ) -> Dict[str, Any]:
        """
        Verifies that perturbing an input when its modality is marked absent (mask=0)
        yields zero change in prediction output (faithfulness check).
        """
        first_node = self.model.config.qvbe_config.node_names[0]
        batch_size = node_inputs[first_node][0].shape[0]
        device = node_inputs[first_node][0].device

        if node_masks is None:
            node_masks = torch.ones(batch_size, self.model.config.qvbe_config.n_nodes, device=device)

        mod_idx = self.model.config.qvbe_config.node_names.index(target_modality_to_mask)
        masked_node_masks = node_masks.clone()
        masked_node_masks[:, mod_idx] = 0.0

        # Run unperturbed with mask=0
        out_unperturbed = self.model(
            node_inputs=node_inputs,
            graph_edges=graph_edges,
            ctclm_seq=ctclm_seq,
            ctclm_times=ctclm_times,
            ctclm_mask=ctclm_mask,
            node_masks=masked_node_masks,
            elapsed_times=elapsed_times,
        )

        # Perturb the masked modality inputs heavily (+10.0)
        perturbed_inputs = {}
        for name, (x, dx) in node_inputs.items():
            if name == target_modality_to_mask:
                perturbed_inputs[name] = (x + 10.0, dx + 10.0)
            else:
                perturbed_inputs[name] = (x.clone(), dx.clone())

        out_perturbed = self.model(
            node_inputs=perturbed_inputs,
            graph_edges=graph_edges,
            ctclm_seq=ctclm_seq,
            ctclm_times=ctclm_times,
            ctclm_mask=ctclm_mask,
            node_masks=masked_node_masks,
            elapsed_times=elapsed_times,
        )

        prob_diff = torch.abs(out_unperturbed.probabilities - out_perturbed.probabilities)
        max_prob_diff = float(prob_diff.max().item())

        return {
            "tested_modality": target_modality_to_mask,
            "max_prob_difference": max_prob_diff,
            "is_invariant": bool(max_prob_diff < 1e-4),
        }

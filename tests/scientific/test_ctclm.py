"""
Comprehensive tests for CTCLM V1 (Continual Temporal Clinical Latent Model).

Tests cover:
  1.  Output shapes
  2.  Deterministic initialization
  3.  Gradient propagation
  4.  Variable sequence lengths
  5.  Padding mask correctness
  6.  Single-visit sequence
  7.  Irregular time offsets
  8.  Zero time offset
  9.  Batch processing
  10. CPU execution
  11. Finite outputs (no NaN/Inf)
  12. No NaN / Inf in any output
  13. Final representation uses LAST VALID timestep
  14. Model parameter count sanity

IMPORTANT: These tests verify computational correctness only. They do NOT
demonstrate clinical superiority, quantum advantage, or continual-learning
resistance. CTCLM V1 is a baseline temporal representation mechanism.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from nadiq.models.quantum.ctclm import (
    CTCLMConfig,
    CTCLMOutput,
    ContinualTemporalClinicalLatentModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> CTCLMConfig:
    return CTCLMConfig(
        input_dim=4,
        latent_dim=16,
        time_embedding_dim=8,
        num_layers=1,
        dropout=0.0,
        seed=42,
    )


@pytest.fixture
def default_ctclm(default_config: CTCLMConfig) -> ContinualTemporalClinicalLatentModel:
    return ContinualTemporalClinicalLatentModel(default_config)


@pytest.fixture
def batch_input(default_config: CTCLMConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (x, time_offsets, padding_mask) for a batch of 3 patients, T=5."""
    torch.manual_seed(0)
    B, T, D = 3, 5, default_config.input_dim
    x = torch.randn(B, T, D)
    # Irregular time offsets (days from first visit)
    time_offsets = torch.tensor([
        [0.0,  30.0, 180.0, 360.0, 720.0],  # patient A: 5 visits
        [0.0,  60.0, 240.0,   0.0,   0.0],  # patient B: 3 visits, 2 padded
        [0.0, 120.0,   0.0,   0.0,   0.0],  # patient C: 2 visits, 3 padded
    ])
    padding_mask = torch.tensor([
        [True, True, True, True, True],   # all valid
        [True, True, True, False, False],  # 3 valid, 2 padded
        [True, True, False, False, False], # 2 valid, 3 padded
    ])
    return x, time_offsets, padding_mask


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------


class TestCTCLMConfig:
    def test_gru_input_dim(self) -> None:
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8)
        assert cfg.gru_input_dim == 4 + 8

    def test_invalid_input_dim(self) -> None:
        with pytest.raises(ValueError, match="input_dim"):
            CTCLMConfig(input_dim=0)

    def test_invalid_latent_dim(self) -> None:
        with pytest.raises(ValueError, match="latent_dim"):
            CTCLMConfig(input_dim=4, latent_dim=0)

    def test_invalid_time_embedding_dim(self) -> None:
        with pytest.raises(ValueError, match="time_embedding_dim"):
            CTCLMConfig(input_dim=4, time_embedding_dim=0)

    def test_invalid_num_layers(self) -> None:
        with pytest.raises(ValueError, match="num_layers"):
            CTCLMConfig(input_dim=4, num_layers=0)

    def test_invalid_dropout(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            CTCLMConfig(input_dim=4, dropout=1.0)


# ---------------------------------------------------------------------------
# 1. Output shapes
# ---------------------------------------------------------------------------


class TestCTCLMOutputShapes:
    def test_sequence_shape(
        self,
        default_ctclm: ContinualTemporalClinicalLatentModel,
        batch_input: tuple,
    ) -> None:
        x, time_offsets, padding_mask = batch_input
        B, T, _ = x.shape
        out = default_ctclm(x, time_offsets, padding_mask)
        assert isinstance(out, CTCLMOutput)
        assert out.sequence.shape == (B, T, default_ctclm.config.latent_dim)

    def test_h_final_shape(
        self,
        default_ctclm: ContinualTemporalClinicalLatentModel,
        batch_input: tuple,
    ) -> None:
        x, time_offsets, padding_mask = batch_input
        B = x.shape[0]
        out = default_ctclm(x, time_offsets, padding_mask)
        assert out.h_final.shape == (B, default_ctclm.config.latent_dim)

    def test_seq_lengths_shape(
        self,
        default_ctclm: ContinualTemporalClinicalLatentModel,
        batch_input: tuple,
    ) -> None:
        x, time_offsets, padding_mask = batch_input
        B = x.shape[0]
        out = default_ctclm(x, time_offsets, padding_mask)
        assert out.seq_lengths.shape == (B,)
        assert out.seq_lengths.dtype == torch.long

    def test_seq_lengths_values(
        self,
        default_ctclm: ContinualTemporalClinicalLatentModel,
        batch_input: tuple,
    ) -> None:
        x, time_offsets, padding_mask = batch_input
        out = default_ctclm(x, time_offsets, padding_mask)
        expected = torch.tensor([5, 3, 2], dtype=torch.long)
        assert torch.equal(out.seq_lengths, expected)


# ---------------------------------------------------------------------------
# 2. Deterministic initialization
# ---------------------------------------------------------------------------


class TestCTCLMDeterministicInit:
    def test_same_seed_same_weights(self) -> None:
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        m1 = ContinualTemporalClinicalLatentModel(cfg)
        m2 = ContinualTemporalClinicalLatentModel(cfg)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)

    def test_different_seeds_different_weights(self) -> None:
        cfg1 = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        cfg2 = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=99)
        m1 = ContinualTemporalClinicalLatentModel(cfg1)
        m2 = ContinualTemporalClinicalLatentModel(cfg2)
        any_different = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(m1.parameters(), m2.parameters())
        )
        assert any_different

    def test_forward_deterministic_same_input(
        self,
        default_ctclm: ContinualTemporalClinicalLatentModel,
        batch_input: tuple,
    ) -> None:
        x, t, m = batch_input
        default_ctclm.eval()
        out1 = default_ctclm(x, t, m)
        out2 = default_ctclm(x, t, m)
        assert torch.allclose(out1.h_final, out2.h_final, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Gradient propagation
# ---------------------------------------------------------------------------


class TestCTCLMGradients:
    def test_gradients_flow_to_gru(
        self,
        default_ctclm: ContinualTemporalClinicalLatentModel,
        batch_input: tuple,
    ) -> None:
        x, t, m = batch_input
        default_ctclm.train()
        out = default_ctclm(x, t, m)
        loss = out.h_final.mean()
        loss.backward()
        gru_grad = default_ctclm.gru.weight_ih_l0.grad
        assert gru_grad is not None
        assert not torch.all(gru_grad == 0)

    def test_gradients_flow_to_time_encoder(
        self,
        default_ctclm: ContinualTemporalClinicalLatentModel,
        batch_input: tuple,
    ) -> None:
        x, t, m = batch_input
        default_ctclm.train()
        out = default_ctclm(x, t, m)
        loss = out.h_final.mean()
        loss.backward()
        te_weight_grad = list(default_ctclm.time_encoder.parameters())[0].grad
        assert te_weight_grad is not None
        assert not torch.all(te_weight_grad == 0)

    def test_gradient_flows_from_sequence_too(
        self,
        default_ctclm: ContinualTemporalClinicalLatentModel,
        batch_input: tuple,
    ) -> None:
        x, t, m = batch_input
        default_ctclm.train()
        out = default_ctclm(x, t, m)
        # Use sequence output, not just h_final
        loss = out.sequence.mean()
        loss.backward()
        assert default_ctclm.gru.weight_ih_l0.grad is not None


# ---------------------------------------------------------------------------
# 4. Variable sequence lengths
# ---------------------------------------------------------------------------


class TestCTCLMVariableLengths:
    @pytest.mark.parametrize("lengths", [[3, 2, 1], [5, 5, 5], [1, 1, 1], [4, 3, 2, 1]])
    def test_variable_lengths_no_crash(self, lengths: list[int]) -> None:
        B = len(lengths)
        T = max(lengths)
        D = 4
        cfg = CTCLMConfig(input_dim=D, latent_dim=16, time_embedding_dim=8, seed=42)
        ctclm = ContinualTemporalClinicalLatentModel(cfg)
        x = torch.randn(B, T, D)
        time_offsets = torch.zeros(B, T)
        mask = torch.zeros(B, T, dtype=torch.bool)
        for i, L in enumerate(lengths):
            mask[i, :L] = True
        out = ctclm(x, time_offsets, mask)
        assert out.h_final.shape == (B, 16)
        assert out.sequence.shape == (B, T, 16)

    def test_seq_lengths_extracted_correctly(self) -> None:
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8)
        ctclm = ContinualTemporalClinicalLatentModel(cfg)
        B, T, D = 4, 6, 4
        x = torch.randn(B, T, D)
        lengths = [6, 4, 2, 1]
        mask = torch.zeros(B, T, dtype=torch.bool)
        for i, L in enumerate(lengths):
            mask[i, :L] = True
        out = ctclm(x, None, mask)
        assert out.seq_lengths.tolist() == lengths


# ---------------------------------------------------------------------------
# 5. Padding mask correctness
# ---------------------------------------------------------------------------


class TestCTCLMPaddingMask:
    def test_padded_positions_do_not_change_h_final(self) -> None:
        """
        Adding extra padding to a sequence should NOT change h_final of
        the original sequence.
        """
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        ctclm = ContinualTemporalClinicalLatentModel(cfg)
        ctclm.eval()

        # Original: 1 patient, T=3
        x_orig = torch.randn(1, 3, 4)
        t_orig = torch.tensor([[0.0, 30.0, 90.0]])
        mask_orig = torch.ones(1, 3, dtype=torch.bool)
        out_orig = ctclm(x_orig, t_orig, mask_orig)

        # Same patient, padded to T=5
        x_padded = torch.cat([x_orig, torch.zeros(1, 2, 4)], dim=1)
        t_padded = torch.tensor([[0.0, 30.0, 90.0, 0.0, 0.0]])
        mask_padded = torch.tensor([[True, True, True, False, False]])
        out_padded = ctclm(x_padded, t_padded, mask_padded)

        assert torch.allclose(out_orig.h_final, out_padded.h_final, atol=1e-5), (
            "Adding padded positions changed h_final — masking is broken."
        )

    def test_no_mask_uses_all_timesteps(self) -> None:
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        ctclm = ContinualTemporalClinicalLatentModel(cfg)
        ctclm.eval()
        x = torch.randn(2, 4, 4)
        out = ctclm(x, None, None)
        # All 4 timesteps should be valid
        assert out.seq_lengths.tolist() == [4, 4]


# ---------------------------------------------------------------------------
# 6. Single-visit sequence
# ---------------------------------------------------------------------------


class TestCTCLMSingleVisit:
    def test_single_visit_batch(self) -> None:
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        ctclm = ContinualTemporalClinicalLatentModel(cfg)
        B, T, D = 3, 1, 4
        x = torch.randn(B, T, D)
        time_offsets = torch.zeros(B, T)
        mask = torch.ones(B, T, dtype=torch.bool)
        out = ctclm(x, time_offsets, mask)
        assert out.h_final.shape == (B, 16)
        assert out.seq_lengths.tolist() == [1, 1, 1]

    def test_mixed_single_and_multi_visit(self) -> None:
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        ctclm = ContinualTemporalClinicalLatentModel(cfg)
        B, T, D = 3, 4, 4
        x = torch.randn(B, T, D)
        # Patient 0: 1 visit, Patient 1: 4 visits, Patient 2: 2 visits
        mask = torch.tensor([
            [True, False, False, False],
            [True, True,  True,  True ],
            [True, True,  False, False],
        ])
        out = ctclm(x, None, mask)
        assert out.seq_lengths.tolist() == [1, 4, 2]
        assert out.h_final.shape == (B, 16)


# ---------------------------------------------------------------------------
# 7. Irregular time offsets
# ---------------------------------------------------------------------------


class TestCTCLMIrregularTime:
    def test_irregular_offsets_no_crash(self) -> None:
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        ctclm = ContinualTemporalClinicalLatentModel(cfg)
        B, T, D = 2, 5, 4
        x = torch.randn(B, T, D)
        # Very irregular intervals (e.g., days from first visit)
        time_offsets = torch.tensor([
            [0.0,  7.0, 30.0, 365.0, 1825.0],
            [0.0, 14.0, 90.0, 180.0,  540.0],
        ])
        out = ctclm(x, time_offsets, None)
        assert out.h_final.shape == (B, 16)
        assert torch.isfinite(out.h_final).all()

    def test_large_time_offsets(self) -> None:
        """Very large Δt values (e.g., thousands of days) must not blow up."""
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        ctclm = ContinualTemporalClinicalLatentModel(cfg)
        x = torch.randn(1, 3, 4)
        time_offsets = torch.tensor([[0.0, 3650.0, 7300.0]])  # 10 years apart
        out = ctclm(x, time_offsets, None)
        assert torch.isfinite(out.h_final).all()


# ---------------------------------------------------------------------------
# 8. Zero time offset
# ---------------------------------------------------------------------------


class TestCTCLMZeroOffset:
    def test_zero_offsets_no_crash(self) -> None:
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        ctclm = ContinualTemporalClinicalLatentModel(cfg)
        B, T, D = 2, 4, 4
        x = torch.randn(B, T, D)
        time_offsets = torch.zeros(B, T)
        out = ctclm(x, time_offsets, None)
        assert torch.isfinite(out.h_final).all()

    def test_none_offsets_fallback(self) -> None:
        """None time_offsets should use uniform spacing [0, 1, ..., T-1]."""
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        ctclm = ContinualTemporalClinicalLatentModel(cfg)
        x = torch.randn(2, 4, 4)
        out = ctclm(x, None, None)
        assert out.sequence.shape == (2, 4, 16)
        assert torch.isfinite(out.h_final).all()


# ---------------------------------------------------------------------------
# 9. Batch processing
# ---------------------------------------------------------------------------


class TestCTCLMBatch:
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_batch_sizes(self, batch_size: int) -> None:
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        ctclm = ContinualTemporalClinicalLatentModel(cfg)
        x = torch.randn(batch_size, 5, 4)
        out = ctclm(x, None, None)
        assert out.h_final.shape == (batch_size, 16)

    @pytest.mark.parametrize("seq_len", [1, 3, 8, 20])
    def test_sequence_lengths(self, seq_len: int) -> None:
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        ctclm = ContinualTemporalClinicalLatentModel(cfg)
        x = torch.randn(2, seq_len, 4)
        out = ctclm(x, None, None)
        assert out.sequence.shape == (2, seq_len, 16)


# ---------------------------------------------------------------------------
# 10. CPU execution
# ---------------------------------------------------------------------------


class TestCTCLMDevice:
    def test_runs_on_cpu(self) -> None:
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        ctclm = ContinualTemporalClinicalLatentModel(cfg).cpu()
        x = torch.randn(2, 4, 4)
        out = ctclm(x, None, None)
        assert out.h_final.device.type == "cpu"


# ---------------------------------------------------------------------------
# 11–12. Finite outputs / No NaN or Inf
# ---------------------------------------------------------------------------


class TestCTCLMFiniteOutputs:
    def test_no_nan_in_h_final(
        self,
        default_ctclm: ContinualTemporalClinicalLatentModel,
        batch_input: tuple,
    ) -> None:
        x, t, m = batch_input
        out = default_ctclm(x, t, m)
        assert not torch.isnan(out.h_final).any(), "h_final contains NaN"

    def test_no_inf_in_h_final(
        self,
        default_ctclm: ContinualTemporalClinicalLatentModel,
        batch_input: tuple,
    ) -> None:
        x, t, m = batch_input
        out = default_ctclm(x, t, m)
        assert not torch.isinf(out.h_final).any(), "h_final contains Inf"

    def test_no_nan_in_sequence(
        self,
        default_ctclm: ContinualTemporalClinicalLatentModel,
        batch_input: tuple,
    ) -> None:
        x, t, m = batch_input
        out = default_ctclm(x, t, m)
        assert not torch.isnan(out.sequence).any(), "sequence contains NaN"

    def test_nan_inputs_handled(self) -> None:
        """NaN inputs should be zeroed, not crash."""
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        ctclm = ContinualTemporalClinicalLatentModel(cfg)
        x = torch.full((2, 3, 4), float("nan"))
        out = ctclm(x, None, None)
        assert torch.isfinite(out.h_final).all()


# ---------------------------------------------------------------------------
# 13. Final representation uses LAST VALID timestep
# ---------------------------------------------------------------------------


class TestCTCLMLastValidTimestep:
    def test_h_final_matches_sequence_at_last_valid_idx(self) -> None:
        """h_final must equal sequence[:, last_valid_idx, :] per sample."""
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        ctclm = ContinualTemporalClinicalLatentModel(cfg)
        ctclm.eval()

        B, T, D = 3, 5, 4
        x = torch.randn(B, T, D)
        lengths = [5, 3, 1]
        mask = torch.zeros(B, T, dtype=torch.bool)
        for i, L in enumerate(lengths):
            mask[i, :L] = True

        out = ctclm(x, None, mask)

        for i, L in enumerate(lengths):
            expected = out.sequence[i, L - 1, :]  # last valid position
            actual = out.h_final[i]
            assert torch.allclose(expected, actual, atol=1e-6), (
                f"Patient {i}: h_final does not match sequence at last valid idx {L-1}"
            )

    def test_padding_beyond_last_valid_does_not_affect_h_final(self) -> None:
        """
        Filling PAD positions with extreme values should not change h_final.
        """
        cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        ctclm = ContinualTemporalClinicalLatentModel(cfg)
        ctclm.eval()

        x_clean = torch.randn(1, 4, 4)
        mask = torch.tensor([[True, True, False, False]])
        out_clean = ctclm(x_clean, None, mask)

        # Replace padded positions with large values
        x_dirty = x_clean.clone()
        x_dirty[0, 2:, :] = 1e6
        out_dirty = ctclm(x_dirty, None, mask)

        # h_final must be identical since masked positions are not processed
        assert torch.allclose(out_clean.h_final, out_dirty.h_final, atol=1e-5), (
            "Padded position values leaked into h_final."
        )


# ---------------------------------------------------------------------------
# 14. Parameter count sanity
# ---------------------------------------------------------------------------


class TestCTCLMParameterCount:
    def test_n_parameters_positive(self, default_ctclm: ContinualTemporalClinicalLatentModel) -> None:
        assert default_ctclm.n_parameters > 0

    def test_n_parameters_scales_with_latent_dim(self) -> None:
        cfg1 = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
        cfg2 = CTCLMConfig(input_dim=4, latent_dim=32, time_embedding_dim=8, seed=42)
        m1 = ContinualTemporalClinicalLatentModel(cfg1)
        m2 = ContinualTemporalClinicalLatentModel(cfg2)
        assert m2.n_parameters > m1.n_parameters

    def test_multilayer_more_params_than_single(self) -> None:
        cfg1 = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, num_layers=1, seed=42)
        cfg2 = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, num_layers=2, dropout=0.1, seed=42)
        m1 = ContinualTemporalClinicalLatentModel(cfg1)
        m2 = ContinualTemporalClinicalLatentModel(cfg2)
        assert m2.n_parameters > m1.n_parameters

    def test_get_config_serialisable(self, default_ctclm: ContinualTemporalClinicalLatentModel) -> None:
        import json
        cfg = default_ctclm.get_config()
        json.dumps(cfg)  # Must not raise

    def test_repr_contains_key_info(self, default_ctclm: ContinualTemporalClinicalLatentModel) -> None:
        r = repr(default_ctclm)
        assert "input_dim" in r
        assert "latent_dim" in r

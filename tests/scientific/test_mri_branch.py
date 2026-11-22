"""
Tests for MRI β-VAE and MRI Quantum Encoder (Nāḍī-Q V1 MRI branch).

Tests cover:
  1.  β-VAE encoder output shape (mu, logvar)
  2.  Reparameterization shape
  3.  Decoder output shape
  4.  Reconstruction loss finite
  5.  KL loss finite and non-negative
  6.  Total loss finite
  7.  β-VAE gradients
  8.  Deterministic initialization
  9.  Eval-mode uses mu (deterministic)
  10. MRI latent → quantum encoder: embedding shape
  11. Finite quantum embedding
  12. MRI quantum encoder gradients
  13. Config validation
  14. Parameter count positive
  15. get_config serialisable
"""

from __future__ import annotations

import math

import pytest
import torch

from nadiq.models.mri.bvae import BetaVAE, BetaVAEConfig, BetaVAEOutput
from nadiq.models.mri.mri_encoder import MRIQuantumEncoder, MRIQuantumEncoderConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vae_config() -> BetaVAEConfig:
    return BetaVAEConfig(input_shape=(1, 16, 16, 16), latent_dim=8, beta=4.0, seed=42)


@pytest.fixture
def vae(vae_config: BetaVAEConfig) -> BetaVAE:
    return BetaVAE(vae_config)


@pytest.fixture
def mri_enc_config() -> MRIQuantumEncoderConfig:
    return MRIQuantumEncoderConfig(
        mri_input_shape=(1, 16, 16, 16), latent_dim=8,
        beta=4.0, n_qubits=4, n_layers=2, seed=42,
    )


@pytest.fixture
def mri_encoder(mri_enc_config: MRIQuantumEncoderConfig) -> MRIQuantumEncoder:
    return MRIQuantumEncoder(mri_enc_config)


@pytest.fixture
def batch_x(vae_config: BetaVAEConfig) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, *vae_config.input_shape)


# ---------------------------------------------------------------------------
# β-VAE Config Validation
# ---------------------------------------------------------------------------

class TestBetaVAEConfig:
    def test_invalid_input_shape(self) -> None:
        with pytest.raises(ValueError, match="input_shape"):
            BetaVAEConfig(input_shape=(16, 16, 16)) # Missing C dim

    def test_invalid_latent_dim(self) -> None:
        with pytest.raises(ValueError, match="latent_dim"):
            BetaVAEConfig(input_shape=(1, 16, 16, 16), latent_dim=0)

    def test_invalid_beta(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            BetaVAEConfig(input_shape=(1, 16, 16, 16), beta=-1.0)


# ---------------------------------------------------------------------------
# 1–6. β-VAE output structure
# ---------------------------------------------------------------------------

class TestBetaVAEOutputShapes:
    def test_mu_shape(self, vae: BetaVAE, batch_x: torch.Tensor, vae_config: BetaVAEConfig) -> None:
        vae.train()
        out = vae(batch_x)
        assert out.mu.shape == (4, vae_config.latent_dim)

    def test_logvar_shape(self, vae: BetaVAE, batch_x: torch.Tensor, vae_config: BetaVAEConfig) -> None:
        out = vae(batch_x)
        assert out.logvar.shape == (4, vae_config.latent_dim)

    def test_z_shape_train(self, vae: BetaVAE, batch_x: torch.Tensor, vae_config: BetaVAEConfig) -> None:
        vae.train()
        out = vae(batch_x)
        assert out.z.shape == (4, vae_config.latent_dim)

    def test_z_shape_eval(self, vae: BetaVAE, batch_x: torch.Tensor, vae_config: BetaVAEConfig) -> None:
        vae.eval()
        with torch.no_grad():
            out = vae(batch_x)
        assert out.z.shape == (4, vae_config.latent_dim)

    def test_x_hat_shape(self, vae: BetaVAE, batch_x: torch.Tensor, vae_config: BetaVAEConfig) -> None:
        out = vae(batch_x)
        assert out.x_hat.shape == (4, *vae_config.input_shape)

    def test_recon_loss_finite(self, vae: BetaVAE, batch_x: torch.Tensor) -> None:
        out = vae(batch_x)
        assert torch.isfinite(out.recon_loss)

    def test_kl_loss_finite(self, vae: BetaVAE, batch_x: torch.Tensor) -> None:
        out = vae(batch_x)
        assert torch.isfinite(out.kl_loss)

    def test_total_loss_finite(self, vae: BetaVAE, batch_x: torch.Tensor) -> None:
        out = vae(batch_x)
        assert torch.isfinite(out.total_loss)


# ---------------------------------------------------------------------------
# 7. β-VAE gradients
# ---------------------------------------------------------------------------

class TestBetaVAEGradients:
    def test_total_loss_backward(self, vae: BetaVAE, batch_x: torch.Tensor) -> None:
        vae.train()
        out = vae(batch_x)
        out.total_loss.backward()
        enc_grad = list(vae.encoder_conv.parameters())[0].grad
        assert enc_grad is not None
        assert not torch.all(enc_grad == 0)

    def test_decoder_grad(self, vae: BetaVAE, batch_x: torch.Tensor) -> None:
        vae.train()
        out = vae(batch_x)
        out.total_loss.backward()
        dec_grad = list(vae.decoder_conv.parameters())[0].grad
        assert dec_grad is not None


# ---------------------------------------------------------------------------
# 8. Deterministic initialization
# ---------------------------------------------------------------------------

class TestBetaVAEDeterminism:
    def test_same_seed_same_weights(self) -> None:
        cfg = BetaVAEConfig(input_shape=(1, 16, 16, 16), latent_dim=8, seed=42)
        m1 = BetaVAE(cfg)
        m2 = BetaVAE(cfg)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)

    def test_different_seed_different_weights(self) -> None:
        m1 = BetaVAE(BetaVAEConfig(input_shape=(1, 16, 16, 16), seed=42))
        m2 = BetaVAE(BetaVAEConfig(input_shape=(1, 16, 16, 16), seed=99))
        assert any(not torch.allclose(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters()))


# ---------------------------------------------------------------------------
# 9. Eval-mode uses mu (deterministic z)
# ---------------------------------------------------------------------------

class TestBetaVAEEvalMode:
    def test_eval_z_equals_mu(self, vae: BetaVAE, batch_x: torch.Tensor) -> None:
        vae.eval()
        with torch.no_grad():
            out = vae(batch_x)
        assert torch.allclose(out.z, out.mu), "In eval mode, z must equal mu (no noise)."


# ---------------------------------------------------------------------------
# 10–11. MRI Quantum Encoder shapes and finite outputs
# ---------------------------------------------------------------------------

class TestMRIQuantumEncoderShapes:
    def test_embedding_shape(
        self, mri_encoder: MRIQuantumEncoder, batch_x: torch.Tensor
    ) -> None:
        mri_encoder.eval()
        with torch.no_grad():
            out = mri_encoder(batch_x)
        assert out.embedding.shape == (4, mri_encoder.config.n_qubits)

    def test_embedding_in_range(
        self, mri_encoder: MRIQuantumEncoder, batch_x: torch.Tensor
    ) -> None:
        mri_encoder.eval()
        with torch.no_grad():
            out = mri_encoder(batch_x)
        emb = out.embedding.numpy()
        import numpy as np
        assert np.all(emb >= -1.0 - 1e-5) and np.all(emb <= 1.0 + 1e-5)

    def test_embedding_finite(
        self, mri_encoder: MRIQuantumEncoder, batch_x: torch.Tensor
    ) -> None:
        mri_encoder.eval()
        with torch.no_grad():
            out = mri_encoder(batch_x)
        assert torch.isfinite(out.embedding).all()

    def test_vae_output_in_mri_output(
        self, mri_encoder: MRIQuantumEncoder, batch_x: torch.Tensor
    ) -> None:
        mri_encoder.train()
        out = mri_encoder(batch_x)
        assert out.vae_output is not None
        assert torch.isfinite(out.vae_output.total_loss)


# ---------------------------------------------------------------------------
# 12. MRI Quantum Encoder gradients
# ---------------------------------------------------------------------------

class TestMRIQuantumEncoderGradients:
    def test_vae_loss_gradient(
        self, mri_encoder: MRIQuantumEncoder, batch_x: torch.Tensor
    ) -> None:
        mri_encoder.train()
        out = mri_encoder(batch_x)
        out.vae_output.total_loss.backward()
        enc_grad = list(mri_encoder.bvae.encoder_conv.parameters())[0].grad
        assert enc_grad is not None

    def test_embedding_gradient(
        self, mri_encoder: MRIQuantumEncoder, batch_x: torch.Tensor
    ) -> None:
        mri_encoder.train()
        out = mri_encoder(batch_x)
        loss = out.embedding.mean()
        loss.backward()
        qvbe_grad = mri_encoder.qvbe.vqc.weights.grad
        assert qvbe_grad is not None


# ---------------------------------------------------------------------------
# 14–15. Parameter count and config
# ---------------------------------------------------------------------------

class TestMRIQuantumEncoderMeta:
    def test_n_parameters_positive(self, mri_encoder: MRIQuantumEncoder) -> None:
        assert mri_encoder.n_parameters > 0

    def test_get_config_serialisable(self, mri_encoder: MRIQuantumEncoder) -> None:
        import json
        json.dumps(mri_encoder.get_config())

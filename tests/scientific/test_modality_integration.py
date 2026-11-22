"""
Integration + Architecture smoke test — Nāḍī-Q V1 Modality Architecture.

Verifies that all three modality branches produce valid representations
from the same synthetic batch, and that ModalityBundle collects them correctly.

Nāḍī-Q V1 Architecture (FROZEN):
    MRI       -> beta-VAE -> QVBE -> MRI quantum embedding
    Biomarkers -> trajectory analysis -> QVBE -> biomarker quantum embedding
    Kinematics -> CTCLM -> kinematic temporal latent

All three -> QCDF (not yet implemented)

Scientific Position:
    - This is a prototype/architectural implementation.
    - Does NOT establish clinical superiority or quantum advantage.
    - Requires real-data empirical validation.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from nadiq.models.mri.mri_encoder import MRIQuantumEncoder, MRIQuantumEncoderConfig
from nadiq.models.biomarkers.biomarker_encoder import (
    BiomarkerQuantumEncoder,
    BiomarkerQuantumEncoderConfig,
)
from nadiq.models.quantum.ctclm import CTCLMConfig, ContinualTemporalClinicalLatentModel
from nadiq.models.representation import (
    ModalityBundle,
    ModalityRepresentation,
    RepresentationType,
)


B = 4  # Batch size for all integration tests


@pytest.fixture(scope="module")
def mri_encoder() -> MRIQuantumEncoder:
    cfg = MRIQuantumEncoderConfig(
        mri_input_shape=(1, 16, 16, 16), latent_dim=8,
        beta=4.0, n_qubits=4, n_layers=2, seed=42,
    )
    return MRIQuantumEncoder(cfg)


@pytest.fixture(scope="module")
def bio_encoder() -> BiomarkerQuantumEncoder:
    cfg = BiomarkerQuantumEncoderConfig(n_biomarkers=5, n_qubits=4, n_layers=2, seed=42)
    return BiomarkerQuantumEncoder(cfg)


@pytest.fixture(scope="module")
def ctclm() -> ContinualTemporalClinicalLatentModel:
    cfg = CTCLMConfig(input_dim=4, latent_dim=16, time_embedding_dim=8, seed=42)
    return ContinualTemporalClinicalLatentModel(cfg)


# ---------------------------------------------------------------------------
# Branch 1: MRI
# ---------------------------------------------------------------------------

class TestMRIBranchIntegration:
    def test_mri_embedding_shape(self, mri_encoder: MRIQuantumEncoder) -> None:
        mri_encoder.eval()
        x = torch.randn(B, 1, 16, 16, 16)
        with torch.no_grad():
            out = mri_encoder(x)
        assert out.embedding.shape == (B, 4)

    def test_mri_embedding_finite(self, mri_encoder: MRIQuantumEncoder) -> None:
        mri_encoder.eval()
        with torch.no_grad():
            out = mri_encoder(torch.randn(B, 1, 16, 16, 16))
        assert torch.isfinite(out.embedding).all()

    def test_mri_representation_type(self, mri_encoder: MRIQuantumEncoder) -> None:
        mri_encoder.eval()
        with torch.no_grad():
            out = mri_encoder(torch.randn(B, 1, 16, 16, 16))
        rep = ModalityRepresentation(
            modality="mri",
            embedding=out.embedding,
            representation_type=RepresentationType.QUANTUM,
        )
        assert rep.representation_type == RepresentationType.QUANTUM
        assert rep.embedding_dim == 4


# ---------------------------------------------------------------------------
# Branch 2: Biomarkers
# ---------------------------------------------------------------------------

class TestBiomarkerBranchIntegration:
    def test_biomarker_embedding_shape(self, bio_encoder: BiomarkerQuantumEncoder) -> None:
        bio_encoder.eval()
        values = torch.randn(B, 6, 5)
        t = torch.arange(6).float().unsqueeze(0).expand(B, -1) * 30.0
        with torch.no_grad():
            out = bio_encoder(values, t)
        assert out.embedding.shape == (B, 4)

    def test_biomarker_embedding_finite(self, bio_encoder: BiomarkerQuantumEncoder) -> None:
        bio_encoder.eval()
        values = torch.randn(B, 4, 5)
        with torch.no_grad():
            out = bio_encoder(values, None)
        assert torch.isfinite(out.embedding).all()

    def test_biomarker_with_nan_values(self, bio_encoder: BiomarkerQuantumEncoder) -> None:
        """Missing biomarker observations (NaN) must not crash the branch."""
        bio_encoder.eval()
        values = torch.randn(B, 4, 5)
        values[:, 2:, :] = float("nan")  # last 2 visits entirely missing
        with torch.no_grad():
            out = bio_encoder(values, None)
        assert torch.isfinite(out.embedding).all()


# ---------------------------------------------------------------------------
# Branch 3: Kinematics (CTCLM)
# ---------------------------------------------------------------------------

class TestKinematicsBranchIntegration:
    def test_ctclm_sequence_shape(self, ctclm: ContinualTemporalClinicalLatentModel) -> None:
        ctclm.eval()
        T = 5
        x = torch.randn(B, T, 4)
        t = torch.arange(T).float().unsqueeze(0).expand(B, -1) * 90.0
        with torch.no_grad():
            out = ctclm(x, t, None)
        assert out.sequence.shape == (B, T, 16)

    def test_ctclm_h_final_shape(self, ctclm: ContinualTemporalClinicalLatentModel) -> None:
        ctclm.eval()
        x = torch.randn(B, 5, 4)
        with torch.no_grad():
            out = ctclm(x, None, None)
        assert out.h_final.shape == (B, 16)

    def test_ctclm_representation_type(self, ctclm: ContinualTemporalClinicalLatentModel) -> None:
        ctclm.eval()
        x = torch.randn(B, 5, 4)
        with torch.no_grad():
            out = ctclm(x, None, None)
        rep = ModalityRepresentation(
            modality="kinematics",
            embedding=out.h_final,
            representation_type=RepresentationType.CLASSICAL_TEMPORAL,
        )
        assert rep.representation_type == RepresentationType.CLASSICAL_TEMPORAL
        assert rep.embedding_dim == 16


# ---------------------------------------------------------------------------
# ModalityBundle — unified representation contract
# ---------------------------------------------------------------------------

class TestModalityBundle:
    def test_bundle_collects_all_three(
        self,
        mri_encoder: MRIQuantumEncoder,
        bio_encoder: BiomarkerQuantumEncoder,
        ctclm: ContinualTemporalClinicalLatentModel,
    ) -> None:
        """All three branches produce representations that can be bundled."""
        for m in [mri_encoder, bio_encoder, ctclm]:
            m.eval()

        with torch.no_grad():
            mri_out = mri_encoder(torch.randn(B, 1, 16, 16, 16))
            bio_out = bio_encoder(torch.randn(B, 4, 5), None)
            kin_out = ctclm(torch.randn(B, 5, 4), None, None)

        bundle = ModalityBundle(
            mri=ModalityRepresentation("mri", mri_out.embedding, RepresentationType.QUANTUM),
            biomarkers=ModalityRepresentation("biomarkers", bio_out.embedding, RepresentationType.QUANTUM),
            kinematics=ModalityRepresentation("kinematics", kin_out.h_final, RepresentationType.CLASSICAL_TEMPORAL),
        )

        bundle.validate_batch_consistency()
        assert set(bundle.available_modalities) == {"mri", "biomarkers", "kinematics"}
        assert bundle.n_available == 3

    def test_bundle_available_modalities_partial(self) -> None:
        """A bundle with only MRI should report 1 available modality."""
        emb = torch.randn(4, 4)
        bundle = ModalityBundle(
            mri=ModalityRepresentation("mri", emb, RepresentationType.QUANTUM),
        )
        assert bundle.available_modalities == ["mri"]
        assert bundle.n_available == 1

    def test_bundle_inconsistent_batch_raises(self) -> None:
        bundle = ModalityBundle(
            mri=ModalityRepresentation("mri", torch.randn(4, 4), RepresentationType.QUANTUM),
            biomarkers=ModalityRepresentation("biomarkers", torch.randn(8, 4), RepresentationType.QUANTUM),
        )
        with pytest.raises(ValueError, match="batch sizes"):
            bundle.validate_batch_consistency()

    def test_representation_type_not_confused(self) -> None:
        """MRI quantum and CTCLM classical types must be distinct."""
        q_rep = ModalityRepresentation("mri", torch.zeros(4, 4), RepresentationType.QUANTUM)
        c_rep = ModalityRepresentation("kin", torch.zeros(4, 16), RepresentationType.CLASSICAL_TEMPORAL)
        assert q_rep.representation_type != c_rep.representation_type

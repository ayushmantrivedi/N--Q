"""
Tests for NĀḌĪ-Q Pre-PPMI Integration Layer.

⚠️  NON-SCIENTIFIC SOFTWARE TESTS ⚠️
--------------------------------------
All tests in this file use SYNTHETIC TENSORS (torch.randn / torch.zeros).
These tests verify software correctness ONLY:
  - Interface contracts
  - Tensor shape correctness
  - Traceability of patient_id + event_id
  - Missing-modality handling
  - QCDF structural forward pass

These tests do NOT:
  - Validate scientific results
  - Use real patient data
  - Generate clinical metrics
  - Train any model
  - Create fake patients (synthetic tensors ≠ fake clinical data)

Synthetic tensors are used purely as shape-correct placeholders to exercise
the software interface. No clinical conclusions may be drawn from them.
"""

from __future__ import annotations

import pytest
import torch

from nadiq.data.missing_modality import (
    MISSING_MODALITY_PATTERNS,
    describe_availability_pattern,
    make_availability_tensor,
    make_zero_embedding,
    validate_embedding_against_availability,
)
from nadiq.data.modality_assembler import AssemblerConfig, ModalityAssembler
from nadiq.data.patient_timeline import (
    ClinicalState,
    ModalityAvailability,
    ModalityReference,
    PatientTimeline,
)
from nadiq.models.fusion.qcdf import QCDF, QCDFConfig
from nadiq.models.representation import ModalityBundle, ModalityRepresentation, RepresentationType


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def standard_config() -> AssemblerConfig:
    """Standard QCDF-compatible assembler config."""
    return AssemblerConfig(
        mri_embedding_dim=4,
        biomarker_embedding_dim=4,
        kinematics_embedding_dim=32,
    )


@pytest.fixture
def assembler(standard_config: AssemblerConfig) -> ModalityAssembler:
    return ModalityAssembler(standard_config)


@pytest.fixture
def timeline_all_available() -> PatientTimeline:
    return PatientTimeline(
        patient_id="TEST001",
        event_id="BL",
        clinical_state=ClinicalState.PARKINSON_DISEASE,
        modality_availability=ModalityAvailability(mri=True, biomarker=True, kinematics=True),
        dataset_id="test",
    )


@pytest.fixture
def timeline_bio_only() -> PatientTimeline:
    return PatientTimeline(
        patient_id="TEST002",
        event_id="V02",
        clinical_state=ClinicalState.HEALTHY_CONTROL,
        modality_availability=ModalityAvailability(mri=False, biomarker=True, kinematics=False),
        dataset_id="test",
    )


@pytest.fixture
def timeline_no_modalities() -> PatientTimeline:
    return PatientTimeline(
        patient_id="TEST003",
        event_id="SC",
        clinical_state=ClinicalState.UNKNOWN,
        modality_availability=ModalityAvailability(mri=False, biomarker=False, kinematics=False),
        dataset_id="test",
    )


@pytest.fixture
def qcdf_model() -> QCDF:
    config = QCDFConfig(
        mri_dim=4,
        biomarker_dim=4,
        kinematics_dim=32,
        projection_dim=16,
        hidden_dim=32,
        num_classes=2,
        dropout=0.0,
        seed=42,
    )
    return QCDF(config)


# ===========================================================================
# 1. PatientTimeline Tests
# ===========================================================================


class TestPatientTimeline:
    """Tests for PatientTimeline data structure."""

    def test_basic_construction(self):
        t = PatientTimeline(
            patient_id="P001",
            event_id="BL",
        )
        assert t.patient_id == "P001"
        assert t.event_id == "BL"
        assert t.clinical_state == ClinicalState.UNKNOWN
        assert t.modality_availability.n_available() == 0

    def test_sample_key_format(self):
        t = PatientTimeline(patient_id="PATNO123", event_id="V04")
        assert t.sample_key == "PATNO123|V04"

    def test_has_modality_true(self, timeline_all_available: PatientTimeline):
        assert timeline_all_available.has_modality("mri") is True
        assert timeline_all_available.has_modality("biomarker") is True
        assert timeline_all_available.has_modality("kinematics") is True

    def test_has_modality_false(self, timeline_bio_only: PatientTimeline):
        assert timeline_bio_only.has_modality("mri") is False
        assert timeline_bio_only.has_modality("biomarker") is True
        assert timeline_bio_only.has_modality("kinematics") is False

    def test_unknown_modality_returns_false(self, timeline_all_available: PatientTimeline):
        assert timeline_all_available.has_modality("genomics") is False

    def test_modality_ref_roundtrip(self):
        ref = ModalityReference(modality="mri", data_key="/data/mri/sub-001.nii.gz")
        t = PatientTimeline(
            patient_id="P002",
            event_id="BL",
            modality_refs={"mri": ref},
        )
        retrieved = t.get_modality_ref("mri")
        assert retrieved is not None
        assert retrieved.data_key == "/data/mri/sub-001.nii.gz"

    def test_missing_modality_ref_returns_none(self, timeline_bio_only: PatientTimeline):
        assert timeline_bio_only.get_modality_ref("mri") is None

    def test_n_available_counts(self):
        avail = ModalityAvailability(mri=True, biomarker=True, kinematics=False)
        assert avail.n_available() == 2

    def test_availability_as_list(self):
        avail = ModalityAvailability(mri=True, biomarker=False, kinematics=True)
        assert avail.as_list() == [True, False, True]

    def test_repr_contains_key_info(self, timeline_all_available: PatientTimeline):
        r = repr(timeline_all_available)
        assert "TEST001" in r
        assert "BL" in r

    def test_ppmi_style_patno_event_id(self):
        """Verify PPMI identifier mapping works."""
        t = PatientTimeline(
            patient_id="3001",    # PPMI PATNO
            event_id="BL",        # PPMI EVENT_ID
            clinical_state=ClinicalState.PARKINSON_DISEASE,
            dataset_id="ppmi",
        )
        assert t.sample_key == "3001|BL"
        assert t.dataset_id == "ppmi"


# ===========================================================================
# 2. ModalityAssembler Tests
# ===========================================================================


class TestModalityAssembler:
    """Tests for ModalityAssembler interface and tensor contracts."""

    def test_all_modalities_present(
        self,
        assembler: ModalityAssembler,
        timeline_all_available: PatientTimeline,
    ):
        """⚠️ SYNTHETIC TENSORS — software test only."""
        mri_z = torch.randn(1, 4)
        bio_z = torch.randn(1, 4)
        kin_z = torch.randn(1, 32)

        sample = assembler.assemble(
            timeline=timeline_all_available,
            mri_embedding=mri_z,
            biomarker_embedding=bio_z,
            kinematics_embedding=kin_z,
        )

        assert sample.bundle.mri is not None
        assert sample.bundle.mri.available is True
        assert sample.bundle.mri.embedding.shape == (1, 4)
        assert sample.bundle.biomarkers is not None
        assert sample.bundle.biomarkers.available is True
        assert sample.bundle.biomarkers.embedding.shape == (1, 4)
        assert sample.bundle.kinematics is not None
        assert sample.bundle.kinematics.available is True
        assert sample.bundle.kinematics.embedding.shape == (1, 32)

    def test_bio_only_absent_modalities_are_zeroed(
        self,
        assembler: ModalityAssembler,
        timeline_bio_only: PatientTimeline,
    ):
        """⚠️ SYNTHETIC TENSORS — verifies zero-tensor + available=False contract."""
        bio_z = torch.randn(1, 4)

        sample = assembler.assemble(
            timeline=timeline_bio_only,
            mri_embedding=None,
            biomarker_embedding=bio_z,
            kinematics_embedding=None,
        )

        # MRI: absent — zero tensor
        assert sample.bundle.mri is not None
        assert sample.bundle.mri.available is False
        assert sample.bundle.mri.embedding.shape == (1, 4)
        assert torch.all(sample.bundle.mri.embedding == 0.0)

        # BIO: present
        assert sample.bundle.biomarkers.available is True

        # KIN: absent — zero tensor
        assert sample.bundle.kinematics.available is False
        assert sample.bundle.kinematics.embedding.shape == (1, 32)
        assert torch.all(sample.bundle.kinematics.embedding == 0.0)

    def test_no_modalities_all_zeroed(
        self,
        assembler: ModalityAssembler,
        timeline_no_modalities: PatientTimeline,
    ):
        """⚠️ SYNTHETIC TENSORS — all modalities absent."""
        sample = assembler.assemble(timeline=timeline_no_modalities)
        for rep in [sample.bundle.mri, sample.bundle.biomarkers, sample.bundle.kinematics]:
            assert rep is not None
            assert rep.available is False
            assert torch.all(rep.embedding == 0.0)

    def test_batch_size_propagated(
        self,
        assembler: ModalityAssembler,
        timeline_all_available: PatientTimeline,
    ):
        """⚠️ SYNTHETIC TENSORS — batch size B=4."""
        B = 4
        mri_z = torch.randn(B, 4)
        bio_z = torch.randn(B, 4)
        kin_z = torch.randn(B, 32)

        sample = assembler.assemble(
            timeline=timeline_all_available,
            mri_embedding=mri_z,
            biomarker_embedding=bio_z,
            kinematics_embedding=kin_z,
            batch_size=B,
        )

        assert sample.bundle.mri.embedding.shape[0] == B
        assert sample.bundle.biomarkers.embedding.shape[0] == B
        assert sample.bundle.kinematics.embedding.shape[0] == B

    def test_dim_mismatch_raises(
        self,
        assembler: ModalityAssembler,
        timeline_all_available: PatientTimeline,
    ):
        """Wrong embedding dimension raises ValueError."""
        bad_mri = torch.randn(1, 8)  # expected 4
        with pytest.raises(ValueError, match="Embedding dim mismatch"):
            assembler.assemble(
                timeline=timeline_all_available,
                mri_embedding=bad_mri,
            )


# ===========================================================================
# 3. Traceability Tests
# ===========================================================================


class TestTraceability:
    """Every assembled sample must preserve patient_id and event_id."""

    def test_patient_id_preserved(
        self,
        assembler: ModalityAssembler,
        timeline_all_available: PatientTimeline,
    ):
        sample = assembler.assemble(timeline=timeline_all_available)
        assert sample.patient_id == "TEST001"

    def test_event_id_preserved(
        self,
        assembler: ModalityAssembler,
        timeline_all_available: PatientTimeline,
    ):
        sample = assembler.assemble(timeline=timeline_all_available)
        assert sample.event_id == "BL"

    def test_sample_key_format(
        self,
        assembler: ModalityAssembler,
        timeline_all_available: PatientTimeline,
    ):
        sample = assembler.assemble(timeline=timeline_all_available)
        assert sample.sample_key == "TEST001|BL"

    def test_metadata_includes_dataset_id(
        self,
        assembler: ModalityAssembler,
        timeline_all_available: PatientTimeline,
    ):
        sample = assembler.assemble(timeline=timeline_all_available)
        assert sample.metadata.get("dataset_id") == "test"

    def test_metadata_includes_clinical_state(
        self,
        assembler: ModalityAssembler,
        timeline_all_available: PatientTimeline,
    ):
        sample = assembler.assemble(timeline=timeline_all_available)
        assert sample.metadata.get("clinical_state") == "PD"

    def test_ppmi_style_ids_preserved(self, assembler: ModalityAssembler):
        """Verify PPMI PATNO/EVENT_ID are not truncated or transformed."""
        timeline = PatientTimeline(
            patient_id="3001",
            event_id="V04",
            dataset_id="ppmi",
        )
        sample = assembler.assemble(timeline=timeline)
        assert sample.patient_id == "3001"
        assert sample.event_id == "V04"
        assert sample.sample_key == "3001|V04"


# ===========================================================================
# 4. Missing-Modality Utility Tests
# ===========================================================================


class TestMissingModalityUtilities:
    """Tests for missing_modality.py utilities."""

    def test_make_zero_embedding_shape(self):
        z = make_zero_embedding(batch_size=3, embedding_dim=4)
        assert z.shape == (3, 4)
        assert torch.all(z == 0.0)

    def test_make_availability_tensor_all_true(self):
        avail = ModalityAvailability(mri=True, biomarker=True, kinematics=True)
        t = make_availability_tensor(avail)
        assert t.shape == (3,)
        assert t.tolist() == [1.0, 1.0, 1.0]

    def test_make_availability_tensor_bio_only(self):
        avail = ModalityAvailability(mri=False, biomarker=True, kinematics=False)
        t = make_availability_tensor(avail)
        assert t.tolist() == [0.0, 1.0, 0.0]

    def test_make_availability_tensor_none(self):
        avail = ModalityAvailability(mri=False, biomarker=False, kinematics=False)
        t = make_availability_tensor(avail)
        assert t.tolist() == [0.0, 0.0, 0.0]

    def test_validate_ok_when_available_with_embedding(self):
        emb = torch.randn(1, 4)
        validate_embedding_against_availability("mri", emb, available=True, expected_dim=4)

    def test_validate_ok_when_absent_with_none(self):
        validate_embedding_against_availability("mri", None, available=False, expected_dim=4)

    def test_validate_raises_available_but_none(self):
        with pytest.raises(ValueError, match="available but embedding is None"):
            validate_embedding_against_availability("mri", None, available=True, expected_dim=4)

    def test_validate_raises_unavailable_but_provided(self):
        emb = torch.randn(1, 4)
        with pytest.raises(ValueError, match="unavailable but a non-None embedding"):
            validate_embedding_against_availability("mri", emb, available=False, expected_dim=4)

    def test_validate_raises_dim_mismatch(self):
        emb = torch.randn(1, 8)
        with pytest.raises(ValueError, match="embedding dim"):
            validate_embedding_against_availability("mri", emb, available=True, expected_dim=4)

    def test_describe_all_available(self):
        avail = ModalityAvailability(mri=True, biomarker=True, kinematics=True)
        desc = describe_availability_pattern(avail)
        assert "MRI" in desc and "BIO" in desc and "KIN" in desc

    def test_describe_none_available(self):
        avail = ModalityAvailability(mri=False, biomarker=False, kinematics=False)
        desc = describe_availability_pattern(avail)
        assert "NONE" in desc

    def test_missing_patterns_are_complete(self):
        assert "all_available" in MISSING_MODALITY_PATTERNS
        assert "bio_only" in MISSING_MODALITY_PATTERNS
        assert "none_available" in MISSING_MODALITY_PATTERNS

    def test_make_zero_embedding_invalid_args(self):
        with pytest.raises(ValueError):
            make_zero_embedding(batch_size=0, embedding_dim=4)
        with pytest.raises(ValueError):
            make_zero_embedding(batch_size=1, embedding_dim=0)


# ===========================================================================
# 5. QCDF Integration Tests (SOFTWARE ONLY — SYNTHETIC TENSORS)
# ===========================================================================


class TestQCDFIntegration:
    """
    ⚠️  NON-SCIENTIFIC — SYNTHETIC TENSORS — SOFTWARE INTERFACE TEST ONLY ⚠️

    These tests verify that the full pipeline:
      MRI(4) + BIO(4) + KIN(32) + availability(3) → QCDF → prediction

    produces the correct tensor shapes and does not raise errors.
    No scientific conclusions may be drawn from these results.
    """

    def test_all_modalities_full_forward(self, qcdf_model: QCDF):
        """⚠️ SYNTHETIC — verifies (4)+(4)+(32) → QCDF → (B, 2) shape."""
        B = 2
        config = AssemblerConfig(mri_embedding_dim=4, biomarker_embedding_dim=4, kinematics_embedding_dim=32)
        assembler = ModalityAssembler(config)

        timeline = PatientTimeline(
            patient_id="SYNTH001",
            event_id="BL",
            modality_availability=ModalityAvailability(mri=True, biomarker=True, kinematics=True),
            dataset_id="synthetic_test",
        )

        mri_z = torch.randn(B, 4)
        bio_z = torch.randn(B, 4)
        kin_z = torch.randn(B, 32)

        sample = assembler.assemble(
            timeline=timeline,
            mri_embedding=mri_z,
            biomarker_embedding=bio_z,
            kinematics_embedding=kin_z,
            batch_size=B,
        )

        out = qcdf_model(sample.bundle)
        assert out.logits.shape == (B, 2)
        assert out.probabilities.shape == (B, 2)
        assert torch.allclose(out.probabilities.sum(dim=-1), torch.ones(B), atol=1e-5)

    def test_bio_only_forward(self, qcdf_model: QCDF):
        """⚠️ SYNTHETIC — biomarker-only modality pattern."""
        B = 1
        config = AssemblerConfig(mri_embedding_dim=4, biomarker_embedding_dim=4, kinematics_embedding_dim=32)
        assembler = ModalityAssembler(config)

        timeline = PatientTimeline(
            patient_id="SYNTH002",
            event_id="V02",
            modality_availability=ModalityAvailability(mri=False, biomarker=True, kinematics=False),
            dataset_id="synthetic_test",
        )

        bio_z = torch.randn(B, 4)
        sample = assembler.assemble(
            timeline=timeline,
            mri_embedding=None,
            biomarker_embedding=bio_z,
            kinematics_embedding=None,
            batch_size=B,
        )

        out = qcdf_model(sample.bundle)
        assert out.logits.shape == (B, 2)
        assert out.probabilities.shape == (B, 2)

    def test_no_modalities_forward(self, qcdf_model: QCDF):
        """⚠️ SYNTHETIC — all modalities absent; QCDF receives all-zero inputs."""
        B = 1
        config = AssemblerConfig(mri_embedding_dim=4, biomarker_embedding_dim=4, kinematics_embedding_dim=32)
        assembler = ModalityAssembler(config)

        timeline = PatientTimeline(
            patient_id="SYNTH003",
            event_id="BL",
            modality_availability=ModalityAvailability(mri=False, biomarker=False, kinematics=False),
            dataset_id="synthetic_test",
        )

        sample = assembler.assemble(timeline=timeline, batch_size=B)
        out = qcdf_model(sample.bundle)
        assert out.logits.shape == (B, 2)

    def test_availability_tensor_shape(self):
        """Availability tensor has correct shape for all patterns."""
        for name, pattern in MISSING_MODALITY_PATTERNS.items():
            t = make_availability_tensor(pattern)
            assert t.shape == (3,), f"Pattern '{name}' availability tensor shape mismatch"

    def test_traceability_survives_qcdf_forward(self, qcdf_model: QCDF):
        """Patient/event IDs are accessible on assembled sample after QCDF forward."""
        B = 1
        config = AssemblerConfig(mri_embedding_dim=4, biomarker_embedding_dim=4, kinematics_embedding_dim=32)
        assembler = ModalityAssembler(config)
        timeline = PatientTimeline(patient_id="TRACE_001", event_id="V06", dataset_id="synthetic_test",
                                   modality_availability=ModalityAvailability(mri=False, biomarker=True, kinematics=False))

        bio_z = torch.randn(B, 4)
        sample = assembler.assemble(timeline=timeline, biomarker_embedding=bio_z, batch_size=B)
        _ = qcdf_model(sample.bundle)  # run forward pass

        # IDs still accessible
        assert sample.patient_id == "TRACE_001"
        assert sample.event_id == "V06"

    def test_qcdf_bundle_consistency_raises_on_bad_batch(self, qcdf_model: QCDF):
        """ModalityBundle rejects inconsistent batch sizes."""
        mri_rep = ModalityRepresentation(
            modality="mri", embedding=torch.randn(2, 4),
            representation_type=RepresentationType.QUANTUM, available=True
        )
        bio_rep = ModalityRepresentation(
            modality="biomarkers", embedding=torch.randn(3, 4),  # different batch
            representation_type=RepresentationType.QUANTUM, available=True
        )
        bundle = ModalityBundle(mri=mri_rep, biomarkers=bio_rep)
        with pytest.raises(ValueError, match="inconsistent batch sizes"):
            qcdf_model(bundle)

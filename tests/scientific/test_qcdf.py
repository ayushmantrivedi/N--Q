"""
Tests for QCDF (Quantum Clinical Decision Fusion).
"""

import pytest
import torch

from nadiq.models.fusion.qcdf import QCDF, QCDFConfig
from nadiq.models.representation import ModalityBundle, ModalityRepresentation, RepresentationType

@pytest.fixture
def qcdf_config():
    return QCDFConfig(
        mri_dim=4,
        biomarker_dim=4,
        kinematics_dim=32,
        projection_dim=16,
        hidden_dim=32,
        num_classes=3
    )

@pytest.fixture
def qcdf(qcdf_config):
    return QCDF(qcdf_config)

def test_qcdf_all_modalities_available(qcdf):
    B = 2
    bundle = ModalityBundle(
        mri=ModalityRepresentation("mri", torch.randn(B, 4), RepresentationType.QUANTUM, True),
        biomarkers=ModalityRepresentation("biomarkers", torch.randn(B, 4), RepresentationType.QUANTUM, True),
        kinematics=ModalityRepresentation("kinematics", torch.randn(B, 32), RepresentationType.CLASSICAL_TEMPORAL, True)
    )
    
    out = qcdf(bundle)
    assert out.logits.shape == (B, 3)
    assert out.probabilities.shape == (B, 3)
    assert out.fused_vector.shape == (B, 32)
    assert torch.isfinite(out.logits).all()

def test_qcdf_missing_modalities(qcdf):
    B = 2
    # Only MRI available
    bundle = ModalityBundle(
        mri=ModalityRepresentation("mri", torch.randn(B, 4), RepresentationType.QUANTUM, True),
        biomarkers=ModalityRepresentation("biomarkers", torch.zeros(B, 4), RepresentationType.QUANTUM, False),
        kinematics=ModalityRepresentation("kinematics", torch.zeros(B, 32), RepresentationType.CLASSICAL_TEMPORAL, False)
    )
    
    out = qcdf(bundle)
    assert out.logits.shape == (B, 3)
    
    # Verify backward pass works even with missing modalities
    loss = out.logits.sum()
    loss.backward()
    
    # Gradients should exist
    assert qcdf.classifier.weight.grad is not None

def test_qcdf_deterministic(qcdf_config):
    qcdf1 = QCDF(qcdf_config)
    qcdf2 = QCDF(qcdf_config)
    
    for p1, p2 in zip(qcdf1.parameters(), qcdf2.parameters()):
        assert torch.allclose(p1, p2)

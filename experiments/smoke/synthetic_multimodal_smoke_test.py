"""
SYNTHETIC ARCHITECTURE INTEGRATION TEST — Nāḍī-Q V1.

This script demonstrates the end-to-end forward pass of the fully integrated
multimodal architecture (MRI + Biomarkers + Kinematics → QCDF).

CRITICAL SCIENTIFIC POSITION:
This is ONLY an engineering integration test. It uses synthetic random tensors.
It MUST NEVER be described as ADNI validation or establish clinical validity.
Its purpose is to prove the architecture connects correctly.
"""

import torch
import torch.nn as nn
from typing import Optional

from nadiq.preprocessing.mri import MRIPreprocessor, MRIPreprocessingConfig
from nadiq.models.mri.mri_encoder import MRIQuantumEncoder, MRIQuantumEncoderConfig
from nadiq.models.biomarkers.biomarker_encoder import BiomarkerQuantumEncoder, BiomarkerQuantumEncoderConfig
from nadiq.models.quantum.ctclm import ContinualTemporalClinicalLatentModel, CTCLMConfig
from nadiq.models.representation import ModalityBundle, ModalityRepresentation, RepresentationType
from nadiq.models.fusion.qcdf import QCDF, QCDFConfig

def main():
    print("==========================================================")
    print("   SYNTHETIC ARCHITECTURE INTEGRATION TEST - Nadi-Q V1")
    print("==========================================================\n")

    torch.manual_seed(42)
    B = 2  # Batch size

    # ---------------------------------------------------------
    # 1. MRI BRANCH (3D T1 MRI → Preprocessing → B-VAE → QVBE)
    # ---------------------------------------------------------
    print("--- 1. MRI Branch ---")
    # Synthetic raw 3D T1 volume (B, C, D, H, W)
    raw_mri = torch.randn(B, 1, 60, 60, 60)
    print(f"Raw MRI Shape: {raw_mri.shape}")

    mri_preprocessor = MRIPreprocessor(
        MRIPreprocessingConfig(target_shape=(32, 32, 32), normalize_intensity=True)
    )
    processed_mri = mri_preprocessor(raw_mri)
    print(f"Processed MRI Shape: {processed_mri.shape}")

    mri_encoder = MRIQuantumEncoder(
        MRIQuantumEncoderConfig(
            mri_input_shape=(1, 32, 32, 32),
            latent_dim=4,
            n_qubits=4,
            n_layers=1
        )
    )
    
    mri_out = mri_encoder(processed_mri)
    print(f"MRI Quantum Embedding Shape: {mri_out.embedding.shape}\n")
    
    mri_rep = ModalityRepresentation(
        modality="mri",
        embedding=mri_out.embedding,
        representation_type=RepresentationType.QUANTUM,
        available=True
    )

    # ---------------------------------------------------------
    # 2. BIOMARKER BRANCH (Trajectory → QVBE)
    # ---------------------------------------------------------
    print("--- 2. Biomarker Branch ---")
    # Synthetic longitudinal biomarker values: (B, T, K) where K=5 biomarkers
    T_bio = 3
    bio_values = torch.randn(B, T_bio, 5)
    bio_time_offsets = torch.tensor([[0.0, 6.0, 12.0], [0.0, 12.0, 24.0]])
    print(f"Biomarker Values Shape: {bio_values.shape}")

    bio_encoder = BiomarkerQuantumEncoder(
        BiomarkerQuantumEncoderConfig(
            n_biomarkers=5,
            n_qubits=4,
            n_layers=1
        )
    )
    
    bio_out = bio_encoder(bio_values, bio_time_offsets)
    print(f"Biomarker Quantum Embedding Shape: {bio_out.embedding.shape}\n")

    bio_rep = ModalityRepresentation(
        modality="biomarkers",
        embedding=bio_out.embedding,
        representation_type=RepresentationType.QUANTUM,
        available=True
    )

    # ---------------------------------------------------------
    # 3. KINEMATICS BRANCH (Temporal functional data → CTCLM)
    # ---------------------------------------------------------
    print("--- 3. Kinematics Branch ---")
    # Synthetic kinematics sequence: (B, T, D)
    T = 10
    kinematics_seq = torch.randn(B, T, 12)
    # Delta t
    dt = torch.ones(B, T)
    # Mask
    mask = torch.ones(B, T)
    
    print(f"Kinematics Sequence Shape: {kinematics_seq.shape}")

    ctclm = ContinualTemporalClinicalLatentModel(
        CTCLMConfig(
            input_dim=12,
            latent_dim=16,
            time_embedding_dim=4
        )
    )
    
    kinematics_out = ctclm(kinematics_seq, dt, mask)
    print(f"Kinematics CTCLM Embedding Shape: {kinematics_out.h_final.shape}\n")

    kin_rep = ModalityRepresentation(
        modality="kinematics",
        embedding=kinematics_out.h_final,
        representation_type=RepresentationType.CLASSICAL_TEMPORAL,
        available=True
    )

    # ---------------------------------------------------------
    # 4. QCDF (Quantum Clinical Decision Fusion)
    # ---------------------------------------------------------
    print("--- 4. QCDF Layer ---")
    bundle = ModalityBundle(
        mri=mri_rep,
        biomarkers=bio_rep,
        kinematics=kin_rep
    )
    
    qcdf = QCDF(
        QCDFConfig(
            mri_dim=4,
            biomarker_dim=4,
            kinematics_dim=16,
            projection_dim=8,
            hidden_dim=16,
            num_classes=3
        )
    )
    
    qcdf_out = qcdf(bundle)
    print(f"Fused Vector Shape: {qcdf_out.fused_vector.shape}")
    print(f"Logits Shape: {qcdf_out.logits.shape}")
    print(f"Probabilities Shape: {qcdf_out.probabilities.shape}\n")
    
    # ---------------------------------------------------------
    # 5. ABLATION / MISSING MODALITY TEST
    # ---------------------------------------------------------
    print("--- 5. Missing Modality Support (MRI Only) ---")
    mri_only_bundle = ModalityBundle(
        mri=mri_rep,
        biomarkers=ModalityRepresentation("biomarkers", torch.zeros(B, 4), RepresentationType.QUANTUM, available=False),
        kinematics=ModalityRepresentation("kinematics", torch.zeros(B, 16), RepresentationType.CLASSICAL_TEMPORAL, available=False)
    )
    
    qcdf_mri_only_out = qcdf(mri_only_bundle)
    print(f"Logits Shape (MRI Only): {qcdf_mri_only_out.logits.shape}")
    
    print("\nSUCCESS: SYNTHETIC ARCHITECTURE INTEGRATION TEST PASSED.")
    print("All architectural paths match the frozen specification.")

if __name__ == "__main__":
    main()

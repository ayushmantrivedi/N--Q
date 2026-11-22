"""
Nāḍī-Q V1 Architecture Smoke Test — Synthetic Data.

Demonstrates the complete modality-specific architecture using tiny
synthetic tensors with realistic dimensions.

Architecture (FROZEN):
    MRI        -> beta-VAE -> QVBE -> MRI quantum embedding
    Biomarkers -> trajectory analysis -> QVBE -> biomarker quantum embedding
    Kinematics -> CTCLM -> kinematic temporal latent

Scientific Position:
    - Prototype / architectural implementation. NOT clinically validated.
    - Does NOT establish quantum advantage.
    - Real-data validation is required before any clinical claims.

Usage:
    python experiments/architecture_smoke_test.py
"""

from __future__ import annotations

import sys
import torch

from nadiq.models.mri.mri_encoder import MRIQuantumEncoder, MRIQuantumEncoderConfig
from nadiq.models.biomarkers.biomarker_encoder import BiomarkerQuantumEncoder, BiomarkerQuantumEncoderConfig
from nadiq.models.quantum.ctclm import CTCLMConfig, ContinualTemporalClinicalLatentModel
from nadiq.models.representation import ModalityBundle, ModalityRepresentation, RepresentationType


def run() -> None:
    B = 4    # batch size
    SEED = 42

    print("\n" + "=" * 72)
    print("Nadi-Q V1 -- Modality Architecture Smoke Test (Synthetic Data)")
    print("=" * 72)

    torch.manual_seed(SEED)

    # ------------------------------------------------------------------
    # Branch 1: MRI -> beta-VAE -> QVBE
    # ------------------------------------------------------------------
    print("\n[Branch 1] MRI")
    mri_cfg = MRIQuantumEncoderConfig(
        mri_input_dim=256,   # e.g., small patch feature vector
        vae_hidden_dim=64,
        latent_dim=8,
        beta=4.0,
        n_qubits=4,
        n_layers=2,
        seed=SEED,
    )
    mri_enc = MRIQuantumEncoder(mri_cfg)
    mri_enc.eval()
    mri_x = torch.randn(B, 256)   # synthetic MRI features

    with torch.no_grad():
        mri_out = mri_enc(mri_x)

    mri_latent = mri_out.vae_output.mu
    mri_embedding = mri_out.embedding

    print(f"  Input shape            : {tuple(mri_x.shape)}")
    print(f"  beta-VAE latent (mu)   : {tuple(mri_latent.shape)}")
    print(f"  MRI quantum embedding  : {tuple(mri_embedding.shape)}")
    print(f"  Finite                 : {torch.isfinite(mri_embedding).all().item()}")
    print(f"  Range                  : [{mri_embedding.min().item():.4f}, {mri_embedding.max().item():.4f}]")
    print(f"  beta-VAE total loss    : {mri_out.vae_output.total_loss.item():.4f}")
    print(f"  Circuit note           : RY/RZ/RY (U3 variant = future)")

    # ------------------------------------------------------------------
    # Branch 2: Biomarkers -> trajectory analysis -> QVBE
    # ------------------------------------------------------------------
    print("\n[Branch 2] Biomarkers")
    bio_cfg = BiomarkerQuantumEncoderConfig(
        n_biomarkers=5,   # e.g., ADAS13, MMSE, Hippocampus, CDRSB, RAVLT
        n_qubits=4,
        n_layers=2,
        seed=SEED,
    )
    bio_enc = BiomarkerQuantumEncoder(bio_cfg)
    bio_enc.eval()
    T_bio = 6
    bio_values = torch.randn(B, T_bio, 5)
    bio_values[0, 4:, :] = float("nan")   # patient 0: 4 valid, 2 missing
    bio_values[1, 3:, :] = float("nan")   # patient 1: 3 valid
    bio_t = torch.arange(T_bio).float().unsqueeze(0).expand(B, -1) * 30.0  # monthly

    with torch.no_grad():
        bio_out = bio_enc(bio_values, bio_t)

    bio_traj = bio_out.trajectory_output.trajectory
    bio_embedding = bio_out.embedding

    print(f"  Input shape            : {tuple(bio_values.shape)}")
    print(f"  Trajectory dim         : {bio_cfg.trajectory_dim}  (5 biomarkers x 6 features)")
    print(f"  Trajectory shape       : {tuple(bio_traj.shape)}")
    print(f"  Biomarker quantum emb  : {tuple(bio_embedding.shape)}")
    print(f"  Finite                 : {torch.isfinite(bio_embedding).all().item()}")
    print(f"  Range                  : [{bio_embedding.min().item():.4f}, {bio_embedding.max().item():.4f}]")

    # ------------------------------------------------------------------
    # Branch 3: Kinematics -> CTCLM
    # ------------------------------------------------------------------
    print("\n[Branch 3] Kinematics (CTCLM)")
    ctclm_cfg = CTCLMConfig(
        input_dim=4,       # e.g., voice/gait feature dim
        latent_dim=16,
        time_embedding_dim=8,
        num_layers=1,
        dropout=0.0,
        seed=SEED,
    )
    ctclm = ContinualTemporalClinicalLatentModel(ctclm_cfg)
    ctclm.eval()
    T_kin = 8
    kin_x = torch.randn(B, T_kin, 4)
    kin_t = torch.arange(T_kin).float().unsqueeze(0).expand(B, -1) * 90.0  # quarterly
    kin_mask = torch.ones(B, T_kin, dtype=torch.bool)
    kin_mask[-1, 5:] = False  # last patient: only 5 valid visits

    with torch.no_grad():
        kin_out = ctclm(kin_x, kin_t, kin_mask)

    print(f"  Input shape            : {tuple(kin_x.shape)}")
    print(f"  CTCLM sequence output  : {tuple(kin_out.sequence.shape)}")
    print(f"  CTCLM h_final          : {tuple(kin_out.h_final.shape)}")
    print(f"  Seq lengths            : {kin_out.seq_lengths.tolist()}")
    print(f"  Finite                 : {torch.isfinite(kin_out.h_final).all().item()}")

    # ------------------------------------------------------------------
    # Bundle
    # ------------------------------------------------------------------
    print("\n[ModalityBundle]")
    bundle = ModalityBundle(
        mri=ModalityRepresentation("mri", mri_embedding, RepresentationType.QUANTUM),
        biomarkers=ModalityRepresentation("biomarkers", bio_embedding, RepresentationType.QUANTUM),
        kinematics=ModalityRepresentation("kinematics", kin_out.h_final, RepresentationType.CLASSICAL_TEMPORAL),
    )
    bundle.validate_batch_consistency()
    print(f"  Available modalities   : {bundle.available_modalities}")
    print(f"  MRI rep type           : {bundle.mri.representation_type.name}")
    print(f"  Biomarker rep type     : {bundle.biomarkers.representation_type.name}")
    print(f"  Kinematics rep type    : {bundle.kinematics.representation_type.name}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("Parameter Counts")
    print("-" * 72)
    print(f"  MRI encoder total      : {mri_enc.n_parameters}")
    print(f"    beta-VAE             : {mri_enc.bvae.n_parameters}")
    print(f"    QVBE                 : {mri_enc.qvbe.n_parameters}")
    print(f"  Biomarker encoder total: {bio_enc.n_parameters}")
    print(f"    Trajectory (det.)    : {bio_enc.trajectory_encoder.n_parameters}  (deterministic)")
    print(f"    QVBE                 : {bio_enc.qvbe.n_parameters}")
    print(f"  CTCLM total            : {ctclm.n_parameters}")

    all_finite = (
        torch.isfinite(mri_embedding).all()
        and torch.isfinite(bio_embedding).all()
        and torch.isfinite(kin_out.h_final).all()
    )

    print("\n" + "=" * 72)
    print(f"all finite = {all_finite.item()}")
    if not all_finite:
        print("SMOKE TEST FAILED: non-finite values detected.")
        sys.exit(1)
    print("ARCHITECTURE SMOKE TEST PASSED")
    print("=" * 72)


if __name__ == "__main__":
    run()

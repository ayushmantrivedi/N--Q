import os
import pandas as pd

models_data = [
    {
        "model_identifier": "TabPFN-v2",
        "modality_domain": "Multimodal Tabular / Clinical & IDPs",
        "official_repository": "https://github.com/priorlabs/TabPFN",
        "license": "Custom Research Non-Commercial (Prior Labs)",
        "commercial_use_permitted": "No (Non-Commercial Research Only)",
        "pretrained_weights_available": "Yes (HuggingFace prior-labs/TabPFN)",
        "input_schema": "Tabular matrix (N, D), D <= 500, N <= 10,000",
        "output_schema": "Class probabilities or hooked penultimate transformer activations (512-dim)",
        "feature_extraction_supported": "Via PyTorch forward hooks on transformer_encoder (In-context)",
        "local_inference_feasible": "Yes (CPU or local CUDA; no external API required)",
        "memory_ram_required": "~4 GB RAM / 2 GB VRAM",
        "recommendation_status": "AUDITED & RETAINED as Classical Foundation Comparator"
    },
    {
        "model_identifier": "TabNet",
        "modality_domain": "Tabular FreeSurfer Morphometry & Motor Scores",
        "official_repository": "https://github.com/dreamquark-ai/tabnet",
        "license": "MIT License",
        "commercial_use_permitted": "Yes (Fully Open-Source)",
        "pretrained_weights_available": "Self-supervised pretraining available; pretrains on tabular mask task",
        "input_schema": "Standardized continuous/categorical tabular tensors (B, D)",
        "output_schema": "Attentive step embeddings (B, d_step) + Sparse Attention Masks (B, D)",
        "feature_extraction_supported": "Native (step embeddings and latent activations accessible prior to head)",
        "local_inference_feasible": "Yes (Native PyTorch, CPU or CUDA)",
        "memory_ram_required": "< 2 GB RAM",
        "recommendation_status": "RECOMMENDED for Attentive Tabular Representation Extraction"
    },
    {
        "model_identifier": "FT-Transformer",
        "modality_domain": "Tabular FreeSurfer Morphometry & Clinical",
        "official_repository": "https://github.com/yandex-research/rtdl",
        "license": "Apache 2.0 License (Yandex Research)",
        "commercial_use_permitted": "Yes",
        "pretrained_weights_available": "Trained from scratch / transfer via feature tokenization",
        "input_schema": "Normalized continuous features + categorical indices",
        "output_schema": "CLS token representation (B, d_token) where d_token in {64, 128, 192}",
        "feature_extraction_supported": "Native (extract [CLS] token embedding before linear head)",
        "local_inference_feasible": "Yes (PyTorch, lightweight)",
        "memory_ram_required": "< 2 GB RAM",
        "recommendation_status": "HIGHLY COMPATIBLE for Tabular Representation Learning"
    },
    {
        "model_identifier": "MONAI 3D DenseNet / ResNet",
        "modality_domain": "Raw 3D Structural T1 MRI (Physical Scans)",
        "official_repository": "https://github.com/Project-MONAI/MONAI",
        "license": "Apache 2.0 License",
        "commercial_use_permitted": "Yes",
        "pretrained_weights_available": "Yes (MedicalNet 3D ResNet50 / DenseNet121 on 23 medical datasets)",
        "input_schema": "3D volumetric tensor (B, 1, 96, 96, 96) or (B, 1, 128, 128, 128)",
        "output_schema": "Penultimate global pooling embedding (512-dim or 1024-dim)",
        "feature_extraction_supported": "Native (backbone feature extraction without classification head)",
        "local_inference_feasible": "Yes (Local PyTorch / CUDA)",
        "memory_ram_required": "8 GB RAM / 4-6 GB VRAM",
        "recommendation_status": "APPROVED BACKEND for Physical 3D MRI Branch (Cohort N=36)"
    },
    {
        "model_identifier": "MOMENT (Time-Series Foundation)",
        "modality_domain": "Longitudinal Kinematics & Sequential Wearable Sensors",
        "official_repository": "https://github.com/moment-timeseries-foundation-model/moment",
        "license": "Apache 2.0 License (Carnegie Mellon / Auton Lab)",
        "commercial_use_permitted": "Yes",
        "pretrained_weights_available": "Yes (HuggingFace AutonLab/MOMENT-1-large, base, small)",
        "input_schema": "Multi-channel time-series (B, C, T) with patch tokenization",
        "output_schema": "Per-channel or pooled time-series representation (512-dim)",
        "feature_extraction_supported": "Native feature extraction mode (embed mode without forecasting head)",
        "local_inference_feasible": "Yes (MOMENT-small/base runs on local CPU / 4GB VRAM)",
        "memory_ram_required": "4-8 GB RAM",
        "recommendation_status": "APPROVED BACKEND for Kinematic CTCLM Sequential Branch"
    },
    {
        "model_identifier": "Supervised 3D Beta-VAE + PLS (Original)",
        "modality_domain": "Raw 3D Structural MRI (NĀḌĪ-Q Native Architecture)",
        "official_repository": "In-repo (src/nadiq/models/mri/bvae.py)",
        "license": "Proprietary NĀḌĪ-Q Research Code",
        "commercial_use_permitted": "Yes",
        "pretrained_weights_available": "Trained in-repo on N=15/36 physical scans",
        "input_schema": "(B, 1, 96, 96, 96) physical 3D T1 MRI",
        "output_schema": "Latent mean mu in R^16, compressed via PLS to 6 components",
        "feature_extraction_supported": "Native (VAE latent encoder mu)",
        "local_inference_feasible": "Yes (Fully local)",
        "memory_ram_required": "< 4 GB RAM",
        "recommendation_status": "ORIGINAL ARCHITECTURAL CORE (Preserved for physical scans)"
    }
]

df = pd.DataFrame(models_data)
out_csv = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_4\MODEL_SELECTION_AND_LICENSE_REGISTER.csv"
df.to_csv(out_csv, index=False)
print(f"Saved {out_csv} ({len(df)} models registered)")


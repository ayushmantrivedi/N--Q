# NĀḌĪ-Q — Open-Source Model Selection & Embedding Report

**Phase**: Phase 5 — Confirmatory Validation & Open-Source Model Integration
**Status**: COMPLETED & VERIFIED

---

## 1. Candidate Model Selection & License Register

| model_name                                     | code_license         | input_type                                                                  |   output_dimensions | training_status                                               |
|:-----------------------------------------------|:---------------------|:----------------------------------------------------------------------------|--------------------:|:--------------------------------------------------------------|
| FT-Transformer (Feature Tokenizer Transformer) | Apache 2.0           | Structured multimodal tabular vector (D=249: FreeSurfer + Motor + Genetics) |                  16 | Locally trained self-supervised/supervised bottleneck encoder |
| TabNet (Attentive Tabular Learning)            | MIT                  | Structured tabular continuous features (D=249)                              |                  16 | Locally trained with sequential sparse attention masks        |
| TabPFN v2 (Prior-data Fitted Networks)         | Apache 2.0           | Small-to-medium tabular classification datasets (N <= 10,000, D <= 500)     |                   1 | Pretrained foundation model prior                             |
| MONAI 3D ResNet-50                             | Apache 2.0           | 3D T1-weighted structural brain MRI volumes (1, 128, 128, 128)              |                 512 | Pretrained on 3D MedicalNet multi-organ CT/MRI                |
| Supervised 3D Beta-VAE + PLS-6 (NĀḌĪ-Q Native) | Academic Open Source | 3D T1 structural brain scans / morphometric IDPs                            |                   6 | Locally trained with disentangled ELBO + PLS supervision      |
| MOMENT-1-small (Multimodal Open Time Series)   | Apache 2.0           | Multi-channel continuous sensor/kinematic time series (L >= 64 timesteps)   |                 512 | Pretrained time-series foundation model                       |

---

## 2. Selected Architecture: FT-Transformer (Feature Tokenizer Transformer)

### A. Theoretical Foundation (Gorishniy et al., NeurIPS 2021)
Standard feed-forward networks struggle on continuous tabular data because coordinate axes represent distinct physical dimensions with varying distributions. FT-Transformer resolves this by transforming each scalar feature $x_i$ into a continuous token vector:

$$\mathbf{e}_i = x_i \mathbf{w}_i + \mathbf{b}_i \in \mathbb{R}^d$$

followed by multi-head self-attention across the feature tokens, capturing higher-order non-linear feature interactions.

### B. Integration Contract with QVBE-v2
1. **Bottleneck Projection**: The `[CLS]` token is projected down to $Z \in \mathbb{R}^{N \times 16}$ ($M=16$ continuous latent embeddings).
2. **Strict Training Partition Freezing**: The FT-Transformer is fitted strictly on $X_{\text{train}}, y_{\text{train}}$ and frozen.
3. **Downstream Quantum Ingestion**: Latent vector $Z$ is directly ingested by `QuantumVariationalBiomarkerEncoderV2(n_qubits=8)` to generate multi-basis quantum observables $Q_Z$.
4. **Zero Pretrained Foundation Model Misrepresentation**: We explicitly document that this is a locally fitted representation encoder; no global foundation weights exist for PPMI FreeSurfer + UPDRS tabular schemas.

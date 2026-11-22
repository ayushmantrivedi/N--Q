# NĀḌĪ-Q — QCDF Architecture Specification

**Component**: Quantum-Classical Diagnostic Fusion (QCDF)
**Phase**: Phase 6 — Multimodal Integration

---

## 1. Dynamic Multimodal Fusion Mechanisms

QCDF supports two complementary fusion modes:

### A. Multimodal Concatenation (Baseline)
$$H_{\text{concat}} = [Q \parallel h_{\text{temporal}} \parallel X_{\text{static}}] \in \mathbb{R}^{288}$$

### B. Adaptive Gated Fusion (Neural QCDF)
Modality projections map heterogeneous inputs into a shared metric subspace $\mathbb{R}^{16}$:

$$h_Q = \text{Proj}_Q(Q), \quad h_T = \text{Proj}_T(h_{\text{temporal}}), \quad h_C = \text{Proj}_C(X_{\text{static}})$$

Dynamic softmax gates $\mathbf{\alpha} = [\alpha_Q, \alpha_T, \alpha_C]$ are calculated conditioned on available representations and quality masks:

$$H_{\text{fused}} = \alpha_Q h_Q + \alpha_T h_T + \alpha_C h_C$$


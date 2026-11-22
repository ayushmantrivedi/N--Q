# NĀḌĪ-Q — CTCLM Architecture Specification & Benchmark Report

**Component**: Classical Temporal-Clinical Learning Module (CTCLM)
**Phase**: Phase 6 — Five-Year Longitudinal Motor Progression

---

## 1. Mathematical Architecture

CTCLM maps variable-length, irregularly sampled clinical motor sequences into a dense continuous representation:

$$h_{\text{temporal}} = \text{GRU}\left( [\mathbf{x}_t \parallel \text{MLP}(\Delta t)]_{t=1}^T \right) \in \mathbb{R}^{16}$$

- **Input Vector ($\mathbf{x}_t \in \mathbb{R}^{35}$)**: MDS-UPDRS Part III total score + 34 motor subscores (speech, rigidity, finger tapping, leg agility, gait, tremor, posture, H&Y).
- **Elapsed Time Encoding ($\Delta t \in \mathbb{R}^1 \to \mathbb{R}^4$)**: 2-layer continuous time-encoding MLP (Linear $\to$ ReLU $\to$ Linear).
- **Sequence Dimension**: $T=3$ visits across early observation window (Baseline Month 0, V02 Month 6, V04 Month 12).

## 2. CTCLM Empirical Benchmark Results (5-Fold Stratified CV)

| model                               |   auroc_mean |   auroc_std |   auprc_mean |   bal_acc_mean |
|:------------------------------------|-------------:|------------:|-------------:|---------------:|
| CTCLM_Latent_LightGBM               |     0.641724 |   0.0438749 |     0.619383 |       0.615202 |
| CTCLM_Native_GRU                    |     0.619322 |   0.0341542 |     0.593966 |       0.583196 |
| Empirical_Trajectory_Stats_LightGBM |     0.648559 |   0.0457153 |     0.63276  |       0.623952 |


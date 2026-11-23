# NĀḌĪ-Q Source-Level Repository Audit Findings

**Audit Date**: November 23, 2026  
**Auditor**: Independent Software Auditor & Quantum ML Research Engineer  
**Inspection Method**: Source code inspection via local shallow clones, AST parsing, and dependency execution in Python 3.12.4.

---

## 1. YAIB (Yet Another ICU Benchmark)

- **Repository**: `https://github.com/rvandewater/YAIB`
- **Inspected Branch**: `development`
- **Commit SHA**: `af9a5b3e135ae11e74cea37940c5c7247e0dfd83`
- **License**: MIT License (declared in `pyproject.toml` and `LICENSE`)
- **Key Modules Audited**:
  - `icu_benchmarks/models/dl_models/grud.py` (Class `GRUD`, function `decay_rate`)
  - `icu_benchmarks/models/dl_models/layers.py`
  - `icu_benchmarks/models/wrappers.py` (Class `DLModelWrapper`)
  - `pyproject.toml`

### Code-Level Findings & Algorithms Implemented

1. **GRU-D Implementation**:
   - In `icu_benchmarks/models/dl_models/grud.py`, YAIB implements a PyTorch translation of Che et al.'s GRU-D (Scientific Reports 2018).
   - The cell accepts an observation mask $m_t \in \{0, 1\}^D$ and explicit inter-observation time intervals $\delta_t \ge 0$.
   - The continuous time decay is parameterized as:
     ```python
     def decay_rate(pre_activation: torch.Tensor) -> torch.Tensor:
         return torch.exp(-torch.relu(pre_activation))
     ```
     where `pre_activation = W_gamma * delta_t + b_gamma`.
   - The input vector is imputed via decay toward the empirical mean:
     $$x_{\text{imputed}} = m_t \odot x_t + (1 - m_t) \odot \left(\gamma_x \odot x_{\text{last}} + (1 - \gamma_x) \odot x_{\text{mean}}\right)$$
   - The previous hidden state is decayed prior to standard GRU gating:
     $$h_{\text{decayed}} = \gamma_h \odot h_{t-1}$$
2. **Framework Entanglement**:
   - The module `grud.py` cannot be imported in isolation due to line 95: `import gin` and class decoration `@gin.configurable`.
   - Outer execution relies on PyTorch Lightning, PyTorch Ignite, Polars, and `uv_build`.
3. **Data Schema Mismatch**:
   - The code expects regular hourly grids where $\delta_t$ counts hours since the last measurement. In PPMI, observations are outpatient visits occurring at $t \in \{0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0\}$ years.
4. **Actionable Extraction**:
   - The mathematical logic of `GRUDCell` is clean and self-contained. It can be adapted in ~150 lines of pure PyTorch, discarding YAIB's outer ICU pipeline.

---

## 2. CEHR-BERT

- **Repository**: `https://github.com/cumc-dbmi/cehrbert`
- **Inspected Branch**: `main`
- **Commit SHA**: `53331604155c77845b55d0bf248bab12e93e5252`
- **License**: MIT License (in `LICENSE`)
- **Key Modules Audited**:
  - `src/cehrbert/models/hf_models/hf_cehrbert.py` (Classes `CehrBertModel`, `CehrBertForPreTraining`)
  - `src/cehrbert/models/hf_models/config.py` (Class `CehrBertConfig`)
  - `src/cehrbert/models/hf_models/tokenization_hf_cehrbert.py`
  - `pyproject.toml`, `deepspeed_configs/`

### Code-Level Findings & Algorithms Implemented

1. **Model Architecture**:
   - CEHR-BERT wraps HuggingFace's `BertEncoder` and `BertSelfAttention` (`transformers.models.bert.modeling_bert`).
   - It represents patient histories as tokenized sequences of medical concept IDs (SNOMED, RxNorm, ICD) mapped to an embedding matrix.
   - It incorporates temporal intervals by adding age embeddings and time-interval embeddings to the input tokens.
2. **Discrepancy with NĀḌĪ-Q Problem Setting**:
   - **Data Domain**: CEHR-BERT requires OMOP CDM databases. PPMI is a clinical trial / observational biomarker dataset formatted as continuous laboratory biofluids (pg/mL), MRI volumetrics ($mm^3$), and motor scores (0–132). PPMI has no OMOP concept vocabulary.
   - **Sample Efficiency**: CEHR-BERT has tens of millions of parameters. Pretraining requires $>100,000$ patient records. On PPMI ($N=341$), training CEHR-BERT would result in immediate severe overfitting and numerical instability.
3. **Dependency Fragility**:
   - Requires `flash-attn-2`, `deepspeed`, `meds`, `femr`, and `dask`. `flash-attn` requires custom CUDA/MSVC compilation on Windows and fails on CPU-only environments.

---

## 3. HGNN (Hypergraph Neural Networks)

- **Repository**: `https://github.com/iMoonLab/HGNN`
- **Inspected Branch**: `master`
- **Commit SHA**: `6212e8a43e75105b7fda7b169fc0063070861bb5`
- **License**: MIT License (in `LICENSE`)
- **Key Modules Audited**:
  - `models/HGNN.py` (Class `HGNN`)
  - `models/HGNN_conv.py` (Class `HGNN_conv`)
  - `utils/hypergraph_utils.py` (Function `_generate_G_from_H`)

### Code-Level Findings & Algorithms Implemented

1. **Algorithm**:
   - Implements the exact 2019 AAAI spectral convolution:
     ```python
     class HGNN_conv(nn.Module):
         def __init__(self, in_ft, out_ft, bias=True):
             super(HGNN_conv, self).__init__()
             self.weight = Parameter(torch.Tensor(in_ft, out_ft))
         def forward(self, x: torch.Tensor, G: torch.Tensor):
             x = x.matmul(self.weight)
             if self.bias is not None:
                 x = x + self.bias
             x = G.matmul(x)
             return x
     ```
   - Matrix $G$ is precomputed as:
     $$G = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}$$
2. **Obsolescence**:
   - The repository is abandoned. The authors explicitly note: *"We have release a deep learning toolbox named DHG for graph neural networks and hypergraph neural networks... This repo is obsolete."*
   - $G$ is a static, square $|V| \times |V|$ clique-expansion matrix. It cannot handle directed hyperedges, edge-level feature vectors, or dynamic temporal updates.

---

## 4. DeepHypergraph (DHG)

- **Repository**: `https://github.com/iMoonLab/DeepHypergraph`
- **Inspected Branch**: `main`
- **Commit SHA**: `d958ab07940465e63ae1df6f784715cab6dc93b5`
- **License**: Apache-2.0 License (in `LICENSE`)
- **Key Modules Audited**:
  - `dhg/structure/hypergraphs/hypergraph.py` (Class `Hypergraph`)
  - `dhg/structure/hypergraphs/directed_hypergraph.py` (0 bytes!)
  - `dhg/models/hypergraphs/hnhn.py` (Class `HNHN`, `HNHNConv`)
  - `dhg/models/hypergraphs/unignn.py` (Class `UniGCN`, `UniGAT`)
  - `pyproject.toml`

### Code-Level Findings & Algorithms Implemented

1. **Environment & Runtime Verification**:
   - We verified that DHG v0.9.7 imports and runs on Windows Python 3.12.4 with PyTorch 2.4.1+cpu without error.
   - It requires only `torch`, `scipy`, `numpy`, and `optuna`. Zero C++ compiled extensions.
2. **Hypergraph Structure**:
   - `dhg.Hypergraph` builds sparse incidence matrices $H \in \mathbb{R}^{|V| \times |E|}$ directly from hyperedge tuples:
     ```python
     hg = dhg.Hypergraph(num_v=6, e_list=[(0, 2), (1, 2), (0, 3), (1, 3), (2, 4), (3, 4), (3, 5), (0, 4)])
     ```
   - Computes degrees, Laplacians, and dual hypergraphs cleanly.
3. **Critical Discovery — Empty Directed Hypergraph**:
   - `dhg/structure/hypergraphs/directed_hypergraph.py` has a file size of **0 bytes**.
   - Although the directory structure implies directed hypergraph support, DHG currently only supports undirected hypergraphs.
4. **Usable Architectures**:
   - `HNHNConv` implements two-stage non-linear hyperedge neuron updates:
     $$x_e = \sigma\left(W_e \cdot \text{Aggr}_{v \in e}(x_v) + b_e\right)$$
     $$x_v = \sigma\left(W_v \cdot \text{Aggr}_{e \ni v}(x_e) + b_v\right)$$
   - This architecture can be directly adapted to model MeTTa-discovered biological relationships.

---

## 5. Hyper-SAGNN

- **Repository**: `https://github.com/ma-compbio/Hyper-SAGNN`
- **Inspected Branch**: `master`
- **Commit SHA**: `69f2fbe21c455aca084497fb2d26a8207a95decd`
- **License**: MIT License (in `LICENSE`)
- **Key Modules Audited**:
  - `Code/main_torch.py`
  - `Code/Modules.py` (Classes `MultiHeadAttention`, `HyperSAGNN`)
  - `Code/random_walk.py`

### Code-Level Findings & Algorithms Implemented

1. **Obsolete Dependencies**:
   - The repository explicitly mandates Python 3.6.8, TensorFlow 1.x (`<2.0.0`), and PyTorch 1.0.
   - It cannot run in our Python 3.12.4 environment without complete environment breakdown.
2. **Methodological Defect for Clinical Prediction**:
   - The core training pipeline (`main_torch.py`) relies on `gensim.models.Word2Vec` to perform random walks on hypergraphs to learn static topological embeddings for discrete node IDs.
   - In NĀḌĪ-Q, nodes are continuous biological variables ($|V|=6$). Random walk discrete tokenization is mathematically inappropriate.

---

## 6. MHNN (Molecular Hypergraph Neural Networks)

- **Repository**: `https://github.com/schwallergroup/mhnn`
- **Inspected Branch**: `main`
- **Commit SHA**: `4a9fc3d760e6357d3b9d41d9a4b545aa6afc1711`
- **License**: MIT License (in `LICENSE.txt`)
- **Key Modules Audited**:
  - `models/mhnn.py` (Class `MHNN`)
  - `models/conv.py` (Class `MHNNConv`)
  - `models/mlp.py` (Class `MLP`)
  - `env.yml`

### Code-Level Findings & Algorithms Implemented

1. **Dual Node/Hyperedge Message Passing**:
   - `models/conv.py` contains the `MHNNConv` layer, which updates node features $X$ and hyperedge features $E$ via alternating MLPs and scatter aggregations:
     ```python
     # Hyperedge update from incident vertices
     e_msg = scatter(X[vertex], edges, dim=0, reduce=self.aggr)
     E = E + self.W1(torch.cat([E, e_msg], dim=-1))
     # Vertex update from incident hyperedges
     v_msg = scatter(E[edges], vertex, dim=0, reduce=self.aggr, dim_size=N)
     X = X + self.W3(torch.cat([X, v_msg], dim=-1))
     ```
2. **Strengths for MeTTa Integration**:
   - Unlike HGNN and DHG (where hyperedges have only scalar weights), MHNN assigns a multidimensional feature vector to each hyperedge $E \in \mathbb{R}^{|E| \times D}$.
   - This directly accommodates MeTTa relationship attributes: correlation coefficient $r$, $p$-value, sample size $n$, and temporal lag $\tau$.
3. **Environment & Domain Barriers**:
   - Hard-coded dependency on `torch_scatter.scatter`, which is notoriously difficult to install on Windows with Python 3.12.
   - Heavily tied to OGB molecular graphs (`AtomEncoder`, bond features).
   - **Conclusion**: The tensor update equations should be adapted into pure PyTorch using native `scatter_add_`, avoiding `torch_scatter`.

---

## 7. GSoC-2024-QGNN

- **Repository**: `https://github.com/Haemanth-V/GSoC-2024-QGNN`
- **Inspected Branch**: `main`
- **Commit SHA**: `c12a9ef65d5801f012e2177e25a4c826ebac4e39`
- **License**: **NO LICENSE FOUND** (Unlicensed public repository)
- **Key Modules Audited**:
  - `code/models/GCNConv_Layers/QGCNConv.py` (Class `QGCNConv`)
  - `code/models/QNN_Node_Embedding.py` (Function `quantum_net`)
  - `code/models/Quantum_GCN.py` (Class `Quantum_GCN`)

### Code-Level Findings & Algorithms Implemented

1. **Absence of Quantum Graph Operations**:
   - Examination of `QGCNConv.py` reveals that the quantum circuit operates solely as a pointwise non-linear node feature preprocessor:
     ```python
     def forward(self, x, edge_index):
         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
         q_out = self.qc(x).float()  # Node-wise VQC! No graph topology used.
         # Standard classical GCN propagation:
         out = self.propagate(edge_index, x=q_out, norm=norm)
         return out + self.bias
     ```
   - The quantum circuit has no knowledge of graph adjacency, edge connections, or hyperedges.
2. **Circuit Design**:
   - `QNN_Node_Embedding.py` implements standard `AngleEmbedding(inputs, rotation="Y")` followed by alternating $RY$, CNOT ring, and $RZ$ rotations, measuring Pauli-Z expectation values.
   - This is identical to standard VQCs that suffer from periodic phase distortion and barren plateaus.
3. **Legal & Practical Barriers**:
   - Unlicensed code cannot be legally adopted into NĀḌĪ-Q.
   - Evaluating a PennyLane QNode sequentially for every node on every batch sample introduces severe simulation bottlenecks on CPU with zero topological benefit.

---

## 8. QuantumGNN Molecules

- **Repository**: `https://github.com/natuan/QuantumGNN_molecules`
- **Inspected Branch**: `master`
- **Commit SHA**: `6f6fd4a2f4192ce497f11abf9fc2292b94afdb9c`
- **License**: Apache-2.0 License (in `LICENSE`)
- **Key Modules Audited**:
  - `code/run_training.py` (Class `QuantumGNN`)
  - `code/preprocess_data.py`

### Code-Level Findings & Algorithms Implemented

1. **Critical Audit Discovery — Zero Quantum Circuits**:
   - The repository claims to implement a "Quantum Graph Neural Network".
   - Code inspection of `code/run_training.py` demonstrates that **it does not contain a single quantum gate, quantum circuit, or quantum library import**:
     ```python
     class QuantumGNN(nn.Module):
         def __init__(self):
             super(QuantumGNN, self).__init__()
             self.embed_atom = nn.Embedding(n_atoms, dim)
             self.W_atom = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_hidden)])
             self.W_property = nn.Linear(2*dim, 1)

         def update(self, xs, V, i, M):
             hs = torch.relu(self.W_atom[i](xs))
             return xs + torch.matmul(V, hs)
     ```
   - The model is a standard classical message-passing network with pairwise inter-atomic distance matrices $V$.
   - The term "Quantum" refers entirely to the quantum-chemical properties of molecules in the QM9 benchmark.
2. **Windows Checkout Failure**:
   - The repository failed to check out normally on Windows due to an invalid path:
     `dataset/QM9_molecularsize<=15/input/Cv(cal|molK).npy` contains `<` and `|`, which are illegal characters on Windows filesystems.

---

## 9. PennyLane

- **Repository**: `https://github.com/PennyLaneAI/pennylane`
- **Local Installed Version**: `0.41.1` (Commit `4b4892f18395e863bcc3a1d67f852b3d5c9e3db3`)
- **License**: Apache-2.0 License
- **Key Modules Audited**:
  - `pennylane/qnode.py`
  - `pennylane/devices/default_qubit.py`
  - `pennylane/qnn/torch.py`
  - `pennylane/kernels/`

### Code-Level Findings & Role in NĀḌĪ-Q

1. **Technical Excellence**:
   - PennyLane is fully stable, mathematically verified, and natively integrated into NĀḌĪ-Q.
2. **Scientific Finding from Local Phase 11–14 Audits**:
   - The failure of quantum machine learning on PPMI progression is **not** caused by software bugs in PennyLane.
   - It is caused by fundamental mathematical constraints:
     - Tabular clinical features lack native quantum structure.
     - Exponential distance concentration (fidelity collapse) makes statevectors near-orthogonal in Hilbert space ($K \to I$).
     - High-frequency phase wrapping $\exp(-i r_{jk} Z_j Z_k)$ scrambles smooth clinical trajectory gradients.
3. **Strategic Placement**:
   - PennyLane is retained in NĀḌĪ-Q strictly as an isolated, standalone research sandbox for synthetic benchmark studies, completely disconnected from the clinical prognostic pipeline.

# NĀḌĪ-Q External Repository Verdict Matrix

**Audit Date**: November 23, 2026  
**Auditor**: Senior Quantum Machine Learning Research Engineer & Biomedical AI Systems Architect  
**Project**: NĀḌĪ-Q Hybrid Quantum-Classical Platform for Parkinson's Disease Progression  
**Dataset**: PPMI Longitudinal Cohort ($N \approx 341$ participants, restricted local storage)  
**Host Environment**: Windows 11 x64, Python 3.12.4, PyTorch 2.4.1+cpu, PennyLane 0.41.1

---

## 1. Master Repository Verdict Table

| Repository | Verified Code | Problem Addressed | Scientific Fit | Environment Compatibility | Integration Complexity | Main Risk | Verdict |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :---: |
| **YAIB**<br>`rvandewater/YAIB` | Yes<br>SHA `af9a5b3` | Problem A:<br>Longitudinal Clinical Prediction | Medium (GRU-D decay math fits irregular visits; ICU pipeline does not) | Low (Requires `gin-config`, `pytorch-ignite`, `lightning`, Python `<3.14`) | High (Full framework); Low (Isolated GRU-D cell) | Heavy framework bloat; high-frequency hourly ICU assumption vs annual PPMI visits | **ADAPT**<br>*(Extract GRU-D cell only)* |
| **CEHR-BERT**<br>`cumc-dbmi/cehrbert` | Yes<br>SHA `5333160` | Problem A:<br>Longitudinal Clinical Prediction | Poor (OMOP concept code vocabulary; NLP tokenization of EHR) | Low (Requires `flash-attn-2`, `transformers`, `deepspeed`, `meds`) | Extreme (Requires OMOP CDM data conversion) | Severe overfitting on $N=341$ cohort; loss of continuous numerical gradients | **REJECT** |
| **HGNN**<br>`iMoonLab/HGNN` | Yes<br>SHA `6212e8a` | Problem B:<br>Hypergraph Representation | Poor (Static, undirected 2-layer hypergraph convolution from 2019) | Medium (Vanilla PyTorch, but legacy codebase) | Low | Abandoned/superseded codebase; zero support for directed or temporal hyperedges | **REJECT** |
| **DeepHypergraph**<br>`iMoonLab/DeepHypergraph` | Yes<br>SHA `d958ab0` | Problem B:<br>Hypergraph Representation | High (Modern incidence matrix $H$, HNHN, UniGNN, HGNN+) | High (Verified working in Python 3.12, pure PyTorch/SciPy, zero C++ extensions) | Low to Medium | `directed_hypergraph.py` is a 0-byte empty placeholder; directionality must be modeled via incidence matrix asymmetry | **ADAPT** |
| **Hyper-SAGNN**<br>`ma-compbio/Hyper-SAGNN` | Yes<br>SHA `69f2fbe` | Problem B:<br>Hypergraph Representation | Very Poor (Word2Vec random walks on massive discrete hypergraphs) | Zero (Requires legacy TensorFlow 1.x, Python 3.6, Gensim Word2Vec) | Extreme | Completely obsolete tech stack; random-walk graph embedding is irrelevant for 6-node multimodal clinical graph | **REJECT** |
| **MHNN**<br>`schwallergroup/mhnn` | Yes<br>SHA `4a9fc3d` | Problem B:<br>Hypergraph Representation | Medium (Dual node/hyperedge message passing; hyperedge features) | Low (Requires `torch_scatter`, `torch_sparse`, RDKit, OGB) | High (Tied to molecular atom/bond representations) | `torch_scatter` compilation failure on Windows Python 3.12; chemical domain coupling | **RESEARCH ONLY**<br>*(Adapt dual-update math)* |
| **GSoC-2024-QGNN**<br>`Haemanth-V/GSoC-2024-QGNN` | Yes<br>SHA `c12a9ef` | Problem C:<br>Quantum Graph Architecture | Very Poor (VQC is merely a node-wise preprocessor; graph message passing is 100% classical PyG GCN) | Low (Requires `torch_geometric`; unlicensed code) | Medium | No quantum graph operations; periodic $2\pi$ phase distortion; severe simulation latency; copyright risk | **REJECT** |
| **QuantumGNN Molecules**<br>`natuan/QuantumGNN_molecules` | Yes<br>SHA `6f6fd4a` | Problem C:<br>Quantum Graph Architecture | Zero (Misleading name: 100% classical PyTorch calculating Hamiltonian distance potentials for QM9) | Zero on Windows without sparse checkout (filename has illegal chars `<` and `\|`) | Zero (Not a quantum model) | Code contains **zero** quantum circuits (no PennyLane, Qiskit, or Cirq); false advertising | **REJECT** |
| **PennyLane**<br>`PennyLaneAI/pennylane` | Yes<br>SHA `4b4892f`<br>(v0.41.1 installed) | Problem C & D:<br>Quantum Simulation & Hybrid Platform | High (Gold-standard differentiable quantum framework; already native to NĀḌĪ-Q) | High (Fully verified in Windows Python 3.12.4) | Low (Existing integration) | Quantum kernel/ansatz failure on clinical progression is an inherent mathematical property, not a library bug | **ADOPT**<br>*(Decoupled research sandbox only)* |

---

## 2. Technical Justification for Each Verdict

### 2.1 YAIB (`rvandewater/YAIB`) — Verdict: ADAPT (Targeted Cell Extraction)
- **Technical Basis**: YAIB implements a standardized ICU benchmark pipeline. Adopting the entire repository is unfeasible due to rigid dependency constraints (`gin-config==0.5.0`, `pytorch-ignite`, `pytorch-lightning`, `uv_build`) and domain mismatch (hourly ICU time-series vs outpatient clinical visits separated by 6 to 12 months).
- **Extracted Component**: `icu_benchmarks/models/dl_models/grud.py` contains an exceptionally well-engineered PyTorch implementation of Che et al.'s **GRU-D** cell. It incorporates:
  $$\gamma_x = \exp\left(-\max(0, W_\gamma \delta_t + b_\gamma)\right), \quad \gamma_h = \exp\left(-\max(0, W_h \delta_t + b_h)\right)$$
  which decays unobserved features toward their empirical mean and hidden states across irregular time intervals $\delta_t$.
- **Action**: Adapt the ~150-line `GRUDCell` into NĀḌĪ-Q's `src/nadiq/models/temporal/` without importing `gin` or YAIB's outer framework, and benchmark it directly against our existing continuous temporal model (CTCLM).

### 2.2 CEHR-BERT (`cumc-dbmi/cehrbert`) — Verdict: REJECT
- **Technical Basis**: CEHR-BERT is a transformer architecture adapted from BERT specifically for structured Electronic Health Records mapped to the OMOP Common Data Model (CDM). It tokenizes discrete concept codes (ICD-10, SNOMED-CT, RxNorm, CPT) and trains with Masked Concept Language Modeling (MLM).
- **Fatal Scientific Mismatch**:
  1. NĀḌĪ-Q operates on continuous numerical biomarkers: CSF $\alpha$-synuclein (pg/mL), p-tau181 (pg/mL), caudate/putamen volumetric MRI ($mm^3$), and continuous UPDRS-III motor scores. It has zero discrete medical concept codes.
  2. CEHR-BERT requires large vocabulary tokenization tables and pretraining across hundreds of thousands of patient histories. On our cohort of $N=341$ participants, a 12-layer BERT transformer would suffer catastrophic parameter over-allocation ($>10^7$ parameters vs 341 samples) and immediate overfitting.
  3. Windows environment conflicts: requires `flash-attn-2` and `deepspeed`, which require complex MSVC C++ compilation.

### 2.3 HGNN (`iMoonLab/HGNN`) — Verdict: REJECT
- **Technical Basis**: The original AAAI 2019 implementation by Feng et al. implements a static spectral hypergraph convolution:
  $$X^{(l+1)} = \sigma\left( D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X^{(l)} \Theta^{(l)} \right)$$
- **Fatal Flaws**:
  1. The repository is officially deprecated. The repository's own README states: *"We have release a deep learning toolbox named DHG for graph neural networks and hypergraph neural networks... This repo is obsolete."*
  2. The implementation only supports static, undirected, unweighted hyperedges. It cannot represent time-lagged MeTTa edges, multi-modal directed dependencies, or edge-level uncertainty weights.

### 2.4 DeepHypergraph (`iMoonLab/DeepHypergraph`) — Verdict: ADAPT
- **Technical Basis**: DeepHypergraph (DHG v0.9.7) is an actively maintained, pure PyTorch toolbox. We experimentally verified that it imports and runs cleanly in our Python 3.12.4 Windows environment with zero compilation required.
- **Valuable Components**:
  - `dhg.Hypergraph`: Robust, sparse-matrix representation of hypergraph structures ($H \in \mathbb{R}^{|V| \times |E|}$), vertex degree $D_v$, hyperedge degree $D_e$, and hypergraph Laplacians.
  - `dhg.models.HNHN` (Hypergraph Networks with Hyperedge Neurons, ICML 2020): Supports distinct non-linear updates for both vertices and hyperedges.
  - `dhg.models.UniGCN` / `UniGNN`: Unified message passing supporting generalized hypergraph reductions.
- **Caveat & Required Modification**:
  - Code inspection revealed that `dhg/structure/hypergraphs/directed_hypergraph.py` is an **empty 0-byte file**. DHG does not natively support directed hyperedges.
  - To support MeTTa's directional relationships (e.g., Molecular $\to$ Structural $\to$ Motor), we must adapt DHG by partitioning the incidence matrix into tail ($H_{\text{src}}$) and head ($H_{\text{dst}}$) incidence matrices or injecting directionality into the message-passing update.

### 2.5 Hyper-SAGNN (`ma-compbio/Hyper-SAGNN`) — Verdict: REJECT
- **Technical Basis**: Published at ICLR 2020 by Zhang et al., Hyper-SAGNN proposes self-attention on hyperedges.
- **Fatal Flaws**:
  1. **Obsolete Dependencies**: Requires Python 3.6.8, TensorFlow 1.x (`<2.0.0`), and PyTorch 1.0. It cannot be installed in Python 3.12 without completely breaking the host environment.
  2. **Methodological Mismatch**: It relies on `gensim.models.Word2Vec` random walks (`Code/random_walk.py`) to generate node context embeddings across large sparse graphs. For NĀḌĪ-Q's 6-node multimodal clinical graph, random-walk sampling is mathematically degenerate and meaningless.

### 2.6 MHNN (`schwallergroup/mhnn`) — Verdict: RESEARCH ONLY (Adapt Equations)
- **Technical Basis**: Published in J. Chem. Phys. 2024 by the Schwaller Group, MHNN models molecular structures with higher-order hyperedges.
- **Strengths**:
  - The `MHNNConv` layer (`models/conv.py`) is mathematically sophisticated: it updates node embeddings $X$ and hyperedge attribute vectors $E$ iteratively using multi-layer perceptrons:
    $$E^{(l+1)} = \text{MLP}_1\left(\left[E^{(l)}, \text{Aggr}_{v \in e}(X^{(l)})\right]\right), \quad X^{(l+1)} = \text{MLP}_2\left(\left[X^{(l)}, \text{Aggr}_{e \ni v}(E^{(l+1)})\right]\right)$$
  - This allows hyperedges to carry explicit biological features (MeTTa confidence scores, temporal lags $\tau$, and uncertainty $\sigma$).
- **Fatal Implementation Barrier**:
  - The code is hard-coded to `torch_scatter.scatter` and molecular chemistry primitives (`ogb.graphproppred.mol_encoder.AtomEncoder`, RDKit). `torch_scatter` fails to build on Windows Python 3.12 without dedicated C++ compilers.
- **Verdict**: Do not install or import MHNN. Adapt its dual-update message-passing equations into native PyTorch using standard `scatter_add_` or index tensors.

### 2.7 GSoC-2024-QGNN (`Haemanth-V/GSoC-2024-QGNN`) — Verdict: REJECT
- **Technical Basis**: A Google Summer of Code 2024 project developed for high-energy physics jet tagging at CERN.
- **Code-Level Reality vs Claims**:
  1. Code inspection of `code/models/GCNConv_Layers/QGCNConv.py` reveals that **the quantum circuit performs zero graph operations**. The class `QGCNConv` simply calls a parameterized Variational Quantum Circuit (VQC) `q_out = self.qc(x)` independently on each node feature vector, and then executes standard classical degree-normalized message passing via PyTorch Geometric: `out = self.propagate(edge_index, x=q_out, norm=norm)`.
  2. The VQC (`code/models/QNN_Node_Embedding.py`) uses basic `AngleEmbedding` with $RY$ rotations and a CNOT ring. As demonstrated in NĀḌĪ-Q Phase 11 through 14A, this encoding strategy suffers from $2\pi$ periodicity, loss of monotonic progression gradients, and barren plateaus.
  3. The repository has **no license** (copyright infringement risk) and requires `torch_geometric`.
  4. Node-by-node sequential QNode execution introduces extreme simulation latency without theoretical or empirical justification.

### 2.8 QuantumGNN Molecules (`natuan/QuantumGNN_molecules`) — Verdict: REJECT
- **Technical Basis**: Purports to be a "Quantum Graph Neural Network for Molecular Property Prediction".
- **Critical Code Discovery**:
  1. Code inspection of `code/run_training.py` reveals that **there is zero quantum computing code in the entire repository**.
  2. It does not import PennyLane, Qiskit, Cirq, or any quantum SDK. It is a 100% classical PyTorch feed-forward network with distance potential matrices ($V_{ij}$) modeling classical atomic interactions on the QM9 dataset. The word "Quantum" in the title refers exclusively to the quantum-chemical properties of molecules being predicted (e.g. HOMO-LUMO gap).
  3. The repository fails to check out cleanly on Windows due to illegal file characters in the dataset directory: `dataset/QM9_molecularsize<=15/input/Cv(cal|molK).npy` contains `<` and `|`.

### 2.9 PennyLane (`PennyLaneAI/pennylane`) — Verdict: ADOPT (Decoupled Research Sandbox Only)
- **Technical Basis**: PennyLane is the industry standard for differentiable quantum programming, fully installed and verified (`v0.41.1`) in our environment.
- **Role in NĀḌĪ-Q**:
  - Must NOT be used in the production clinical progression pipeline (where Phase 14A proved that quantum kernels collapse to AUROC 0.4158 and degrade CTCLM performance by $-0.1104$ AUROC).
  - Adopted strictly as an isolated, standalone research sandbox for studying quantum state fidelity concentration, barren plateaus, and synthetic geometric benchmarks, completely decoupled from patient prognosis.

---

## 3. Summary Assessment

```
                      +-----------------------------+
                      | 9 External Repositories     |
                      +-----------------------------+
                                     |
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
  [ REJECT (5) ]                                        [ EVALUATED (4) ]
  - CEHR-BERT (NLP/OMOP mismatch, overfitting)          - YAIB (ADAPT: GRU-D cell only)
  - HGNN (Deprecated, static only)                      - DeepHypergraph (ADAPT: HNHN/Incidence)
  - Hyper-SAGNN (TF 1.x, Word2Vec random walks)         - MHNN (RESEARCH ONLY: Math equations)
  - GSoC-QGNN (No graph quantum ops, unlicensed)        - PennyLane (ADOPT: Decoupled sandbox)
  - QuantumGNN (100% classical, misleading title)
```

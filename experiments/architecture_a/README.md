# Architecture A — Benchmark Artifacts & Evidence Exports

This directory contains the authoritative, reproducible benchmark results, forensic audit verdicts, and exported biological evidence graphs for **NĀḌĪ-Q Architecture A (Quantum-Assisted Evidence Discovery)**.

---

## 1. Directory Inventory

| File | Format | Description | Key Metric / Verdict |
|:---|:---:|:---|:---|
| `ARCHITECTURE_A_BENCHMARK_RESULTS.csv` | CSV | Tabular summary of the 5 validation benchmarks. | All 5 benchmarks **PASS** |
| `ARCHITECTURE_A_BENCHMARK_VERDICT.json` | JSON | Formal machine-readable benchmark audit verdict with threshold comparisons. | Overall Verdict: **PASS** |
| `ARCHITECTURE_A_VERDICT.json` | JSON | Pipeline execution summary covering both Plane 1 and Plane 2. | Overall Status: **PASS** |
| `CLINICAL_PREDICTION_RESULT.json` | JSON | Longitudinal Task T2 evaluation metrics across random seeds. | Mean AUROC: **0.6366**, AUPRC: **0.5879** |
| `evidence_graph.json` | JSON | Discovered biological hypergraph with nodes, edges, effect sizes, and FDR statuses. | 4 Edges selected via Tabu Search |
| `evidence_graph.metta` | MeTTa | OpenCog Hyperon AtomSpace knowledge base triplets. | Scheme-compatible relational syntax |

---

## 2. Benchmark Summary

Evaluated on the local authoritative PPMI cohort ($N=341$ participants, 159 progression events [46.6%]):

```json
{
  "unit_tests": {
    "status": "PASS",
    "details": "413/413 passed across all suites"
  },
  "ensemble_auroc": {
    "status": "PASS",
    "value": 0.6366,
    "threshold": 0.635,
    "std": 0.005
  },
  "prediction_invariant": {
    "status": "PASS",
    "delta_max": 0.00e+00,
    "quantum_outage_tolerant": true
  },
  "discovery_runtime_ms": {
    "status": "PASS",
    "value": 195.7,
    "threshold_ms": 600.0
  },
  "invariance_stress": {
    "status": "PASS",
    "delta_max": 0.00e+00
  }
}
```

---

## 3. Discovered Biological Network

The discovered graph captures statistically validated multimodal relationships across molecular biofluids, striatal MRI, and clinical motor progression:

```
asyn (CSF α-synuclein) <───> ptau (CSF phospho-Tau181)
      [r = +0.770, p < 0.0001 | Validated (q <= 0.05)]

caudate (Striatal MRI) <───> putamen (Striatal MRI)
      [r = +0.523, p < 0.0001 | Validated (q <= 0.05)]

putamen (Striatal MRI) ─────> motor_total (UPDRS-III Score)
      [r = -0.175, p = 0.0013 | Validated (q <= 0.05)]

motor_total (UPDRS-III) ────> motor_subscore (1-Yr Rate Delta)
      [r = +0.528, p < 0.0001 | Validated (q <= 0.05)]
```

Every discovered relationship is tagged with the standard quality label: **`Provisional/Explanatory`**.

---

## 4. Reproducing the Benchmark

To regenerate and independently verify these results, execute:

```bash
python scripts/run_architecture_a_benchmark.py
```

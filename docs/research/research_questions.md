# Research Questions — Nāḍī-Q

> **Document Status Legend**
>
> - **[A] Established** — Backed by prior literature; used as foundation.
> - **[B] Hypothesis** — A testable scientific claim we intend to investigate.
> - **[C] Experimental Claim** — A claim we expect to evaluate through controlled experiments (results pending).
> - **[D] Future Architecture** — A planned direction not yet designed or tested.

---

## 1. Primary Research Question

**[B] Hypothesis RQ-1:**  
Can hybrid quantum-classical machine learning models — specifically those using parameterized quantum circuits (PQCs) as a feature-embedding layer — demonstrate any measurable advantage over purely classical baselines on structured multimodal clinical biomarker data for early disease-risk detection tasks?

This is the **central open question**. It has not been answered by this project. We do not assume a positive result.

### Sub-questions

**[B] RQ-1a:** Does quantum feature encoding (e.g., angle encoding, amplitude encoding) produce representations that are meaningfully different from classical projections for structured tabular clinical data?

**[B] RQ-1b:** Does the addition of a quantum embedding layer improve or degrade discrimination performance (AUROC, F1) on held-out risk stratification tasks?

**[B] RQ-1c:** Is any observed difference attributable to the quantum circuit structure, or does it vanish under classical dequantization / random Fourier feature baselines?

---

## 2. Secondary Research Questions

**[B] RQ-2: Noise Sensitivity**  
How sensitive are quantum biomarker embeddings to realistic hardware noise (bit-flip, depolarizing, amplitude-damping channels)? Do clinical predictions degrade gracefully or catastrophically under noise?

**[B] RQ-3: Explainability**  
Can gradient-based or circuit-decomposition methods produce clinically interpretable attributions for quantum circuit components? Are quantum attributions as interpretable as SHAP values from a gradient-boosted classical model?

**[B] RQ-4: Modality Interaction**  
Does the quantum fusion layer capture cross-modal interactions (e.g., temporal vital signs × static biomarkers) that a classical attention mechanism does not?

**[B] RQ-5: Data Regime**  
In what data-size regime (N samples, D features), if any, does quantum embedding show behaviour that classical methods cannot replicate without exponential resources? (Theoretical lower bound: unknown; empirical answer: to be investigated.)

---

## 3. Non-Questions (Explicitly Out of Scope)

The following are **not** research questions for this project:

- Whether quantum computing is "better than classical" in general — this is not a well-posed scientific question.
- Whether Nāḍī-Q can be deployed in a clinical setting — no regulatory pathway exists or is planned.
- Whether quantum hardware available today (NISQ era) is mature enough for clinical inference — it almost certainly is not; this is an acknowledged constraint, not an assumption.
- Whether synthetic or simulated data results generalize to real-world EHR distributions — they do not without explicit validation.

---

## 4. Null Hypotheses (to be rigorously tested)

**H₀-1:** The quantum embedding layer produces no statistically significant improvement in AUROC over the best-performing classical baseline trained on the same feature set.

**H₀-2:** The quantum circuit's parameter landscape does not provide additional expressive power beyond a classical MLP of comparable parameter count on structured biomarker data.

**H₀-3:** Noise-aware quantum execution does not alter clinical risk stratification results beyond the uncertainty bands of the classical baseline.

---

## 5. Open Questions (Unresolved at Architecture Stage)

- What is the appropriate quantum circuit depth and qubit count for clinical tabular data dimensions (D = 10–500 features)?
- Which encoding scheme (angle, amplitude, IQP, re-uploading) is most suitable for mixed-type clinical data (continuous + categorical + missing)?
- What constitutes a fair ablation: fully classical vs. quantum-only vs. hybrid?
- How do we control for the risk that quantum circuits simply implement a non-linear kernel that any kernel SVM could replicate?

---

## 6. Success Criteria

A successful research outcome from Nāḍī-Q is **not necessarily** demonstrating quantum advantage. It is:

1. Rigorous, reproducible experimental infrastructure that can answer the above questions.
2. Honest reporting of null results if H₀ is not rejected.
3. Identification of the conditions (if any) under which a quantum component is beneficial.
4. Published code and configurations that allow independent replication.

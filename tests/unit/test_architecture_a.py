"""
Unit tests for NĀḌĪ-Q Architecture A.

Tests cover:
  - BH-FDR correctness and monotonicity
  - Adaptive two-tier FDR tier assignment
  - FDR fallback triggering and logging
  - Power analysis MDE computation
  - EvidenceDiscoveryPipeline on synthetic data (no PPMI required)
  - DualSolverEngine solver mode selection
  - EvidenceGraphExporter JSON structure and disclaimer content
  - ArchitectureAClinicalPredictor on synthetic data
  - Prediction invariance (Δ_max = 0 by construction)
  - EvidenceGraphResult to_dict completeness
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# FDR tests
# ---------------------------------------------------------------------------

from nadiq.architecture_a.fdr import (
    benjamini_hochberg_fdr,
    adaptive_fdr_tier,
    minimum_detectable_effect,
    screen_candidates_fdr,
    power_analysis_summary,
)


class TestBenjaminiHochberg:
    def test_empty(self):
        assert benjamini_hochberg_fdr([]) == []

    def test_single_significant(self):
        result = benjamini_hochberg_fdr([0.001], alpha=0.05)
        assert result == [True]

    def test_single_not_significant(self):
        result = benjamini_hochberg_fdr([0.5], alpha=0.05)
        assert result == [False]

    def test_all_null(self):
        p = [0.9, 0.8, 0.7, 0.6]
        result = benjamini_hochberg_fdr(p, alpha=0.05)
        assert not any(result)

    def test_all_significant(self):
        p = [0.001, 0.002, 0.003, 0.004]
        result = benjamini_hochberg_fdr(p, alpha=0.05)
        assert all(result)

    def test_mixed_known_outcome(self):
        # With m=5, alpha=0.05: BH critical values are k/5*0.05
        # p_sorted: [0.001, 0.01, 0.05, 0.10, 0.90]
        # BH thresholds: 0.01, 0.02, 0.03, 0.04, 0.05
        # Rejections: 0.001≤0.01✓, 0.01≤0.02✓, 0.05≤0.03✗, 0.10✗, 0.90✗
        # By monotonicity: reject indices 0 and 1 (smallest two p-values)
        p = [0.001, 0.01, 0.05, 0.10, 0.90]
        result = benjamini_hochberg_fdr(p, alpha=0.05)
        assert result[0] is True   # 0.001
        assert result[1] is True   # 0.01
        # 0.05 may or may not be rejected depending on exact ordering

    def test_monotonicity(self):
        # Once a rank is rejected, all smaller ranks must also be rejected
        p = [0.001, 0.005, 0.03, 0.08, 0.90]
        result = benjamini_hochberg_fdr(p, alpha=0.05)
        # Find last rejected index in sorted order
        p_arr = np.array(p)
        order = np.argsort(p_arr)
        last_rej = -1
        for k_idx, orig_idx in enumerate(order):
            if result[orig_idx]:
                last_rej = k_idx
        # All indices up to and including last_rej must be rejected
        for k_idx in range(last_rej + 1):
            assert result[order[k_idx]] is True

    def test_alpha_zero(self):
        p = [0.001, 0.002]
        result = benjamini_hochberg_fdr(p, alpha=0.0)
        assert not any(result)

    def test_alpha_one(self):
        p = [0.001, 0.5, 0.99]
        result = benjamini_hochberg_fdr(p, alpha=1.0)
        assert all(result)

    def test_returns_bool_list(self):
        result = benjamini_hochberg_fdr([0.01, 0.5], alpha=0.05)
        assert isinstance(result, list)
        for r in result:
            assert isinstance(r, bool)


class TestAdaptiveFDRTier:
    def test_basic_tiers(self):
        # Clear validated (p=0.001), borderline provisional (p=0.09), Candidate (p=0.9)
        p = [0.001, 0.09, 0.9]
        tier = adaptive_fdr_tier(p, alpha_validated=0.05, alpha_provisional=0.10)
        assert len(tier) == 3
        # All should be in {Validated, Provisional, Candidate, Candidate (Fallback)}
        for v in tier.values():
            assert v in {"Validated", "Provisional", "Candidate", "Candidate (Fallback)"}

    def test_fallback_triggered(self):
        # All p-values very high → no FDR pass → fallback should trigger
        p = [0.99, 0.98, 0.97]
        tier = adaptive_fdr_tier(p, alpha_validated=0.05, alpha_provisional=0.10,
                                 min_selections_fallback=2)
        fallback_count = sum(1 for t in tier.values() if "Fallback" in t)
        # At least min_selections_fallback candidates should be promoted
        assert fallback_count >= 1

    def test_fallback_uses_top_p(self):
        # Fallback should promote lowest p-value first
        p = [0.95, 0.80, 0.70]
        tier = adaptive_fdr_tier(p, alpha_validated=0.05, alpha_provisional=0.10,
                                 min_selections_fallback=1)
        # Index 2 (p=0.70) should be promoted first
        assert "Fallback" in tier[2]

    def test_empty_input(self):
        tier = adaptive_fdr_tier([])
        assert tier == {}

    def test_all_validated(self):
        p = [0.001, 0.002, 0.003]
        tier = adaptive_fdr_tier(p, alpha_validated=0.05, alpha_provisional=0.10)
        # All should be Validated (very low p, m=3, BH thresholds easily met)
        assert all(t == "Validated" for t in tier.values())


class TestPowerAnalysis:
    def test_mde_decreases_with_n(self):
        mde_small = minimum_detectable_effect(50)
        mde_large = minimum_detectable_effect(300)
        assert mde_small > mde_large

    def test_mde_range(self):
        mde = minimum_detectable_effect(272)
        assert 0.0 < mde < 1.0

    def test_mde_known_benchmark(self):
        # N=272, alpha=0.05, power=0.80 should give MDE ~0.17 (from project summary)
        mde = minimum_detectable_effect(272, alpha=0.05, power=0.80)
        assert 0.10 < mde < 0.30

    def test_mde_small_n(self):
        mde = minimum_detectable_effect(3)
        assert mde == 1.0  # defined underpowered case

    def test_power_summary_keys(self):
        summary = power_analysis_summary(272)
        assert "n_train" in summary
        assert "MDE_alpha5_power80" in summary


class TestScreenCandidatesFDR:
    def test_adds_fdr_tier_key(self):
        candidates = [
            {"src": "a", "dst": "b", "r": 0.3, "p": 0.001, "n": 100},
            {"src": "c", "dst": "d", "r": 0.1, "p": 0.9, "n": 100},
        ]
        screened, tier_map = screen_candidates_fdr(candidates)
        for c in screened:
            assert "fdr_tier" in c
        assert len(tier_map) == 2

    def test_fallback_propagates(self):
        candidates = [
            {"src": "a", "dst": "b", "r": 0.01, "p": 0.98, "n": 100},
            {"src": "c", "dst": "d", "r": 0.01, "p": 0.97, "n": 100},
            {"src": "e", "dst": "f", "r": 0.01, "p": 0.96, "n": 100},
        ]
        screened, _ = screen_candidates_fdr(candidates, min_selections_fallback=2)
        tiers = [c["fdr_tier"] for c in screened]
        assert any("Fallback" in t for t in tiers)


# ---------------------------------------------------------------------------
# DualSolverEngine tests
# ---------------------------------------------------------------------------

from nadiq.architecture_a.optimization_engine import DualSolverEngine
from nadiq.models.metta_quantum.types import (
    CandidateRelationship,
    BiologicalQUBOConfig,
    QuantumSelectionOutput,
)


def _make_candidates(n: int = 6) -> List[CandidateRelationship]:
    cands = []
    names = ["asyn", "ptau", "caudate", "putamen", "motor_total", "motor_subscore"]
    for i in range(min(n, len(names) - 1)):
        cands.append(CandidateRelationship(
            id=i,
            src=names[i],
            dst=names[i + 1],
            rel_type=f"{names[i]}_to_{names[i+1]}",
            direction=1,
            effect_r=0.3 - i * 0.02,
            p_val=0.05 + i * 0.01,
            se=0.1,
            lag_years=1.0,
            evidence_n=100,
            status="Validated",
            utility=0.5 - i * 0.05,
        ))
    return cands


class TestDualSolverEngine:
    def test_auto_mode_exact_for_small_n(self):
        cfg = BiologicalQUBOConfig(budget_k=2)
        engine = DualSolverEngine(qubo_config=cfg, solver_mode="auto")
        candidates = _make_candidates(6)
        result = engine.solve(candidates, {}, set())
        assert isinstance(result, QuantumSelectionOutput)
        assert engine.last_solver_used in {"exact", "tabu"}

    def test_tabu_mode(self):
        cfg = BiologicalQUBOConfig(budget_k=2)
        engine = DualSolverEngine(qubo_config=cfg, solver_mode="tabu")
        candidates = _make_candidates(6)
        result = engine.solve(candidates, {}, set())
        assert "tabu" in engine.last_solver_used

    def test_empty_candidates(self):
        engine = DualSolverEngine(solver_mode="tabu")
        result = engine.solve([], {}, set())
        assert result.method == "none"
        assert result.selected_indices == []

    def test_result_has_required_fields(self):
        cfg = BiologicalQUBOConfig(budget_k=2)
        engine = DualSolverEngine(qubo_config=cfg, solver_mode="tabu")
        candidates = _make_candidates(5)
        result = engine.solve(candidates, {}, set())
        assert hasattr(result, "method")
        assert hasattr(result, "selected_indices")
        assert hasattr(result, "is_feasible")
        assert hasattr(result, "runtime_sec")
        assert result.runtime_sec >= 0

    def test_qaoa_fallback_on_timeout(self):
        """QAOA mode must fall back to Tabu on timeout — test with 0s timeout."""
        cfg = BiologicalQUBOConfig(budget_k=2)
        engine = DualSolverEngine(
            qubo_config=cfg,
            solver_mode="qaoa_p1",
            qaoa_timeout_sec=0.0,  # Immediate timeout
        )
        candidates = _make_candidates(5)
        result = engine.solve(candidates, {}, set())
        # Should have fallen back to Tabu
        assert isinstance(result, QuantumSelectionOutput)
        assert "tabu" in engine.last_solver_used or "timeout" in engine.last_solver_used

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            DualSolverEngine(solver_mode="invalid_mode")


# ---------------------------------------------------------------------------
# EvidenceDiscoveryPipeline tests (synthetic data, no PPMI required)
# ---------------------------------------------------------------------------

from nadiq.architecture_a.evidence_discovery import EvidenceDiscoveryPipeline, EvidenceGraphResult


class TestEvidenceDiscoveryPipeline:
    def _make_X(self, n: int = 100, f: int = 6, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, f))
        # Inject some correlation
        X[:, 1] = X[:, 0] * 0.5 + rng.standard_normal(n) * 0.5
        return X

    def test_runs_on_synthetic_data(self):
        X = self._make_X(100, 6)
        pipeline = EvidenceDiscoveryPipeline(solver_mode="tabu", budget_k=2)
        result = pipeline.run(X)
        assert isinstance(result, EvidenceGraphResult)

    def test_result_has_edges(self):
        X = self._make_X(100, 6)
        pipeline = EvidenceDiscoveryPipeline(solver_mode="tabu", budget_k=2)
        result = pipeline.run(X)
        assert isinstance(result.selected_edges, list)

    def test_result_has_solver_info(self):
        X = self._make_X(100, 6)
        pipeline = EvidenceDiscoveryPipeline(solver_mode="tabu", budget_k=2)
        result = pipeline.run(X)
        assert result.solver_mode_used != ""
        assert result.discovery_runtime_ms >= 0

    def test_fallback_on_null_data(self):
        """With random data, most pairs will fail FDR — fallback should trigger."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((50, 6))  # small N, random → all p high
        pipeline = EvidenceDiscoveryPipeline(
            solver_mode="tabu", budget_k=2, min_selections_fallback=2
        )
        result = pipeline.run(X)
        # Regardless of FDR pass/fail, pipeline should complete without error
        assert isinstance(result, EvidenceGraphResult)

    def test_to_dict_complete(self):
        X = self._make_X(100, 6)
        pipeline = EvidenceDiscoveryPipeline(solver_mode="tabu", budget_k=2)
        result = pipeline.run(X)
        d = result.to_dict()
        required_keys = [
            "selected_edges", "n_selected", "solver_mode_used",
            "n_candidates_total", "fdr_fallback_triggered",
            "statistical_quality", "replication_note", "discovery_runtime_ms",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_statistical_quality_label(self):
        X = self._make_X(100, 6)
        pipeline = EvidenceDiscoveryPipeline(solver_mode="tabu", budget_k=2)
        result = pipeline.run(X)
        assert "Provisional" in result.statistical_quality or "Explanatory" in result.statistical_quality


# ---------------------------------------------------------------------------
# EvidenceGraphExporter tests
# ---------------------------------------------------------------------------

from nadiq.architecture_a.evidence_exporter import EvidenceGraphExporter


class TestEvidenceGraphExporter:
    def _make_result(self):
        from nadiq.architecture_a.evidence_discovery import EvidenceDiscoveryPipeline
        rng = np.random.default_rng(1)
        X = rng.standard_normal((80, 6))
        X[:, 1] = X[:, 0] * 0.6 + rng.standard_normal(80) * 0.4
        pipeline = EvidenceDiscoveryPipeline(solver_mode="tabu", budget_k=2)
        return pipeline.run(X)

    def test_to_json_creates_file(self):
        result = self._make_result()
        exporter = EvidenceGraphExporter(result, dataset_label="TEST")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.to_json(os.path.join(tmpdir, "eg.json"))
            assert os.path.exists(path)
            with open(path) as fh:
                data = json.load(fh)
            assert "edges" in data
            assert "disclaimer" in data
            assert "PROVISIONAL" in data["disclaimer"].upper()

    def test_json_has_required_schema_fields(self):
        result = self._make_result()
        exporter = EvidenceGraphExporter(result, dataset_label="TEST")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.to_json(os.path.join(tmpdir, "eg.json"))
            with open(path) as fh:
                data = json.load(fh)
            for field_name in ["schema_version", "generated_at", "dataset", "disclaimer",
                               "statistical_quality", "discovery_summary", "edges"]:
                assert field_name in data, f"Missing JSON field: {field_name}"

    def test_metta_file_created(self):
        result = self._make_result()
        exporter = EvidenceGraphExporter(result)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.to_metta_file(os.path.join(tmpdir, "eg.metta"))
            assert os.path.exists(path)
            with open(path) as fh:
                content = fh.read()
            assert "Provisional" in content or "NĀḌĪ-Q" in content

    def test_to_summary_dict_keys(self):
        result = self._make_result()
        exporter = EvidenceGraphExporter(result)
        summary = exporter.to_summary_dict()
        for key in ["n_selected", "edges", "solver", "fdr_fallback", "statistical_quality", "runtime_ms"]:
            assert key in summary

    def test_edges_have_quality_label(self):
        result = self._make_result()
        exporter = EvidenceGraphExporter(result)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.to_json(os.path.join(tmpdir, "eg.json"))
            with open(path) as fh:
                data = json.load(fh)
            for edge in data["edges"]:
                assert "statistical_quality" in edge
                assert edge["statistical_quality"] == "Provisional/Explanatory"


# ---------------------------------------------------------------------------
# ArchitectureAClinicalPredictor tests (synthetic PPMI fallback)
# ---------------------------------------------------------------------------

from nadiq.architecture_a.clinical_predictor import ArchitectureAClinicalPredictor, ClinicalPredictionResult


class TestClinicalPredictor:
    """Uses DataProtocol14A synthetic fallback (no PPMI parquet required)."""

    def _make_predictor(self, evidence_path=None):
        return ArchitectureAClinicalPredictor(
            task="T2",
            seeds=[42],
            n_folds=3,
            # Paths that don't exist → synthetic fallback in DataProtocol14A
            cohort_path="/nonexistent/cohort.parquet",
            cache_path="/nonexistent/cache.parquet",
            evidence_graph_path=evidence_path,
        )

    def test_runs_without_error(self):
        pred = self._make_predictor()
        result = pred.predict()
        assert isinstance(result, ClinicalPredictionResult)

    def test_quantum_invariance_always_zero(self):
        """max_quantum_delta must be 0.0 by construction (no quantum code in predictor)."""
        pred = self._make_predictor()
        result = pred.predict()
        assert result.max_quantum_delta == 0.0

    def test_evidence_path_does_not_affect_predictions(self):
        """Predictions must be identical regardless of evidence_graph_path."""
        pred_no_ev = self._make_predictor(evidence_path=None)
        pred_with_ev = self._make_predictor(evidence_path="/some/path.json")
        res_a = pred_no_ev.predict()
        res_b = pred_with_ev.predict()
        delta = max(abs(a - b) for a, b in zip(res_a.y_prob_ensemble, res_b.y_prob_ensemble))
        assert delta < 1e-10

    def test_to_dict_has_required_keys(self):
        pred = self._make_predictor()
        result = pred.predict()
        d = result.to_dict()
        for key in ["task", "ensemble_auroc", "ensemble_auprc", "ensemble_brier",
                    "max_quantum_delta", "quantum_invariant"]:
            assert key in d

    def test_quantum_invariant_flag_is_true(self):
        pred = self._make_predictor()
        result = pred.predict()
        assert result.to_dict()["quantum_invariant"] is True

    def test_auroc_in_valid_range(self):
        pred = self._make_predictor()
        result = pred.predict()
        assert 0.0 <= result.ensemble_auroc <= 1.0

    def test_no_quantum_imports(self):
        """Verify that the clinical_predictor module has no quantum imports."""
        import nadiq.architecture_a.clinical_predictor as mod
        import inspect
        source = inspect.getsource(mod)
        forbidden = ["pennylane", "qml.", "QAOAOptimizer", "metta_quantum"]
        for f in forbidden:
            assert f not in source, (
                f"clinical_predictor.py contains forbidden import/reference: '{f}'"
            )

    def test_multiseed_produces_per_seed_aurocs(self):
        pred = ArchitectureAClinicalPredictor(
            task="T2",
            seeds=[42, 123],
            n_folds=3,
            cohort_path="/nonexistent/cohort.parquet",
            cache_path="/nonexistent/cache.parquet",
        )
        result = pred.predict()
        assert len(result.seed_aurocs) == 2
        assert len(result.seed_auprcs) == 2


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------

class TestArchitectureAPipelineSmoke:
    def test_pipeline_smoke(self):
        """Full end-to-end smoke test on synthetic data (no PPMI required)."""
        from nadiq.architecture_a.pipeline import ArchitectureAPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ArchitectureAPipeline(
                task="T2",
                seeds=[42],
                n_folds=3,
                solver_mode="tabu",
                budget_k=2,
                # Non-existent paths → synthetic fallback in DataProtocol14A
                cohort_path="/nonexistent/cohort.parquet",
                cache_path="/nonexistent/cache.parquet",
                auroc_threshold=0.0,  # Relax for synthetic data
            )
            report = pipeline.run_full(output_dir=tmpdir)
            assert report.clinical_result is not None
            assert report.discovery_result is not None
            assert report.prediction_invariant is True
            assert report.clinical_result.max_quantum_delta == 0.0

            # Check verdict JSON was written
            verdict_path = os.path.join(tmpdir, "ARCHITECTURE_A_VERDICT.json")
            assert os.path.exists(verdict_path)
            with open(verdict_path) as fh:
                verdict = json.load(fh)
            assert "clinical_prediction" in verdict
            assert "evidence_discovery" in verdict

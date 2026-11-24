"""
Comprehensive unit, thread-safety, and end-to-end Streamlit AppTest integration tests for app.py.
Tests every preset, all 341 real PPMI participants, manual inputs, sliders, solvers, and concurrent threads.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from app import (
    PPMI_REF_STATS,
    FEATURE_META,
    build_biomarker_deviation_chart,
    build_evidence_network_figure,
    get_clinical_interpretation,
    get_model,
    get_evidence_graph_edges,
    load_ppmi_cohort,
    normalize_val,
    run_patient_inference,
)


def test_ppmi_ref_stats():
    for key in ["asyn", "ptau", "caudate", "putamen", "motor_total", "motor_subscore"]:
        assert key in PPMI_REF_STATS
        assert key in FEATURE_META
        assert PPMI_REF_STATS[key]["std"] > 0


def test_normalize_val():
    z = normalize_val(PPMI_REF_STATS["asyn"]["mean"], "asyn")
    assert abs(z) < 1e-6


def test_get_model():
    model = get_model()
    assert model is not None
    assert hasattr(model, "qvbe")
    assert hasattr(model, "ctclm")
    assert hasattr(model, "qcdf")


def test_get_evidence_graph_edges():
    edges = get_evidence_graph_edges()
    assert isinstance(edges, list)
    assert len(edges) >= 3
    for e in edges:
        assert "source" in e
        assert "target" in e
        assert "weight" in e


def test_build_biomarker_deviation_chart():
    z_scores = {k: 0.5 for k in PPMI_REF_STATS}
    fig = build_biomarker_deviation_chart(z_scores)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_build_evidence_network_figure():
    sample_edges = [
        {"src": "asyn", "dst": "ptau", "effect_r": 0.77, "p_val": 1e-6},
        {"src": "putamen", "dst": "motor_total", "effect_r": -0.18, "p_val": 0.0013},
    ]
    fig = build_evidence_network_figure(sample_edges)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 6


def test_get_clinical_interpretation():
    desc = get_clinical_interpretation("putamen", "motor_total", -0.175)
    assert isinstance(desc, str)
    assert len(desc) > 10


def test_all_341_ppmi_participants_within_ui_bounds():
    """Verify every single PPMI participant fits inside the UI number_input bounds."""
    X_raw, y, pats = load_ppmi_cohort()
    assert X_raw is not None
    assert len(pats) == 341

    bounds = [
        (100.0, 6000.0),   # asyn
        (1.0, 80.0),       # ptau
        (0.0, 5.0),        # caudate SBR
        (0.0, 5.0),        # putamen SBR
        (0.0, 108.0),      # motor_total
        (-40.0, 50.0),     # motor_subscore
    ]
    for col_idx, (lo, hi) in enumerate(bounds):
        col = X_raw[:, col_idx]
        valid = col[~np.isnan(col)]
        assert np.all(valid >= lo), f"col {col_idx} has value {valid.min()} < {lo}"
        assert np.all(valid <= hi), f"col {col_idx} has value {valid.max()} > {hi}"


def test_concurrent_quantum_inference_thread_safety():
    """Verify concurrent threads calling PennyLane inference never trigger QueuingManager pop errors."""
    inputs = [
        (1380.0 + i * 10, 15.5, 0.72, 0.58, 26.0, 3.5)
        for i in range(6)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda args: run_patient_inference(*args), inputs))
    assert len(results) == 6
    for res in results:
        assert 0.0 <= res["prob"] <= 1.0


def test_apptest_initial_render_presets_and_manual_inputs():
    """End-to-end Streamlit AppTest: render initial page, switch presets, edit inputs, and test PPMI patients."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Initial render failed: {at.exception}"
    assert len(at.number_input) == 6

    # Edit number_input manually
    at.number_input[0].set_value(950.0).run()
    assert not at.exception, f"Manual number_input failed: {at.exception}"

    presets = [
        "Demo Patient B (High-Risk Rapid Progressor)",
        "Demo Patient C (Low-Risk Stable Profile)",
        "Custom Clinical Entry",
        "Real PPMI Study Participant (N=341)",
    ]
    for p in presets:
        at.selectbox[0].set_value(p).run()
        assert not at.exception, f"Preset '{p}' failed: {at.exception}"

    # Verify selecting different real PPMI participants
    pat_options = at.selectbox[1].options
    assert len(pat_options) == 341
    for idx in [0, 10, 100, 250, 340]:
        at.selectbox[1].set_value(pat_options[idx]).run()
        assert not at.exception, f"PPMI participant idx {idx} ({pat_options[idx]}) failed: {at.exception}"


def test_apptest_discovery_tab_sliders_and_solvers():
    """End-to-end Streamlit AppTest: test discovery sliders and Tabu/Exact solvers."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception

    # Adjust budget slider and FDR slider, then click Run Network Discovery
    at.slider[0].set_value(2).run()
    assert not at.exception
    at.select_slider[0].set_value(0.01).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception, f"Tabu discovery (k=2, alpha=0.01) failed: {at.exception}"
    assert len(at.success) >= 1

    # Switch to Exact Global Search, k=6, alpha=0.10 and run
    at.selectbox[1].set_value("Exact Global Search").run()
    assert not at.exception
    at.slider[0].set_value(6).run()
    assert not at.exception
    at.select_slider[0].set_value(0.10).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception, f"Exact discovery (k=6, alpha=0.10) failed: {at.exception}"

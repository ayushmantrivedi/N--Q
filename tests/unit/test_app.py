"""
Comprehensive unit and end-to-end Streamlit AppTest integration tests for app.py.
Tests every preset, all 341 real PPMI participants, all tabs, and all discovery solvers.
"""

from __future__ import annotations

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


def test_apptest_initial_render_and_all_presets():
    """End-to-end Streamlit AppTest: render initial page and switch through every preset."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Initial render failed: {at.exception}"
    assert len(at.number_input) == 6

    presets = [
        "Demo Patient B (High-Risk Rapid Progressor)",
        "Demo Patient C (Low-Risk Stable Profile)",
        "Custom Clinical Entry",
        "Real PPMI Study Participant (N=341)",
    ]
    for p in presets:
        at.selectbox[0].set_value(p).run()
        assert not at.exception, f"Preset '{p}' failed: {at.exception}"

    # Verify selecting different real PPMI participants (including index 0, 10, 100, 340)
    pat_options = at.selectbox[1].options
    assert len(pat_options) == 341
    for idx in [0, 10, 100, 340]:
        at.selectbox[1].set_value(pat_options[idx]).run()
        assert not at.exception, f"PPMI participant idx {idx} ({pat_options[idx]}) failed: {at.exception}"


def test_apptest_discovery_tab_solvers():
    """End-to-end Streamlit AppTest: run network discovery with Tabu and Exact solvers."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception

    # Click Run Network Discovery with default Classical Tabu Search
    at.button[0].click().run()
    assert not at.exception, f"Tabu discovery failed: {at.exception}"
    assert len(at.success) >= 1

    # Switch to Exact Global Search and run
    at.selectbox[1].set_value("Exact Global Search").run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception, f"Exact discovery failed: {at.exception}"

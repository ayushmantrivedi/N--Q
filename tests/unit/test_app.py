"""
Tests for Streamlit application components, Plotly graph builders, and utility functions.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from app import (
    PPMI_REF_STATS,
    FEATURE_META,
    build_biomarker_deviation_chart,
    build_evidence_network_figure,
    get_clinical_interpretation,
    get_model,
    get_evidence_graph_edges,
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
    assert len(fig.data) >= 6  # edges + labels + 6 nodes


def test_get_clinical_interpretation():
    desc = get_clinical_interpretation("putamen", "motor_total", -0.175)
    assert isinstance(desc, str)
    assert len(desc) > 10

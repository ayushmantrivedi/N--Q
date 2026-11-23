"""
Tests for Streamlit application components and utility functions.
"""

from __future__ import annotations

import os
import pytest
import torch

from app import (
    PPMI_REF_STATS,
    get_model,
    get_evidence_graph_edges,
    normalize_val,
)


def test_ppmi_ref_stats():
    for key in ["asyn", "ptau", "caudate", "putamen", "motor_total", "motor_subscore"]:
        assert key in PPMI_REF_STATS
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

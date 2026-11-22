"""
NĀḌĪ-Q Fusion Package — Quantum-Classical Diagnostic Fusion (QCDF).
"""

from nadiq.models.fusion.qcdf import (
    QCDF,
    QCDFConfig,
    QCDFOutput,
)

# Canonical alias
QuantumClinicalDecisionFusion = QCDF

__all__ = [
    "QCDF",
    "QuantumClinicalDecisionFusion",
    "QCDFConfig",
    "QCDFOutput",
]

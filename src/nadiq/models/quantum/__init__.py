"""
NĀḌĪ-Q Quantum Models Package.

Includes:
- QVBE: Quantum Variational Biomarker Encoder (V1)
- QVBE-v2: Quantum Variational Biomarker Encoder (V2)
- CTCLM: Continual Temporal-Clinical Learning Module
- QCDF: Quantum-Classical Diagnostic Fusion (re-exported from fusion)
"""

from nadiq.models.quantum.qvbe import (
    QVBEConfig,
    QVBEOutput,
    QuantumVariationalBiomarkerEncoder,
)
from nadiq.models.quantum.qvbe_v2 import (
    QVBEV2Config,
    QuantumVariationalBiomarkerEncoderV2,
)
from nadiq.models.quantum.ctclm import (
    CTCLMConfig,
    CTCLMOutput,
    ContinualTemporalClinicalLatentModel,
)
from nadiq.models.fusion.qcdf import (
    QCDF,
    QCDFConfig,
    QCDFOutput,
)
from nadiq.models.quantum.brqk import (
    BRQKConfig,
    BiologicalRelationalQuantumKernel,
)

__all__ = [
    "QVBEConfig",
    "QVBEOutput",
    "QuantumVariationalBiomarkerEncoder",
    "QVBEV2Config",
    "QuantumVariationalBiomarkerEncoderV2",
    "CTCLMConfig",
    "CTCLMOutput",
    "ContinualTemporalClinicalLatentModel",
    "QCDF",
    "QCDFConfig",
    "QCDFOutput",
    "BRQKConfig",
    "BiologicalRelationalQuantumKernel",
]

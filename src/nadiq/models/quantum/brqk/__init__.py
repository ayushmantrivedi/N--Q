"""
NĀḌĪ-Q Phase 14A: Biological Relational Quantum Kernel (BRQK) Subpackage.
"""

from nadiq.models.quantum.brqk.config import BRQKConfig
from nadiq.models.quantum.brqk.feature_map import (
    FeatureStandardizer,
    compute_angles,
    compute_interaction_terms,
    biological_feature_map,
    adjoint_biological_feature_map,
)
from nadiq.models.quantum.brqk.quantum_kernel import (
    BiologicalRelationalQuantumKernel,
)
from nadiq.models.quantum.brqk.kernel_validation import (
    validate_kernel_matrix,
    regularize_kernel_matrix,
)
from nadiq.models.quantum.brqk.shot_estimator import (
    FiniteShotKernelEstimator,
)

__all__ = [
    "BRQKConfig",
    "FeatureStandardizer",
    "compute_angles",
    "compute_interaction_terms",
    "biological_feature_map",
    "adjoint_biological_feature_map",
    "BiologicalRelationalQuantumKernel",
    "validate_kernel_matrix",
    "regularize_kernel_matrix",
    "FiniteShotKernelEstimator",
]

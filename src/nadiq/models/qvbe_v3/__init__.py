"""
NĀḌĪ-Q QVBE-v3 Module.
Quantum Variational Biomarker Encoder v3 (Remediated Architecture).
"""

from nadiq.models.qvbe_v3.config import QVBEv3Config
from nadiq.models.qvbe_v3.circuit import QVBEv3Circuit
from nadiq.models.qvbe_v3.encoding import DataReuploadingEncoder, TrainableInputAdapter
from nadiq.models.qvbe_v3.observables import compute_connected_correlators
from nadiq.models.qvbe_v3.objectives import infonce_loss, masked_reconstruction_loss
from nadiq.models.qvbe_v3.trainer import QVBEv3Trainer
from nadiq.models.qvbe_v3.diagnostics import effective_rank, seed_variance_vs_patient_variance

__all__ = [
    "QVBEv3Config",
    "QVBEv3Circuit",
    "DataReuploadingEncoder",
    "TrainableInputAdapter",
    "compute_connected_correlators",
    "infonce_loss",
    "masked_reconstruction_loss",
    "QVBEv3Trainer",
    "effective_rank",
    "seed_variance_vs_patient_variance",
]

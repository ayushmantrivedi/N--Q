"""
Abstract quantum backend interface for Nāḍī-Q.

All quantum execution in this platform goes through this interface.
This guarantees that:
  - Circuit definitions are backend-agnostic.
  - Backend selection is purely configuration-driven.
  - Switching from simulator to real hardware requires only a config change.

Implemented backends (future, in nadiq/quantum/backends/):
  - PennyLaneLightningBackend   — Fast CPU simulation (default)
  - QiskitAerBackend            — Noise-realistic simulation
  - IBMQuantumBackend           — Real hardware (future, requires credentials)

See: docs/quantum_design.md §4 Backend Abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import numpy as np

__all__ = [
    "BackendType",
    "DeviceInfo",
    "ExecutionConfig",
    "MeasurementResult",
    "QuantumBackend",
]


class BackendType(Enum):
    """Supported quantum backend types."""

    PENNYLANE_LIGHTNING = "pennylane_lightning"
    QISKIT_AER = "qiskit_aer"
    IBM_QUANTUM = "ibm_quantum"  # Requires credentials; [D] future


@dataclass
class DeviceInfo:
    """
    Properties of a quantum device or simulator.

    For simulators, these represent the simulated device parameters.
    For real hardware, these are pulled from device calibration data.

    Attributes:
        backend_type:    Which backend this device is.
        n_qubits:        Number of available qubits.
        basis_gates:     The native gate set supported.
        connectivity:    Qubit connectivity map (adjacency list).
                         None implies all-to-all (typical for simulators).
        t1_times_us:     T1 relaxation times in microseconds (per qubit).
                         None for simulators without noise modeling.
        t2_times_us:     T2 dephasing times in microseconds (per qubit).
                         None for simulators without noise modeling.
        gate_fidelities: Dict mapping gate name → average fidelity.
                         None for ideal simulators.
    """

    backend_type: BackendType
    n_qubits: int
    basis_gates: list[str]
    connectivity: Optional[dict[int, list[int]]] = None
    t1_times_us: Optional[list[float]] = None
    t2_times_us: Optional[list[float]] = None
    gate_fidelities: Optional[dict[str, float]] = None

    @property
    def is_ideal(self) -> bool:
        """True if this device has no noise model (ideal simulation)."""
        return self.t1_times_us is None and self.gate_fidelities is None


@dataclass
class ExecutionConfig:
    """
    Parameters controlling a single circuit execution.

    Attributes:
        shots:        Number of measurement shots. Use None for exact statevector.
        seed:         Random seed for shot sampling. Must be explicitly set.
        noise_model:  Identifier for a noise model config. None = ideal simulation.
        optimization_level: Circuit compilation optimization (0–3).
    """

    shots: Optional[int] = 1024
    seed: Optional[int] = None
    noise_model: Optional[str] = None
    optimization_level: int = 1


@dataclass
class MeasurementResult:
    """
    Output of a quantum circuit execution.

    Attributes:
        expectation_values: Dict mapping observable name → expectation value.
                            Shape: scalar or array depending on observable.
        counts:             Raw measurement counts (bitstring → count).
                            None if the backend returns expectation values directly.
        shots_used:         Actual number of shots executed.
        execution_time_ms:  Wall-clock execution time in milliseconds.
        metadata:           Backend-specific additional information.
    """

    expectation_values: dict[str, float | np.ndarray]
    counts: Optional[dict[str, int]] = None
    shots_used: Optional[int] = None
    execution_time_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class QuantumBackend(ABC):
    """
    Abstract interface for all quantum execution backends.

    Concrete implementations must be placed in nadiq/quantum/backends/.
    This abstract class must remain free of framework-specific imports.

    The parameter-shift gradient method is the expected training mechanism.
    Backends that do not support gradients (e.g., real hardware) may be
    used for inference and noise benchmarking only.
    """

    @abstractmethod
    def execute(
        self,
        circuit_fn: Any,
        parameters: np.ndarray,
        input_data: np.ndarray,
        exec_config: ExecutionConfig,
    ) -> MeasurementResult:
        """
        Execute a parameterized quantum circuit.

        Args:
            circuit_fn:   A callable that defines the quantum circuit.
                          Its signature and conventions are backend-specific
                          but must be defined by the concrete backend.
            parameters:   Trainable circuit parameters (θ), shape (n_params,).
            input_data:   Input features to encode (x), shape (n_features,).
            exec_config:  Execution configuration (shots, seed, noise).

        Returns:
            MeasurementResult containing expectation values and metadata.
        """

    @abstractmethod
    def get_device_info(self) -> DeviceInfo:
        """
        Return the device properties of this backend.

        Must be callable without executing any circuit.
        """

    @property
    @abstractmethod
    def supports_gradients(self) -> bool:
        """
        Whether this backend supports automatic gradient computation.

        If True, parameter-shift gradients can be computed natively.
        If False, the backend can only be used for forward-pass inference.
        """

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the BackendType enum value for this backend."""

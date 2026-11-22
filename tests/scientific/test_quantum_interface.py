"""
Scientific tests for quantum backend interface correctness.

These tests verify mathematical properties that must hold regardless of
which specific quantum framework is used. They test the CONTRACT,
not the implementation.

Tests in this file are marked @pytest.mark.scientific.
Some are marked @pytest.mark.requires_quantum and are skipped if
PennyLane is not installed.

IMPORTANT: These tests verify INTERFACE correctness and basic
mathematical properties. They do NOT demonstrate quantum advantage
or validate any clinical hypothesis.
"""

import pytest
import numpy as np

from nadiq.quantum.backends.base import (
    BackendType,
    DeviceInfo,
    ExecutionConfig,
    MeasurementResult,
)


class TestDeviceInfo:
    """Tests for the DeviceInfo dataclass contract."""

    @pytest.mark.scientific
    def test_ideal_device_has_no_noise(self) -> None:
        """An ideal simulator should report is_ideal = True."""
        device = DeviceInfo(
            backend_type=BackendType.PENNYLANE_LIGHTNING,
            n_qubits=8,
            basis_gates=["rx", "ry", "rz", "cnot"],
        )
        assert device.is_ideal is True

    @pytest.mark.scientific
    def test_noisy_device_is_not_ideal(self) -> None:
        """A device with T1 times should report is_ideal = False."""
        device = DeviceInfo(
            backend_type=BackendType.QISKIT_AER,
            n_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            t1_times_us=[100.0] * 5,
            t2_times_us=[80.0] * 5,
        )
        assert device.is_ideal is False


class TestExecutionConfig:
    """Tests for ExecutionConfig defaults and constraints."""

    @pytest.mark.scientific
    def test_default_shots(self) -> None:
        config = ExecutionConfig()
        assert config.shots == 1024

    @pytest.mark.scientific
    def test_statevector_mode_is_none_shots(self) -> None:
        """None shots signals exact statevector computation."""
        config = ExecutionConfig(shots=None)
        assert config.shots is None


class TestMeasurementResult:
    """Tests for MeasurementResult structure."""

    @pytest.mark.scientific
    def test_expectation_values_in_range(self) -> None:
        """
        Pauli Z expectation values must be in [-1, 1].
        This is a fundamental quantum mechanical constraint.
        """
        result = MeasurementResult(
            expectation_values={"Z0": 0.5, "Z1": -0.3, "Z2": 1.0},
            shots_used=1024,
        )
        for name, val in result.expectation_values.items():
            if isinstance(val, float):
                assert -1.0 <= val <= 1.0, (
                    f"Expectation value {name}={val} is outside [-1, 1]. "
                    "This violates quantum mechanical constraints."
                )

    @pytest.mark.scientific
    def test_result_has_metadata(self) -> None:
        result = MeasurementResult(
            expectation_values={"Z0": 0.0},
        )
        assert isinstance(result.metadata, dict)


class TestConfigHashing:
    """Tests for deterministic configuration hashing."""

    @pytest.mark.unit
    def test_identical_configs_same_hash(self) -> None:
        from nadiq.utils.config import hash_config

        cfg1 = {"seed": 42, "model": {"name": "logreg", "C": 1.0}}
        cfg2 = {"seed": 42, "model": {"name": "logreg", "C": 1.0}}

        assert hash_config(cfg1) == hash_config(cfg2)

    @pytest.mark.unit
    def test_different_configs_different_hash(self) -> None:
        from nadiq.utils.config import hash_config

        cfg1 = {"seed": 42}
        cfg2 = {"seed": 99}

        assert hash_config(cfg1) != hash_config(cfg2)

    @pytest.mark.unit
    def test_hash_is_hex_string(self) -> None:
        from nadiq.utils.config import hash_config

        h = hash_config({"seed": 42})
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest
        int(h, 16)  # Must be valid hex

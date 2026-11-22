"""
Quantum backend abstraction for Nāḍī-Q.

Defines the abstract interface that all quantum backends must implement.
Circuit code is written against this interface, not against specific
framework APIs (PennyLane, Qiskit, etc.).

See: docs/quantum_design.md for design rationale.
"""

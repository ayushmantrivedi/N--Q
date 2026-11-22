"""
Data access layer for Nāḍī-Q.

This subpackage defines the data contract that all dataset loaders must implement.
No raw clinical data is stored in this package.

See: docs/data_contract.md for the full specification.

Modules:
    contracts   — Abstract base classes and data structures
    loaders     — Concrete dataset loader implementations
    registry    — Dataset registry (config-driven loader resolution)
    splits      — Split management and cross-validation utilities
"""

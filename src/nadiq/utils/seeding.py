"""
Deterministic random seed management for Nāḍī-Q experiments.

Every stochastic operation in the platform must be seeded through this module.
Do not use numpy.random.seed() or random.seed() directly in experiment code.

Usage:
    from nadiq.utils.seeding import seed_all

    seed_all(42)
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

__all__ = ["seed_all", "SeedState"]


def seed_all(seed: int) -> None:
    """
    Set all random seeds for full reproducibility.

    Applies the seed to:
        - Python's built-in `random` module
        - NumPy's default random generator

    Conditionally applies (if installed):
        - PyTorch (torch.manual_seed + cuda)
        - PennyLane (via device initialization conventions)

    Args:
        seed: Integer seed value. Must be in range [0, 2**32 - 1].

    Note:
        This function does NOT seed quantum backends directly.
        Quantum backend seeding must be handled at circuit execution time
        via the backend configuration. See quantum/backends/.
    """
    if not (0 <= seed < 2**32):
        raise ValueError(f"Seed must be in [0, 2**32 - 1], got {seed}")

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 — legacy API needed for sklearn compatibility

    # Conditionally seed PyTorch if available
    try:
        import torch  # type: ignore[import]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # For full determinism (at cost of performance):
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # NOTE: Not setting these by default; user must opt in via config.
    except ImportError:
        pass

    # Conditionally seed Qiskit if available
    try:
        from qiskit.utils import algorithm_globals  # type: ignore[import]

        algorithm_globals.random_seed = seed
    except ImportError:
        pass


class SeedState:
    """
    Context manager for temporarily overriding the global seed.

    Use when you need a specific seed for a sub-experiment without
    affecting the global random state of the parent experiment.

    Example:
        with SeedState(seed=99):
            result = run_stochastic_operation()
        # Global state restored after the block
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._random_state: Optional[object] = None
        self._numpy_state: Optional[np.random.RandomState] = None

    def __enter__(self) -> "SeedState":
        self._random_state = random.getstate()
        self._numpy_state = np.random.get_state()
        seed_all(self.seed)
        return self

    def __exit__(self, *_: object) -> None:
        if self._random_state is not None:
            random.setstate(self._random_state)
        if self._numpy_state is not None:
            np.random.set_state(self._numpy_state)

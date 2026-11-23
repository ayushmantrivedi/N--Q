"""
NĀḌĪ-Q Phase 14A: Quantum Kernel Validation & Mathematical Verification.
Verifies Gram matrix properties: symmetry, unit diagonal, bounds in [0, 1],
and positive semi-definiteness via eigenvalue spectrum analysis.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict


def validate_kernel_matrix(K: np.ndarray, tol: float = 1e-5) -> Dict[str, Any]:
    """
    Validate mathematical and structural integrity of a kernel Gram matrix K.

    Checks:
    1. Shape is square (N, N).
    2. Symmetry: max |K - K^T| < tol.
    3. Unit diagonal: max |K_ii - 1.0| < tol.
    4. Bounds: all entries within [-tol, 1.0 + tol].
    5. Positive Semi-Definiteness (PSD): min eigenvalue >= -tol.
    6. Numerical condition number.

    Returns:
        Dict containing validation metrics and boolean flags.
    """
    K_arr = np.asarray(K, dtype=np.float64)

    if K_arr.ndim != 2 or K_arr.shape[0] != K_arr.shape[1]:
        return {
            "is_square": False,
            "shape": list(K_arr.shape),
            "is_symmetric": False,
            "has_unit_diagonal": False,
            "is_bounded": False,
            "is_psd": False,
            "min_eigenval": None,
            "max_eigenval": None,
            "condition_number": None,
            "max_symmetry_error": None,
            "max_diagonal_error": None,
        }

    N = K_arr.shape[0]
    sym_error = float(np.max(np.abs(K_arr - K_arr.T)))
    diag_error = float(np.max(np.abs(np.diag(K_arr) - 1.0)))
    min_val = float(np.min(K_arr))
    max_val = float(np.max(K_arr))

    # Symmetrize prior to eigenvalue calculation to avoid imaginary artifacts
    K_sym = 0.5 * (K_arr + K_arr.T)
    eigenvalues = np.linalg.eigvalsh(K_sym)
    min_eig = float(np.min(eigenvalues))
    max_eig = float(np.max(eigenvalues))

    # Condition number (handle near-zero min eigenvalue)
    if min_eig > 1e-12:
        cond_num = float(max_eig / min_eig)
    else:
        cond_num = float("inf")

    is_symmetric = bool(sym_error < tol)
    has_unit_diagonal = bool(diag_error < tol)
    is_bounded = bool(min_val >= -tol and max_val <= 1.0 + tol)
    is_psd = bool(min_eig >= -tol)

    return {
        "is_square": True,
        "shape": [N, N],
        "is_symmetric": is_symmetric,
        "has_unit_diagonal": has_unit_diagonal,
        "is_bounded": is_bounded,
        "is_psd": is_psd,
        "min_eigenval": min_eig,
        "max_eigenval": max_eig,
        "condition_number": cond_num,
        "max_symmetry_error": sym_error,
        "max_diagonal_error": diag_error,
        "min_entry": min_val,
        "max_entry": max_val,
    }


def regularize_kernel_matrix(K: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    Ensure strict positive definiteness by adding a small diagonal ridge.
    """
    K_reg = np.copy(K)
    np.fill_diagonal(K_reg, np.diag(K_reg) + ridge)
    return K_reg

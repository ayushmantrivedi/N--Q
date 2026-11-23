"""
NĀḌĪ-Q Architecture A: Dual-Solver Optimization Engine.

Primary path  : Classical Tabu Search (default) or Exact Exhaustive (N ≤ 12).
Secondary path: QAOA via PennyLane, running in a background thread with a
                configurable timeout (default 15 s) and automatic fallback
                to the Tabu result on timeout or error.

QAOA is NEVER in the clinical prediction path.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional

from nadiq.models.metta_quantum.types import (
    CandidateRelationship,
    BiologicalQUBOConfig,
    QuantumSelectionOutput,
)
from nadiq.models.metta_quantum.qubo import BiologicalQUBOBuilder
from nadiq.models.metta_quantum.classical_solvers import ClassicalSolvers

logger = logging.getLogger(__name__)

# QAOA import is deferred so that the engine can still function even if PennyLane
# is unavailable or the thread times out.
_QAOA_AVAILABLE = False
try:
    from nadiq.models.metta_quantum.qaoa import QAOAOptimizer  # type: ignore
    _QAOA_AVAILABLE = True
except ImportError:
    pass


class DualSolverEngine:
    """
    Two-path optimization engine for biological relationship selection.

    Solver mode selection:
      - 'auto'    : Exact if N ≤ 12, else Tabu (default)
      - 'tabu'    : Always Tabu Search
      - 'exact'   : Always Exhaustive (slow for N > 20)
      - 'qaoa_p1' : QAOA p=1 (async background thread, falls back to Tabu)
      - 'qaoa_p2' : QAOA p=2 (async background thread, falls back to Tabu)
    """

    _EXACT_MAX_N = 12  # Above this, exact is impractical

    def __init__(
        self,
        qubo_config: Optional[BiologicalQUBOConfig] = None,
        solver_mode: str = "auto",
        qaoa_timeout_sec: float = 15.0,
        qaoa_n_steps: int = 60,
    ) -> None:
        self.qubo_config = qubo_config or BiologicalQUBOConfig()
        self.solver_mode = solver_mode.lower()
        self.qaoa_timeout_sec = qaoa_timeout_sec
        self.qaoa_n_steps = qaoa_n_steps

        if self.solver_mode not in {"auto", "tabu", "exact", "qaoa_p1", "qaoa_p2"}:
            raise ValueError(f"Unknown solver_mode: '{self.solver_mode}'")

        self._last_solver_used: str = "none"

    @property
    def last_solver_used(self) -> str:
        return self._last_solver_used

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        candidates: List[CandidateRelationship],
        redundancy_map: dict,
        contradiction_pairs: set,
    ) -> QuantumSelectionOutput:
        """
        Solve the relationship-selection QUBO and return a QuantumSelectionOutput.

        QAOA modes run in a daemon thread. If they exceed qaoa_timeout_sec, the
        engine falls back to the Tabu result with an explicit log message.

        Parameters
        ----------
        candidates : List[CandidateRelationship]
        redundancy_map : dict
            (i, j) → redundancy weight.
        contradiction_pairs : set
            Set of (i, j) hard-incompatibility pairs.

        Returns
        -------
        QuantumSelectionOutput
        """
        n = len(candidates)
        if n == 0:
            return self._empty_output()

        # Build QUBO
        builder = BiologicalQUBOBuilder(
            candidates=candidates,
            redundancy_map=redundancy_map,
            contradiction_pairs=contradiction_pairs,
            config=self.qubo_config,
        )
        solver = ClassicalSolvers(builder)

        # Determine effective mode
        effective_mode = self._effective_mode(n)

        t0 = time.perf_counter()

        if effective_mode == "exact":
            result = solver.solve_exhaustive()
            self._last_solver_used = "exact"

        elif effective_mode == "tabu":
            result = solver.solve_tabu_search()
            self._last_solver_used = "tabu"

        elif effective_mode in {"qaoa_p1", "qaoa_p2"}:
            result = self._solve_qaoa_with_fallback(
                builder, solver, effective_mode
            )
        else:
            raise RuntimeError(f"Unexpected effective_mode: '{effective_mode}'")

        elapsed = time.perf_counter() - t0
        logger.info(
            "DualSolverEngine: solver=%s  N=%d  selected=%d  feasible=%s  gap=%.4f  time=%.3fs",
            self._last_solver_used,
            n,
            len(result.selected_indices),
            result.is_feasible,
            result.optimality_gap,
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_mode(self, n: int) -> str:
        if self.solver_mode == "auto":
            return "exact" if n <= self._EXACT_MAX_N else "tabu"
        return self.solver_mode

    def _solve_qaoa_with_fallback(
        self,
        builder: BiologicalQUBOBuilder,
        solver: ClassicalSolvers,
        mode: str,
    ) -> QuantumSelectionOutput:
        """Run QAOA in a daemon thread; fall back to Tabu on timeout or error."""
        if not _QAOA_AVAILABLE:
            logger.warning(
                "QAOA not available (PennyLane import failed). Falling back to Tabu."
            )
            result = solver.solve_tabu_search()
            self._last_solver_used = "tabu (qaoa_unavailable)"
            return result

        # Compute tabu baseline in this thread first (cheap, used as fallback)
        fallback_result = solver.solve_tabu_search()

        # Spin up QAOA in daemon thread
        p_layers = 1 if mode == "qaoa_p1" else 2
        qaoa_result_holder: list = []
        qaoa_error_holder: list = []

        def _run_qaoa():
            try:
                opt = QAOAOptimizer(
                    qubo_builder=builder,
                    p_layers=p_layers,
                    n_steps=self.qaoa_n_steps,
                )
                res = opt.optimize()
                qaoa_result_holder.append(res)
            except Exception as exc:
                qaoa_error_holder.append(exc)

        t = threading.Thread(target=_run_qaoa, daemon=True)
        t0 = time.perf_counter()
        t.start()
        t.join(timeout=self.qaoa_timeout_sec)
        elapsed = time.perf_counter() - t0

        if t.is_alive():
            logger.warning(
                "QAOA thread exceeded timeout (%.1fs after %.1fs). Falling back to Tabu result.",
                self.qaoa_timeout_sec,
                elapsed,
            )
            self._last_solver_used = f"tabu (qaoa_{mode}_timeout)"
            return fallback_result

        if qaoa_error_holder:
            logger.warning(
                "QAOA raised an exception: %s. Falling back to Tabu result.",
                qaoa_error_holder[0],
            )
            self._last_solver_used = f"tabu (qaoa_{mode}_error)"
            return fallback_result

        if qaoa_result_holder:
            self._last_solver_used = mode
            return qaoa_result_holder[0]

        # Unexpected: thread ended but no result
        self._last_solver_used = f"tabu (qaoa_{mode}_no_result)"
        return fallback_result

    @staticmethod
    def _empty_output() -> QuantumSelectionOutput:
        return QuantumSelectionOutput(
            method="none",
            bitstring="",
            selected_indices=[],
            selected_edges=[],
            energy=0.0,
            utility_score=0.0,
            is_feasible=False,
            budget_error=0,
            num_contradictions=0,
            optimality_gap=0.0,
            probability=1.0,
            circuit_depth=0,
            gate_count={},
            n_evaluations=0,
            runtime_sec=0.0,
        )

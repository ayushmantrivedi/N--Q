"""
Leakage Firewall for Nāḍī-Q.

Provides rigid verification layer asserting temporal constraints and
partitioning validity. Breaches are considered fatal errors.
"""

from __future__ import annotations

import logging
from typing import Iterable

from nadiq.data.temporal.domain import OutcomeState, PredictionSnapshot

logger = logging.getLogger(__name__)


class TemporalLeakageError(Exception):
    """Raised when a temporal leakage constraint is violated."""
    pass


class TemporalLeakageChecker:
    """
    Enforces strict chronological and cross-partition boundaries.
    """
    
    @staticmethod
    def verify_snapshot(snapshot: PredictionSnapshot) -> None:
        """
        Verify that a single PredictionSnapshot does not contain future data.
        
        Raises:
            TemporalLeakageError if a violation is detected.
        """
        T = snapshot.snapshot_time
        
        # 1. No feature timestamp > snapshot_time
        for visit in snapshot.available_visits:
            if visit.date > T:
                raise TemporalLeakageError(
                    f"Leakage detected! Patient {snapshot.patient_id} has "
                    f"a visit on {visit.date} included in a snapshot taken at {T}."
                )
                
        # 2. No future diagnosis used as an input feature (implicitly handled by #1)
        # But we also ensure the *actual event* didn't occur <= T if it's POSITIVE.
        # Wait, if they converted TO the target state <= T, then the snapshot shouldn't be predicting it.
        # This is a population inclusion criteria, but from a temporal perspective, 
        # predicting an event that has already occurred is leakage/trivial.
        if snapshot.outcome.state == OutcomeState.POSITIVE and snapshot.outcome.actual_event_date:
            if snapshot.outcome.actual_event_date <= T:
                raise TemporalLeakageError(
                    f"Leakage detected! Patient {snapshot.patient_id} had the outcome event "
                    f"on {snapshot.outcome.actual_event_date}, which is <= snapshot time {T}."
                )

    @staticmethod
    def verify_dataset_partitions(
        train_snapshots: Iterable[PredictionSnapshot],
        val_snapshots: Iterable[PredictionSnapshot],
        test_snapshots: Iterable[PredictionSnapshot]
    ) -> None:
        """
        Verify that patient IDs do not leak across partitions.
        
        Raises:
            TemporalLeakageError if an overlap is detected.
        """
        train_patients = {s.patient_id for s in train_snapshots}
        val_patients = {s.patient_id for s in val_snapshots}
        test_patients = {s.patient_id for s in test_snapshots}
        
        if not train_patients.isdisjoint(test_patients):
            overlap = train_patients.intersection(test_patients)
            raise TemporalLeakageError(
                f"Leakage detected! {len(overlap)} patients overlap between train and test partitions. "
                f"Sample overlap IDs: {list(overlap)[:5]}"
            )
            
        if not train_patients.isdisjoint(val_patients):
            overlap = train_patients.intersection(val_patients)
            raise TemporalLeakageError(
                f"Leakage detected! {len(overlap)} patients overlap between train and val partitions."
            )
            
        if not val_patients.isdisjoint(test_patients):
            overlap = val_patients.intersection(test_patients)
            raise TemporalLeakageError(
                f"Leakage detected! {len(overlap)} patients overlap between val and test partitions."
            )
        
        logger.info("Leakage firewall: Partition overlap checks passed.")

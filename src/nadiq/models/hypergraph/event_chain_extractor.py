"""
NĀḌĪ-Q Phase 10A: Temporal Event Chain Extractor.
Extracts timestamped clinical event transitions and tests recurring sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ClinicalEvent:
    """
    Represents a clinically meaningful threshold-crossing event.
    """
    patient_id: int
    event_type: str  # 'MolElevation', 'CaudateDecline', 'MotorWorsening'
    feature_name: str
    change_value: float
    timestamp_months: float
    confidence: float = 1.0


@dataclass
class EventSequence:
    """
    Represents an ordered sequence of events for a patient.
    """
    patient_id: int
    sequence_type: str  # e.g., 'Mol_then_Struct_then_Func'
    events: List[ClinicalEvent]
    latency_months: float  # Elapsed time between first and last event
    progression_5yr: Optional[int] = None


class EventChainExtractor:
    """
    Identifies discrete, statistically supported biomarker-imaging-motor event chains.
    """

    def __init__(
        self,
        asyn_elevation_threshold: float = 0.5,  # z-score or normalized change
        caudate_decline_threshold: float = -0.5,
        motor_worsening_threshold: float = 5.0  # UPDRS-III MCID
    ):
        self.asyn_elevation_threshold = asyn_elevation_threshold
        self.caudate_decline_threshold = caudate_decline_threshold
        self.motor_worsening_threshold = motor_worsening_threshold

    def extract_patient_events(
        self,
        patient_id: int,
        longitudinal_df: pd.DataFrame
    ) -> List[ClinicalEvent]:
        """
        Extracts timestamped threshold events for a single patient trajectory.
        """
        events: List[ClinicalEvent] = []
        pat_df = longitudinal_df[longitudinal_df["PATNO"] == patient_id].sort_values("timestamp_months")
        if len(pat_df) < 2:
            return events

        # Baseline reference values
        bl_row = pat_df.iloc[0]
        bl_motor = bl_row.get("updrs3_score", np.nan)
        bl_asyn = bl_row.get("asyn", np.nan)
        bl_caudate = bl_row.get("MIA_CAUDATE_BILAT", np.nan)

        for _, row in pat_df.iloc[1:].iterrows():
            t_m = float(row.get("timestamp_months", 0.0))

            # 1. Molecular Elevation Event
            curr_asyn = row.get("asyn", np.nan)
            if not np.isnan(curr_asyn) and not np.isnan(bl_asyn) and bl_asyn > 0:
                rel_change = (curr_asyn - bl_asyn) / bl_asyn
                if rel_change >= self.asyn_elevation_threshold:
                    events.append(ClinicalEvent(
                        patient_id=patient_id, event_type="MolElevation",
                        feature_name="asyn", change_value=float(rel_change),
                        timestamp_months=t_m
                    ))

            # 2. Structural Caudate Decline Event
            curr_caudate = row.get("MIA_CAUDATE_BILAT", np.nan)
            if not np.isnan(curr_caudate) and not np.isnan(bl_caudate) and bl_caudate > 0:
                rel_caudate = (curr_caudate - bl_caudate) / bl_caudate
                if rel_caudate <= self.caudate_decline_threshold:
                    events.append(ClinicalEvent(
                        patient_id=patient_id, event_type="CaudateDecline",
                        feature_name="caudate", change_value=float(rel_caudate),
                        timestamp_months=t_m
                    ))

            # 3. Motor Worsening Event (MCID >= 5 points)
            curr_motor = row.get("updrs3_score", np.nan)
            if not np.isnan(curr_motor) and not np.isnan(bl_motor):
                delta_motor = curr_motor - bl_motor
                if delta_motor >= self.motor_worsening_threshold:
                    events.append(ClinicalEvent(
                        patient_id=patient_id, event_type="MotorWorsening",
                        feature_name="updrs3_score", change_value=float(delta_motor),
                        timestamp_months=t_m
                    ))

        return events

    def analyze_event_sequences(
        self,
        cohort_longitudinal_df: pd.DataFrame,
        patient_targets: Dict[int, int]
    ) -> Dict[str, Any]:
        """
        Analyzes sequence ordering across all patients:
        Tests hypothesis: MolElevation -> CaudateDecline -> MotorWorsening
        vs. alternatives and controls.
        """
        all_patients = cohort_longitudinal_df["PATNO"].unique()
        chain_mol_to_motor = []
        chain_struct_to_motor = []
        chain_full_trimodal = []
        reverse_chain = []

        for p in all_patients:
            evs = self.extract_patient_events(p, cohort_longitudinal_df)
            prog = patient_targets.get(p, 0)

            # Find earliest occurrence of each event type
            mol_evs = [e for e in evs if e.event_type == "MolElevation"]
            struct_evs = [e for e in evs if e.event_type == "CaudateDecline"]
            motor_evs = [e for e in evs if e.event_type == "MotorWorsening"]

            t_mol = min([e.timestamp_months for e in mol_evs]) if len(mol_evs) > 0 else None
            t_struct = min([e.timestamp_months for e in struct_evs]) if len(struct_evs) > 0 else None
            t_motor = min([e.timestamp_months for e in motor_evs]) if len(motor_evs) > 0 else None

            # Check sequences
            # Mol -> Motor
            if t_mol is not None and t_motor is not None and t_mol < t_motor:
                chain_mol_to_motor.append((p, prog, t_motor - t_mol))

            # Struct -> Motor
            if t_struct is not None and t_motor is not None and t_struct < t_motor:
                chain_struct_to_motor.append((p, prog, t_motor - t_struct))

            # Full Trimodal: Mol -> Struct -> Motor
            if t_mol is not None and t_struct is not None and t_motor is not None:
                if t_mol <= t_struct and t_struct <= t_motor:
                    chain_full_trimodal.append((p, prog, t_motor - t_mol))

            # Reverse Control: Motor -> Mol
            if t_mol is not None and t_motor is not None and t_motor < t_mol:
                reverse_chain.append((p, prog, t_mol - t_motor))

        # Compute contingency statistics
        def get_chain_stats(chain_list, total_progressors, total_stable):
            n_present = len(chain_list)
            if n_present == 0:
                return {"count": 0, "prog_rate": 0.0, "odds_ratio": 1.0, "p_val": 1.0, "mean_latency": 0.0}
            
            progs_with_chain = sum(1 for item in chain_list if item[1] == 1)
            stable_with_chain = sum(1 for item in chain_list if item[1] == 0)

            # 2x2 table: [[prog_with, prog_without], [stable_with, stable_without]]
            table = [
                [progs_with_chain, max(0, total_progressors - progs_with_chain)],
                [stable_with_chain, max(0, total_stable - stable_with_chain)]
            ]
            odds_ratio, p_val = stats.fisher_exact(table)
            latencies = [item[2] for item in chain_list]

            return {
                "count": n_present,
                "prog_count": progs_with_chain,
                "stable_count": stable_with_chain,
                "prog_rate": float(progs_with_chain / n_present) if n_present > 0 else 0.0,
                "odds_ratio": float(odds_ratio),
                "p_value": float(p_val),
                "mean_latency_months": float(np.mean(latencies)) if len(latencies) > 0 else 0.0
            }

        total_progs = sum(1 for p in all_patients if patient_targets.get(p, 0) == 1)
        total_stables = sum(1 for p in all_patients if patient_targets.get(p, 0) == 0)

        results = {
            "total_patients": len(all_patients),
            "total_progressors": total_progs,
            "total_stables": total_stables,
            "Mol_precedes_Motor": get_chain_stats(chain_mol_to_motor, total_progs, total_stables),
            "Struct_precedes_Motor": get_chain_stats(chain_struct_to_motor, total_progs, total_stables),
            "Full_Trimodal_Mol_Struct_Motor": get_chain_stats(chain_full_trimodal, total_progs, total_stables),
            "Reverse_Motor_precedes_Mol_Control": get_chain_stats(reverse_chain, total_progs, total_stables),
        }
        return results

"""
NĀḌĪ-Q Phase 10A: Symbolic MeTTa Event Query & Reasoning Layer.
Uses OpenCog Hyperon MeTTa to query discrete clinical event precedence and candidate chains.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import hyperon
from hyperon import MeTTa


class MeTTAEventReasoner:
    """
    Symbolic MeTTa query engine for discrete clinical event sequences.
    Maintains an AtomSpace of observed events, clinical outcomes, and precedence relations.
    """

    def __init__(self):
        self.metta = MeTTa()
        self._init_kb()

    def _init_kb(self):
        # Define types and base axioms
        axioms = """
        (: EventType Type)
        (: MolElevation EventType)
        (: CaudateDecline EventType)
        (: MotorWorsening EventType)

        (: Outcome Type)
        (: Progressor Outcome)
        (: Stable Outcome)

        ;; Define precedence rule: Event1 precedes Event2 if t1 < t2 for the same patient
        (= (precedes $pat $e1 $e2)
           (match &self (, (PatientEvent $pat $e1 $t1 $c1)
                           (PatientEvent $pat $e2 $t2 $c2))
                  (if (< $t1 $t2) True False)))

        ;; Define full trimodal cascade rule: e1 < e2 < e3
        (= (trimodal_cascade $pat)
           (match &self (, (PatientEvent $pat MolElevation $t1 $c1)
                           (PatientEvent $pat CaudateDecline $t2 $c2)
                           (PatientEvent $pat MotorWorsening $t3 $c3))
                  (if (and (< $t1 $t2) (< $t2 $t3)) True False)))
        """
        self.metta.run(axioms)

    def load_events(self, events: List[Dict[str, Any]], patient_outcomes: Dict[int, str]):
        """
        Populates AtomSpace with timestamped patient events.
        """
        statements = []
        for ev in events:
            p = ev["patient_id"]
            e_type = ev["event_type"]
            t_m = int(round(ev["timestamp_months"]))
            conf = float(ev.get("confidence", 1.0))
            stmt = f"(PatientEvent P{p} {e_type} {t_m} {conf:.2f})"
            statements.append(stmt)

        for p, outcome in patient_outcomes.items():
            status = "Progressor" if outcome == 1 or outcome == "Progressor" else "Stable"
            stmt = f"(PatientOutcome P{p} {status})"
            statements.append(stmt)

        code_block = "\n".join(statements)
        self.metta.run(code_block)

    def query_precedence(self, e1: str = "MolElevation", e2: str = "MotorWorsening") -> List[str]:
        """
        Finds all patients where event e1 precedes event e2.
        """
        query = f"!(match &self (, (PatientEvent $pat {e1} $t1 $c1) (PatientEvent $pat {e2} $t2 $c2)) (if (< $t1 $t2) $pat (empty)))"
        res = self.metta.run(query)
        # Parse returned patient tokens
        patients = []
        if len(res) > 0:
            for item in res[0]:
                val = str(item).strip()
                if val.startswith("P") and val not in patients:
                    patients.append(val)
        return patients

    def query_outcome_association(self, precedence_patients: List[str]) -> Dict[str, int]:
        """
        Queries whether patients matching a symbolic sequence are Progressors or Stable.
        """
        counts = {"Progressor": 0, "Stable": 0}
        for pat in precedence_patients:
            q = f"!(match &self (PatientOutcome {pat} $out) $out)"
            res = self.metta.run(q)
            if len(res) > 0 and len(res[0]) > 0:
                out_str = str(res[0][0]).strip()
                if out_str in counts:
                    counts[out_str] += 1
        return counts

    def query_trimodal_cascade(self) -> List[str]:
        """
        Queries patients exhibiting the full MolElevation -> CaudateDecline -> MotorWorsening cascade.
        """
        q = "!(match &self (, (PatientEvent $pat MolElevation $t1 $c1) (PatientEvent $pat CaudateDecline $t2 $c2) (PatientEvent $pat MotorWorsening $t3 $c3)) (if (and (< $t1 $t2) (< $t2 $t3)) $pat (empty)))"
        res = self.metta.run(q)
        patients = []
        if len(res) > 0:
            for item in res[0]:
                val = str(item).strip()
                if val.startswith("P") and val not in patients:
                    patients.append(val)
        return patients

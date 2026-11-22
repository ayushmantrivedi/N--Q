"""
Temporal Domain Model for Nāḍī-Q.

Defines the explicit domain objects for temporal reasoning, guaranteeing
strict separation between measurements available at snapshot_time (X)
and future outcomes (Y).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum, auto
from typing import Any, Optional

import numpy as np


class ClinicalState(Enum):
    """Normalized clinical states. Serves as a registry for current disease tasks."""
    CN = auto()         # Cognitively Normal
    MCI = auto()        # Mild Cognitive Impairment
    AD = auto()         # Alzheimer's Disease (specifically Alzheimer's pathology)
    DEMENTIA = auto()   # Dementia (general/unspecified, explicitly not equated to AD)
    UNKNOWN = auto()    # State cannot be determined or is missing


class HorizonDefinition(Enum):
    """Semantics for resolving future outcomes."""
    EXACT = auto()      # Outcome must be observed exactly at T + H (within a narrow window)
    WITHIN = auto()     # Outcome occurs at any point in (T, T + H]
    AT_LEAST = auto()   # Outcome is known to occur, and we require follow-up of at least H if it doesn't


class OutcomeState(Enum):
    """The resolved state of a future outcome."""
    POSITIVE = auto()   # The target event occurred within the defined horizon rules
    NEGATIVE = auto()   # The target event did not occur, and sufficient follow-up exists to prove it
    CENSORED = auto()   # Follow-up ended before the horizon, or outcome is otherwise unknown


class FeaturePolicy(Enum):
    """Policy for extracting features from the PatientTimeline up to T."""
    LATEST = auto()       # Flattened vector of the most recent measurement for each feature <= T
    TIME_SERIES = auto()  # Full sequence of all visits <= T


@dataclass
class Visit:
    """A single clinical encounter."""
    patient_id: str
    date: date
    raw_state: str
    normalized_state: ClinicalState
    features: dict[str, float]  # Flattened key-value features for this visit
    
    def __post_init__(self):
        if not isinstance(self.date, date):
            raise TypeError("Visit date must be a datetime.date object.")


@dataclass
class PatientTimeline:
    """The complete longitudinal history for a patient."""
    patient_id: str
    visits: list[Visit] = field(default_factory=list)
    
    def __post_init__(self):
        # Enforce chronological ordering
        self.visits.sort(key=lambda v: v.date)
        
    def get_visits_up_to(self, target_date: date) -> list[Visit]:
        """Returns all visits strictly on or before target_date."""
        return [v for v in self.visits if v.date <= target_date]
        
    def get_visits_after(self, target_date: date) -> list[Visit]:
        """Returns all visits strictly after target_date."""
        return [v for v in self.visits if v.date > target_date]


@dataclass
class PredictionHorizon:
    """Definition of the prediction window."""
    months: int
    definition: HorizonDefinition
    window_days: int = 90  # Allowable slack for 'EXACT' matches (e.g., +/- 90 days)


@dataclass
class FutureOutcome:
    """The resolved future state for a given snapshot and horizon."""
    state: OutcomeState
    target_date: date            # The nominal T + H date
    actual_event_date: Optional[date] = None  # The date the event actually occurred (if positive)
    last_followup_date: Optional[date] = None # The last date the patient was seen
    actual_state: Optional[ClinicalState] = None


@dataclass
class TemporalLabel:
    """The formatted supervision target for modeling."""
    value: float             # 1.0 for POSITIVE, 0.0 for NEGATIVE (NaN for CENSORED)
    is_censored: bool
    time_to_event_days: Optional[int] = None


@dataclass
class PredictionSnapshot:
    """
    A strictly left-truncated temporal snapshot for prediction.
    
    X_{<=T} -> Y_{T+H}
    """
    snapshot_id: str
    patient_id: str
    snapshot_time: date
    available_visits: list[Visit]  # MUST ONLY contain visits <= snapshot_time
    feature_policy: FeaturePolicy
    horizon: PredictionHorizon
    outcome: FutureOutcome
    label: TemporalLabel

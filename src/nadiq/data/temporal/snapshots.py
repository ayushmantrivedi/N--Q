"""
Snapshot Generation for Nāḍī-Q.

Constructs left-truncated prediction snapshots (X_{<=T}) and evaluates
future outcomes (Y_{T+H}) according to strict horizon semantics.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import Optional

from nadiq.data.temporal.domain import (
    ClinicalState,
    FeaturePolicy,
    FutureOutcome,
    HorizonDefinition,
    OutcomeState,
    PatientTimeline,
    PredictionHorizon,
    PredictionSnapshot,
    TemporalLabel,
)


def resolve_future_outcome(
    timeline: PatientTimeline,
    snapshot_date: date,
    horizon: PredictionHorizon,
    target_state: ClinicalState
) -> FutureOutcome:
    """
    Evaluates the patient's future visits to determine the outcome.
    
    Args:
        timeline: The full patient timeline.
        snapshot_date: The time T.
        horizon: The prediction horizon H and semantics.
        target_state: The target diagnosis we are trying to predict.
        
    Returns:
        A FutureOutcome object containing the resolved state.
    """
    future_visits = timeline.get_visits_after(snapshot_date)
    
    target_date = snapshot_date + timedelta(days=int(horizon.months * 30.4375))
    
    if not future_visits:
        return FutureOutcome(
            state=OutcomeState.CENSORED,
            target_date=target_date,
            actual_event_date=None,
            last_followup_date=None,
            actual_state=None
        )
        
    last_followup = max(v.date for v in future_visits)
    
    if horizon.definition == HorizonDefinition.EXACT:
        # Must find a visit within target_date +/- window_days
        window_start = target_date - timedelta(days=horizon.window_days)
        window_end = target_date + timedelta(days=horizon.window_days)
        
        valid_visits = [v for v in future_visits if window_start <= v.date <= window_end]
        
        if not valid_visits:
            return FutureOutcome(
                state=OutcomeState.CENSORED,
                target_date=target_date,
                actual_event_date=None,
                last_followup_date=last_followup,
                actual_state=None
            )
            
        # Take the visit closest to target_date
        closest_visit = min(valid_visits, key=lambda v: abs((v.date - target_date).days))
        
        if closest_visit.normalized_state == target_state:
            state = OutcomeState.POSITIVE
        elif closest_visit.normalized_state == ClinicalState.UNKNOWN:
            state = OutcomeState.CENSORED
        else:
            state = OutcomeState.NEGATIVE
            
        return FutureOutcome(
            state=state,
            target_date=target_date,
            actual_event_date=closest_visit.date if state == OutcomeState.POSITIVE else None,
            last_followup_date=last_followup,
            actual_state=closest_visit.normalized_state
        )
        
    elif horizon.definition == HorizonDefinition.WITHIN:
        # Event happens anywhere in (T, T+H]
        # To call it NEGATIVE, we need at least one visit >= T+H
        
        for v in future_visits:
            if v.date <= target_date and v.normalized_state == target_state:
                return FutureOutcome(
                    state=OutcomeState.POSITIVE,
                    target_date=target_date,
                    actual_event_date=v.date,
                    last_followup_date=last_followup,
                    actual_state=v.normalized_state
                )
                
        # If we get here, no positive event occurred within the horizon.
        # Do we have sufficient follow-up to prove it?
        if last_followup >= target_date:
            # We followed them past the horizon, and they didn't convert
            return FutureOutcome(
                state=OutcomeState.NEGATIVE,
                target_date=target_date,
                actual_event_date=None,
                last_followup_date=last_followup,
                actual_state=next((v.normalized_state for v in reversed(future_visits) if v.date <= target_date), None)
            )
        else:
            # We lost them before the horizon
            return FutureOutcome(
                state=OutcomeState.CENSORED,
                target_date=target_date,
                actual_event_date=None,
                last_followup_date=last_followup,
                actual_state=None
            )
            
    elif horizon.definition == HorizonDefinition.AT_LEAST:
        # Event happens anywhere > T. 
        # If no event, we require last_followup >= target_date
        for v in future_visits:
            if v.normalized_state == target_state:
                return FutureOutcome(
                    state=OutcomeState.POSITIVE,
                    target_date=target_date,
                    actual_event_date=v.date,
                    last_followup_date=last_followup,
                    actual_state=v.normalized_state
                )
                
        if last_followup >= target_date:
            return FutureOutcome(
                state=OutcomeState.NEGATIVE,
                target_date=target_date,
                actual_event_date=None,
                last_followup_date=last_followup,
                actual_state=future_visits[-1].normalized_state
            )
        else:
            return FutureOutcome(
                state=OutcomeState.CENSORED,
                target_date=target_date,
                actual_event_date=None,
                last_followup_date=last_followup,
                actual_state=None
            )
            
    raise ValueError(f"Unknown horizon definition: {horizon.definition}")


def generate_snapshot(
    timeline: PatientTimeline,
    snapshot_date: date,
    horizon: PredictionHorizon,
    target_state: ClinicalState,
    feature_policy: FeaturePolicy = FeaturePolicy.LATEST
) -> Optional[PredictionSnapshot]:
    """
    Constructs a PredictionSnapshot for a specific patient at time T.
    
    Args:
        timeline: The complete patient timeline.
        snapshot_date: Time T.
        horizon: The prediction horizon H.
        target_state: The target diagnosis.
        feature_policy: How to represent X_{<=T}.
        
    Returns:
        A PredictionSnapshot, or None if no visits are available at or before T.
    """
    past_visits = timeline.get_visits_up_to(snapshot_date)
    
    if not past_visits:
        return None
        
    outcome = resolve_future_outcome(timeline, snapshot_date, horizon, target_state)
    
    # Construct Label
    if outcome.state == OutcomeState.POSITIVE:
        val = 1.0
        censored = False
        if outcome.actual_event_date is not None:
            t_to_event = (outcome.actual_event_date - snapshot_date).days
        else:
            t_to_event = None
    elif outcome.state == OutcomeState.NEGATIVE:
        val = 0.0
        censored = False
        t_to_event = (outcome.last_followup_date - snapshot_date).days if outcome.last_followup_date else None
    else:
        val = float('nan')
        censored = True
        t_to_event = (outcome.last_followup_date - snapshot_date).days if outcome.last_followup_date else None
        
    label = TemporalLabel(value=val, is_censored=censored, time_to_event_days=t_to_event)
    
    return PredictionSnapshot(
        snapshot_id=str(uuid.uuid4()),
        patient_id=timeline.patient_id,
        snapshot_time=snapshot_date,
        available_visits=past_visits,
        feature_policy=feature_policy,
        horizon=horizon,
        outcome=outcome,
        label=label
    )
